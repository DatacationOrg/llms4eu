"""Temporary: answers, evidence and a cross-lingual version for every kept item, for RAG indexing tests. Items: a
page's kept corpus items (verdict keep, not ambiguous_across_pages) and kept challenging items (challenge_ok, not
likely_ambiguous).
  gen    one call per page: per item an answer (copied if the item has one), 1-3 evidence sentences copied verbatim
         from the article (located -> char spans; `verbatim` false if any is not found), a question type, and the
         question + answer translated into one other EU language (balanced by a hash of page|item).
         -> table answers (id, items, error, model)
  judge  one call per page: the qspec.RAG criteria per item with context = its evidence (what a chunk retriever
         would return), plus x_ok (the translation is faithful and fluent). -> table answer_labels (id, items, error,
         model). Run with the other model than gen, so no model grades its own text.
Model per process: CLEAN_MODEL=ling|bunny. Usage: gen_rag.py gen|judge [--limit N] [--workers 40] [--ids file] [--out table]"""

import argparse
import re
import collections
import json
import sqlite3
import zlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

import bunny
from qspec import RAG

EU = "bg cs da de el en es et fi fr ga hr hu it lt lv mt nl pl pt ro sk sl sv".split()
QTYPES = Literal[
    "identify", "location", "number", "date", "name", "description", "reason"
]


class Ans(BaseModel):
    j: int
    answer: str
    evidence: list[str] = Field(min_length=1, max_length=3)
    qtype: QTYPES
    question_x: str
    answer_x: str


class Judged(BaseModel):
    j: int
    context_relevant: bool
    answerable: bool
    answer_relevant: bool
    answer_supported: bool
    answer_complete: bool
    language_match: bool
    answer_fluent: bool
    x_ok: bool


class Answers(BaseModel):
    items: list[Ans]

    @model_validator(mode="before")
    @classmethod
    def wrap_bare_list(cls, v):  # free models often return the bare list
        return {"items": v} if isinstance(v, list) else v


class Judgements(Answers):
    items: list[Judged]


GEN_PROMPT = """You prepare a RAG test set over Wikipedia pages about European places. For each item below and the
article, return one object with:
- j: the item number
- answer: if the item has an answer, copy it; else write one: 1-2 sentences in the article's language that answer the
  question using only the article.
- evidence: 1-3 sentences copied EXACTLY, character for character, from the article (no paraphrase, no ellipsis),
  that together state every fact in the answer; the answer must not use facts outside the evidence and the title.
- qtype: what the question asks for: identify (which place is meant), location, number, date, name (a person,
  organisation or other name), description (what something is like), reason (why or how something happened).
- question_x, answer_x: faithful, fluent translations of the question and the answer into the TARGET language of the
  item; keep names, inflect them correctly. The translated question must stand alone like the original.
Return {"items": [...]}, one object per item."""

JUDGE_PROMPT = (
    """You check a RAG test set. Each item has a question, an answer, a context (the page title and the evidence a
retriever would return) and a translation (question_x, answer_x) into another language. Judge the criteria only from the context,
never from your knowledge:
"""
    + "\n".join(f"- {k}: {v}" for k, v in RAG.items())
    + """
- x_ok: Are question_x and answer_x faithful, fluent translations of question and answer into language x?
Return {"items": [...]}, one object per item with j and every criterion as true/false."""
)


def target(pid, j, lang):
    others = [x for x in EU if x != lang]
    return others[zlib.crc32(f"{pid}|{j}".encode()) % len(others)]


def kept_items():
    """page id -> kept items, each with kind corpus|challenge and its index src_j in its source table."""
    import build_clean

    out = collections.defaultdict(list)
    for r in build_clean.rows():
        if (
            r["labels"]
            and r["labels"]["verdict"] == "keep"
            and not r["ambiguous_across_pages"]
        ):
            out[r["id"]].append(r["item"] | {"kind": "corpus", "src_j": r["j"]})
    for r in build_clean.challenge_rows():
        if r["challenge_ok"] and not r["likely_ambiguous"]:
            out[r["id"]].append(r["item"] | {"kind": "challenge", "src_j": r["j"]})
    return out


MARKUP = re.compile(r"\*+|\[\d+\]|\s+")


def plain(s):
    """s without Markdown emphasis and [n] footnotes, whitespace collapsed; pos[i] = offset in s of plain char i."""
    out, pos, last = [], [], 0
    for m in MARKUP.finditer(s + " "):
        out += s[last : m.start()]
        pos += range(last, m.start())
        if m.group()[0].isspace() and out[-1:] != [" "]:
            out.append(" ")
            pos.append(m.start())
        last = m.end()
    return "".join(out), pos


def spans(text, quotes):
    """Char spans in text of the quotes, ignoring markup; ok = every quote found."""
    t, pos = plain(text)
    res = []
    for q in quotes:
        q = plain(q)[0].strip()
        s = t.find(q) if q else -1
        if s >= 0:
            res.append([pos[s], pos[s + len(q) - 1] + 1])
    return res, len(res) == len(quotes)


def gen_message(p, items):
    lines = [
        json.dumps(
            {
                "j": j,
                "question": it["question"],
                "facts": it["facts"],
                "answer": it.get("answer"),
                "TARGET": target(p["id"], j, p["in_language"]),
            },
            ensure_ascii=False,
        )
        for j, it in enumerate(items)
    ]
    return (
        f"Article ({p['in_language']}): {p['title']}\n\n{p['text'][:60000]}\n\nItems:\n"
        + "\n".join(lines)
    )


def judge_message(title, items):
    return "\n".join(
        json.dumps(
            {
                "j": j,
                "question": it["question"],
                "answer": it["answer"],
                "context": f"{title}: " + " ".join(it["evidence"]),
                "x": it["x_lang"],
                "question_x": it["question_x"],
                "answer_x": it["answer_x"],
            },
            ensure_ascii=False,
        )
        for j, it in enumerate(items)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "judge"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ids")
    ap.add_argument(
        "--out",
        default="answer_labels",
        help="judge table (a second judge writes its own)",
    )
    a = ap.parse_args()
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    src, table = "answers", a.out
    for t in (src, table):
        con.execute(
            f"create table if not exists {t} (id text primary key, items text, error text, model text, created_at text default current_timestamp)"
        )
    out_table = src if a.step == "gen" else table
    # done = a row without error that is still current: answers cover exactly the page's kept questions (new labels,
    # repairs and hardening change them), and labels are newer than their answers row
    made = {
        i: (set(it["question"] for it in json.loads(x)), t)
        for i, x, t in con.execute(
            "select id, items, created_at from answers where error is null"
        )
    }
    judged = dict(
        con.execute(f"select id, created_at from {table} where error is null")
    )
    want = set(open(a.ids).read().split()) if a.ids else None

    if a.step == "gen":
        todo = {
            pid: its
            for pid, its in kept_items().items()
            if made.get(pid, (None,))[0] != {it["question"] for it in its}
            and (want is None or pid in want)
        }
        chain, prompt = bunny.llm(Answers, temperature=0.3), GEN_PROMPT
    else:
        todo = {
            pid: json.loads(its)
            for pid, its, t in con.execute(  # its own created_at: gen writes meanwhile
                "select id, items, created_at from answers where error is null"
            )
            if judged.get(pid, "") < t and (want is None or pid in want)
        }
        chain, prompt = bunny.llm(Judgements, temperature=0.0), JUDGE_PROMPT
    ids = set(sorted(todo, key=lambda i: zlib.crc32(i.encode()))[: a.limit])
    print(len(ids), "pages to", a.step, flush=True)

    def run(p):
        its = todo[p["id"]]
        msg = gen_message(p, its) if a.step == "gen" else judge_message(p["title"], its)
        for _ in range(3):
            out = bunny.call(chain, [("system", prompt), ("user", msg)])
            if out and sorted(x.j for x in out.items) == list(range(len(its))):
                res = sorted(out.items, key=lambda x: x.j)
                if a.step == "judge":
                    return p["id"], [x.model_dump(exclude={"j"}) for x in res], None
                rows = []
                for j, (it, x) in enumerate(zip(its, res)):
                    sp, ok = spans(p["text"], x.evidence)
                    rows.append(
                        {k: it[k] for k in ("kind", "src_j", "question")}
                        | x.model_dump(exclude={"j"})
                        | {
                            "x_lang": target(p["id"], j, p["in_language"]),
                            "spans": sp,
                            "verbatim": ok,
                        }
                    )
                return p["id"], rows, None
        return p["id"], None, "request failed or wrong item count"

    pages = [p for p in map(json.loads, open(bunny.PAGES)) if p["id"] in ids]
    for n, (pid, res, err) in enumerate(bunny.run_all(run, pages, a.workers), 1):
        if res is None and bunny.STOP.is_set():
            continue
        con.execute(
            f"insert or replace into {out_table} (id, items, error, model) values (?, ?, ?, ?)",
            (
                pid,
                json.dumps(res, ensure_ascii=False) if res else None,
                err,
                bunny.MODEL_ID,
            ),
        )
        con.commit()
        if n % 100 == 0:
            print(n, "done", flush=True)
    print("finished", "(stopped on 429)" if bunny.STOP.is_set() else "", flush=True)


if __name__ == "__main__":
    main()
