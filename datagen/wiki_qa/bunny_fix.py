"""Temporary: Space Bunny repairs a page's items given their quality labels: `fix` items are corrected (only the
flagged problems), `drop` items are replaced by a new item on an aspect no other item covers, so nothing is lost
without a replacement. One request per page with any non-keep item. Originals stay in `questions`; output table
repairs (id, items = [{j, action, item}], error). Labels come from a table with (id, items = label dicts).
Needs OPENROUTER_API_KEY. Usage: bunny_fix.py [--labels bunny_labels] [--ids ids.txt] [--workers 8]"""

import argparse
import json
import sqlite3

from pydantic import BaseModel, Field

import bunny
from gen_questions import PROMPT as QG_PROMPT
from gen_questions import QA

# label -> problem description shown to the rewriter
PROBLEM = {
    "duplicate": (True, "asks the same thing as another item on the page"),
    "language_ok": (False, "the question is not in the article's language"),
    "fluent": (
        False,
        "the question is not fluent or grammatical (check inflection of names)",
    ),
    "realistic": (
        False,
        "nobody would really ask this (register/database trivia or circular)",
    ),
    "self_answering": (True, "the question gives away its own answer"),
    "unique_place": (
        False,
        "the question does not single out this place among ~100k European places",
    ),
    "facts_answer": (False, "the facts do not answer the question"),
    "translation_ok": (False, "question_en is not a faithful translation"),
    "query_ok": (
        False,
        "the query is not English keywords from the question only (no answer words)",
    ),
}


def problems(lab):
    out = [txt for k, (bad, txt) in PROBLEM.items() if lab[k] == bad]
    if lab["facts_supported"] != "yes":
        out.append(f"facts not supported by the article ({lab['facts_supported']})")
    return out


class Repair(BaseModel):
    j: int
    item: QA


class Repairs(BaseModel):
    repairs: list[Repair] = Field(default_factory=list)


PROMPT = (
    """You repair generated RAG test items for one Wikipedia article about a European place. The original rules for
writing the items were:

"""
    + QG_PROMPT
    + """

You get all items of the page, and for some of them a task:
- FIX: correct exactly the listed problems and keep everything else (same information need where possible).
- REPLACE: the item is unusable; write a new item following the rules, about an aspect of the place that no
  other item on the page covers, with facts stated in the article.
Return {"repairs": [{"j": <item index>, "item": {...}}]}, one entry per FIX or REPLACE task, nothing else."""
)


def message(page, items, labs):
    lines = []
    for j, (it, lab) in enumerate(zip(items, labs)):
        lines.append(f"{j}: {json.dumps(it, ensure_ascii=False)}")
        if lab["verdict"] == "fix":
            lines.append(f"   FIX: {'; '.join(problems(lab)) or 'small edit needed'}")
        elif lab["verdict"] == "drop":
            lines.append(f"   REPLACE (was: {'; '.join(problems(lab)) or 'unusable'})")
    return (
        f"Article ({page['in_language']}, {page['country']}): {page['title']}\n\n{page['text'][:60000]}\n\n"
        + "Items:\n"
        + "\n".join(lines)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="bunny_labels")
    ap.add_argument("--ids")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    con.execute(
        "create table if not exists repairs (id text primary key, items text, error text, created_at text default current_timestamp)"
    )
    done = {r[0] for r in con.execute("select id from repairs where error is null")}
    labels = {
        i: json.loads(its)
        for i, its in con.execute(
            f"select id, items from {a.labels} where items is not null"
        )
        if i not in done and any(lb["verdict"] != "keep" for lb in json.loads(its))
    }
    if a.ids:
        labels = {
            i: v for i, v in labels.items() if i in {x.strip() for x in open(a.ids)}
        }
    todo = [(p, its, labels[p["id"]]) for p, its in bunny.pages(set(labels))]
    todo = [t for t in todo if len(t[1]) == len(t[2])]
    print(len(todo), "pages to repair", flush=True)
    chain = bunny.llm(Repairs, temperature=0.3)

    def run(t):
        page, items, labs = t
        want = {
            j: lb["verdict"] for j, lb in enumerate(labs) if lb["verdict"] != "keep"
        }
        out = bunny.call(
            chain, [("system", PROMPT), ("user", message(page, items, labs))]
        )
        if out is None:
            return page["id"], None, "request failed"
        got = [
            {"j": r.j, "action": want[r.j], "item": r.item.model_dump()}
            for r in out.repairs
            if r.j in want
        ]
        missing = set(want) - {r["j"] for r in got}
        return page["id"], got, f"missing {sorted(missing)}" if missing else None

    for n, (pid, got, err) in enumerate(bunny.run_all(run, todo, a.workers), 1):
        if got is None and bunny.STOP.is_set():
            continue
        con.execute(
            "insert or replace into repairs (id, items, error, model) values (?, ?, ?, ?)",
            (
                pid,
                json.dumps(got, ensure_ascii=False) if got is not None else None,
                err,
                bunny.MODEL_ID,
            ),
        )
        con.commit()
        if n % 50 == 0:
            print(n, "done", flush=True)
    print("finished", "(stopped on 429)" if bunny.STOP.is_set() else "", flush=True)


if __name__ == "__main__":
    main()
