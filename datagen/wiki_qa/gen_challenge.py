"""Temporary: challenging items that keyword search (BM25) struggles with: vague, paraphrased questions about one
page ("Which lake in eastern Finland is known for ...?", "In which country is the castle that ...?") that avoid the
article's distinctive words (bm25.distinctive) and its title, with a challenging answer: a short synthesized answer
that names the place and combines 2-3 facts (plus the facts as evidence). 5 per page, languages round-robin, runs
until stopped. Each item gets its measured BM25 rank of the source page (1 = trivial, 101 = not in the top 100).
Each page's items are then labelled by the qspec labeller (bunny_label), stored as item["labels"], so training can
keep verdict == keep only. They never replace anything. Output table challenge_items (id, items, error). Needs OPENROUTER_API_KEY.
Usage: gen_challenge.py [--rpm 12] [--workers 10] [--limit N]"""

import argparse
import collections
import itertools
import json
import random
import sqlite3
import zlib

from pydantic import BaseModel, Field, field_validator, model_validator

import bm25
import bunny
from bunny_label import Page, label

QA_KEYS = (
    "question",
    "question_en",
    "answer",
    "facts",
    "query",
)  # the answer names the place


class Item(BaseModel):
    question: str
    question_en: str
    answer: str
    facts: list[str] = Field(min_length=1, max_length=3)
    query: str

    @field_validator("query", mode="before")
    @classmethod
    def join_keywords(cls, v):  # Bunny sometimes returns the keywords as a list
        return " ".join(map(str, v)) if isinstance(v, list) else v


class Items(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=5)

    @model_validator(mode="before")
    @classmethod
    def wrap_bare_list(cls, v):  # Bunny often returns the bare list
        return {"items": v} if isinstance(v, list) else v


PROMPT = """You write CHALLENGING test questions for a search system over ~100k Wikipedia pages about European places.
Keyword search finds a page easily when the question repeats its name or rare words, so these questions must not.
Write 5 items about the article below. Each question:
- is vague and paraphrased, the way someone asks who half-remembers the place or only knows what it is like:
  "Which lake in eastern Finland is said to have ...?", "In which country is the hilltop castle that was ...?",
  "Which nature reserve on the Danube protects ...?". It may ask for the place itself, its country or region, or
  a property, but the answer must come from this article.
- never uses the place's name, nor any of the FORBIDDEN words given below (they are the words keyword search keys
  on). Use general words, synonyms and descriptions instead.
- must still single out THIS place: always include one locator (country, region or nearest large town/river/range,
  unless it is a forbidden word) plus one or two properties from the article that together fit no other place.
  Ambiguous is worse than easy: if many lakes/castles in that region could match, add a more distinctive property.
- stands alone: the asker starts with no context (has not seen the article and knows the collection holds many
  pages), so never "the article", "this place", "here", "the text" or a bare "it"; describe the place instead.
- asks ONE thing; never a list of measurements, never two questions joined by "and".
- is natural, grammatical and correctly spelled in the article's language; no codes, coordinates or register numbers.
Each item: question, question_en (faithful English translation), answer (the challenging answer: 1-3 sentences in
the article's language that name the place and explain using 2-3 facts from the article), facts (1-3 short facts
from the article supporting the answer), query (at most 8 English keywords from the question only, no name, no
answer). Never invent facts. Return {"items": [...]}."""


def message(p):
    forbidden = bm25.distinctive(p["text"]) + bm25.tok(p["title"])
    return (
        f"Article ({p['in_language']}, {p['country']}): {p['title']}\n\n{p['text'][:60000]}\n\n"
        f"FORBIDDEN words: {', '.join(dict.fromkeys(forbidden))}"
    )


def queue(done):
    """Pages with >= 1500 chars, round-robin over languages (shuffled inside each language)."""
    by = collections.defaultdict(list)
    for line in open(bunny.PAGES):
        p = json.loads(line)
        if p["char_count"] >= 1500 and p["id"] not in done:
            by[p["in_language"]].append(p["id"])
    rng = random.Random(4)
    for v in by.values():
        rng.shuffle(v)
    return [i for grp in itertools.zip_longest(*by.values()) for i in grp if i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--part",
        type=int,
        choices=[0, 1],
        help="half of the queue by id hash (two generators, both language-balanced)",
    )
    a = ap.parse_args()
    bunny.RPM = a.rpm
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    con.execute(
        "create table if not exists challenge_items (id text primary key, items text, error text, created_at text default current_timestamp)"
    )
    done = {
        r[0] for r in con.execute("select id from challenge_items where error is null")
    }
    ids = [  # a stable half by id hash: restarts never shift pages between the two generators
        i for i in queue(done) if a.part is None or zlib.crc32(i.encode()) % 2 == a.part
    ]
    ids = ids[: a.limit]
    order = {i: n for n, i in enumerate(ids)}
    print(len(ids), "pages queued", flush=True)
    chain = bunny.llm(Items, temperature=0.6)
    judge = bunny.llm(
        Page, temperature=0.0
    )  # the qspec quality labeller, as a filter before training
    bm25.load()

    def run(p):
        msg = message(p)
        for _ in range(3):
            out = bunny.call(chain, [("system", PROMPT), ("user", msg)])
            if out:
                its = [it.model_dump() for it in out.items]
                labs, _ = label(judge, p, [{k: it[k] for k in QA_KEYS} for it in its])
                for it, lb in zip(its, labs or [None] * len(its)):
                    it["labels"] = lb  # None when labelling failed
                    it["labels_model"] = bunny.MODEL_ID
                return p["id"], its, None
        return p["id"], None, "request failed"

    # backfill first: v2 pages generated before the built-in labelling get their labels (items unchanged)
    unlabelled = {
        i: json.loads(its)
        for i, its in con.execute(
            "select id, items from challenge_items where items is not null"
        )
        if json.loads(its)[0].get("prompt") in ("v2", "v3")
        and any("labels" not in it for it in json.loads(its))
    }
    print(len(unlabelled), "v2 pages to backfill labels", flush=True)

    def backfill(p):
        its = unlabelled[p["id"]]
        labs, _ = label(judge, p, [{k: it[k] for k in QA_KEYS} for it in its])
        return p["id"], its, labs

    todo = [p for p, _ in bunny.pages(set(unlabelled))]
    for n, (pid, its, labs) in enumerate(bunny.run_all(backfill, todo, a.workers), 1):
        if labs is None:
            continue  # stays unlabelled; the next start retries it
        for it, lb in zip(its, labs):
            it["labels"], it["labels_model"] = lb, bunny.MODEL_ID
        con.execute(
            "update challenge_items set items = ? where id = ?",
            (json.dumps(its, ensure_ascii=False), pid),
        )
        con.commit()
        if n % 50 == 0:
            print(n, "backfilled", flush=True)

    pages = sorted(
        (
            p
            for p, _ in ((json.loads(line), 0) for line in open(bunny.PAGES))
            if p["id"] in order
        ),
        key=lambda p: order[p["id"]],
    )
    for n, (pid, its, err) in enumerate(bunny.run_all(run, pages, a.workers), 1):
        if its is None and bunny.STOP.is_set():
            continue
        if its:
            for it, r in zip(
                its, bm25.rank([it["question"] for it in its], [pid] * len(its))
            ):
                it |= {
                    "challenging": True,
                    "bm25_rank": r,
                    "prompt": "v3",
                }  # v1 items have no "prompt"; v3 adds the no-context rule
        con.execute(
            "insert or replace into challenge_items (id, items, error, model) values (?, ?, ?, ?)",
            (
                pid,
                json.dumps(its, ensure_ascii=False) if its else None,
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
