"""Temporary: Space Bunny writes the item kinds the corpus lacks (teacher labels: 96-98% retrieval-easy): per page one
`described` item (the place is described, not named, yet only it fits) and one `multi` item (needs facts from
several parts of the article). ~20 rich pages per language, so ≤1k items, for training a Qwen LoRA; they never
replace anything. Output table extra_items (id, items, error). Needs OPENROUTER_API_KEY.
Usage: gen_hard.py [per_lang=20] [--rpm 25]"""

import argparse
import collections
import json
import random
import sqlite3

from typing import Literal

from pydantic import BaseModel, Field, model_validator

import bunny
from gen_questions import QA


class HardQA(QA):
    kind: Literal["described", "multi"]


class Hard(BaseModel):
    items: list[HardQA] = Field(min_length=2, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def wrap_bare_list(
        cls, v
    ):  # Bunny often returns the bare list, and names the field "type"
        v = {"items": v} if isinstance(v, list) else v
        for it in v.get("items", []) if isinstance(v, dict) else []:
            if isinstance(it, dict) and "kind" not in it and "type" in it:
                it["kind"] = it.pop("type")
        return v


PROMPT = """You write hard test items for a search system over ~100k Wikipedia pages about places in Europe.
Write two items about the article below, {"items": [...]}, each with: kind ("described" or "multi"), question (in the article's language, natural and grammatical,
what a curious visitor would ask), question_en (faithful English translation), facts (1-3 short facts from the
article that answer it), query (English search keywords from the question only, never the answer).
- described: do NOT name the place. Describe it the way a person who half-remembers it would: its type, the
  region, and one or two distinguishing details, so that only this one place fits among all European places.
  Short and natural, at most 25 words, e.g. "Which ruined castle above the Moselle near Cochem was rebuilt in
  neo-Gothic style?". Never stack many details. The query must not contain the name either.
- multi: name the place and ask ONE natural question whose answer needs facts from two or three different parts of
  the article (e.g. how its history explains what visitors see today). Not two questions joined by "and".
Queries: at most 8 English keywords.
Never ask about codes, coordinates, register numbers or other database fields. Never invent facts."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_lang", type=int, nargs="?", default=20)
    ap.add_argument("--rpm", type=int, default=25)
    ap.add_argument(
        "--redo",
        action="store_true",
        help="also redo done pages (replaced by the new output)",
    )
    a = ap.parse_args()
    bunny.RPM = a.rpm
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    con.execute(
        "create table if not exists extra_items (id text primary key, items text, error text, created_at text default current_timestamp)"
    )
    done = (
        set()
        if a.redo
        else {
            r[0] for r in con.execute("select id from extra_items where error is null")
        }
    )
    good = {  # teacher-labelled pages with at least two keep items
        i
        for i, its in con.execute(
            "select id, items from bunny_labels where items is not null"
        )
        if sum(lb["verdict"] == "keep" for lb in json.loads(its)) >= 2
    }
    by_lang = collections.defaultdict(list)
    for p, items in bunny.pages(good - done):
        if p["char_count"] >= 3000:
            by_lang[p["in_language"]].append((p, items))
    rng = random.Random(2)
    todo = [t for v in by_lang.values() for t in rng.sample(v, min(a.per_lang, len(v)))]
    print(
        len(todo),
        "pages",
        {k: min(a.per_lang, len(v)) for k, v in sorted(by_lang.items())},
        flush=True,
    )
    chain = bunny.llm(Hard, temperature=0.5)

    def run(t):
        p, items = t
        msg = (
            f"Article ({p['in_language']}, {p['country']}): {p['title']}\n\n{p['text'][:60000]}\n\n"
            "Existing questions (ask something else):\n"
            + "\n".join(it["question"] for it in items)
        )
        for _ in range(3):
            out = bunny.call(chain, [("system", PROMPT), ("user", msg)])
            if out:
                if {it.kind for it in out.items} == {"described", "multi"}:
                    return p["id"], [it.model_dump() for it in out.items], None
        return p["id"], None, "request failed"

    for n, (pid, its, err) in enumerate(bunny.run_all(run, todo, 10), 1):
        if its is None and bunny.STOP.is_set():
            continue
        con.execute(
            "insert or replace into extra_items (id, items, error, model) values (?, ?, ?, ?)",
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
