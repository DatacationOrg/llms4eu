"""Temporary: Space Bunny as the teacher for the question-quality labels (qspec), one page per request with its full
article and all its items. Labels only, no confidences. A reply that breaks qspec.consistency is retried once with
the errors. Output: table bunny_labels (id, items = label dicts in item order, error) or, with --val, the jsonl
qlabel_bunny_val.jsonl for the 100 gold val pages (qlabel2/val_a_*).
Needs OPENROUTER_API_KEY. Usage: bunny_label.py --val | --ids ids.txt  [--workers 12]"""

import argparse
import glob
import json
import os
import sqlite3
from typing import Literal

from pydantic import BaseModel, create_model, model_validator

import bunny
from qspec import SPEC, consistency

Label = create_model(
    "Label",
    **{
        k: (bool if kind == "bool" else Literal[tuple(opts)], ...)
        for k, (kind, _, opts) in SPEC.items()
    },
)


class Page(BaseModel):
    labels: list[Label]

    @model_validator(mode="before")
    @classmethod
    def unwrap(
        cls, v
    ):  # Bunny sometimes returns [{"item": 0, "labels": {...}}, ...] or a bare list
        v = {"labels": v} if isinstance(v, list) else v
        if isinstance(v, dict) and isinstance(v.get("labels"), list):
            v["labels"] = [
                x["labels"]
                if isinstance(x, dict) and isinstance(x.get("labels"), dict)
                else x
                for x in v["labels"]
            ]
        return v


GUIDE = (
    open("qlabel/GUIDE.md").read().split("Label every item")[1].split("\nOutput:")[0]
)
PROMPT = (
    """You label generated RAG test questions for quality. The input is one Wikipedia article about a European place
and its generated items: question (article language), question_en, facts (should answer it), query (English search
keywords). A chat user who has not seen the page asks the question; a search system must find this page among ~100k
European place pages.

Label every item"""
    + GUIDE
    + '\nReturn {"labels": [...]} with exactly one label object per item, in item order.'
)


def message(page, items):
    return (
        f"Article ({page['in_language']}, {page['country']}): {page['title']}\n\n{page['text'][:60000]}\n\n"
        + "Items:\n"
        + "\n".join(
            f"{j}: {json.dumps(it, ensure_ascii=False)}" for j, it in enumerate(items)
        )
    )


def label(chain, page, items):
    msgs = [("system", PROMPT), ("user", message(page, items))]
    for _ in range(2):
        out = None
        for _ in range(3):  # unparsable replies are common on long pages
            out = out or bunny.call(chain, msgs)
        if out is None:
            return None, "request failed"
        labs = [lb.model_dump() for lb in out.labels]
        errs = (
            [f"need {len(items)} label objects, got {len(labs)}"]
            if len(labs) != len(items)
            else consistency(page["id"], items, labs)
        )
        if not errs:
            return labs, None
        msgs += [
            ("assistant", out.model_dump_json()),
            ("user", "Fix these and return all labels again:\n" + "\n".join(errs)),
        ]
    return labs, "; ".join(errs)[:500]


def val_pages():
    for d in sorted(glob.glob("qlabel2/val_a_*")):
        p = json.loads(open(f"{d}/batch.jsonl").readline())
        p |= {"in_language": p["lang"]}
        yield p, p["items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--ids")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--effort",
        help="OpenRouter reasoning effort (low/medium/high); default: the model's",
    )
    ap.add_argument("--out", default="qlabel_bunny_val.jsonl", help="--val output file")
    a = ap.parse_args()
    chain = bunny.llm(Page, temperature=0.0, effort=a.effort)
    if a.val:
        f = a.out
        ok = (
            {r["id"] for r in map(json.loads, open(f)) if r["labels"]}
            if os.path.exists(f)
            else set()
        )
        todo, out = [t for t in val_pages() if t[0]["id"] not in ok], open(f, "a")

        def save(p, labs, err):
            out.write(json.dumps({"id": p["id"], "labels": labs, "error": err}) + "\n")
            out.flush()
    else:
        con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
        con.execute(
            "create table if not exists bunny_labels (id text primary key, items text, error text, created_at text default current_timestamp)"
        )
        done = {
            r[0] for r in con.execute("select id from bunny_labels where error is null")
        }
        ids = {i.strip() for i in open(a.ids)} - done
        todo = list(bunny.pages(ids))
        print(len(todo), "pages to label", flush=True)

        def save(p, labs, err):
            # insert or replace is safe: rows without an error are never redone
            con.execute(
                "insert or replace into bunny_labels (id, items, error, model) values (?, ?, ?, ?)",
                (p["id"], json.dumps(labs) if labs else None, err, bunny.MODEL_ID),
            )
            con.commit()

    def run(t):
        labs, err = label(chain, *t)
        return t[0], labs, err

    for n, (p, labs, err) in enumerate(bunny.run_all(run, todo, a.workers), 1):
        if bunny.STOP.is_set() and labs is None:
            continue
        save(p, labs, err)
        if n % 100 == 0:
            print(n, "done", flush=True)
    print("finished", "(stopped on 429)" if bunny.STOP.is_set() else "", flush=True)


if __name__ == "__main__":
    main()
