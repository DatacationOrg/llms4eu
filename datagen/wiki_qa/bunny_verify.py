"""Temporary: relabel repaired pages with the repaired items substituted (Bunny teacher, same prompt as
bunny_label.py), so every repair is checked before it replaces anything. Output table repair_labels (id, items =
label dicts for the repaired page, in item order, error). Needs OPENROUTER_API_KEY.
Usage: bunny_verify.py [--ids ids.txt] [--workers 30]"""

import argparse
import json
import sqlite3

import bunny
from bunny_label import Page, label


def repaired(items, reps):
    out = list(items)
    for r in reps:
        out[r["j"]] = r["item"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    con.execute(
        "create table if not exists repair_labels (id text primary key, items text, error text, created_at text default current_timestamp)"
    )
    done = {
        r[0] for r in con.execute("select id from repair_labels where error is null")
    }
    reps = {
        i: json.loads(its)
        for i, its in con.execute(
            "select id, items from repairs where items is not null"
        )
        if i not in done
    }
    if a.ids:
        reps = {i: v for i, v in reps.items() if i in {x.strip() for x in open(a.ids)}}
    todo = [(p, repaired(its, reps[p["id"]])) for p, its in bunny.pages(set(reps))]
    print(len(todo), "pages to verify", flush=True)
    chain = bunny.llm(Page, temperature=0.0)

    def run(t):
        labs, err = label(chain, *t)
        return t[0]["id"], labs, err

    for n, (pid, labs, err) in enumerate(bunny.run_all(run, todo, a.workers), 1):
        if labs is None and bunny.STOP.is_set():
            continue
        con.execute(  # rows without an error are never redone, so replace is safe
            "insert or replace into repair_labels (id, items, error, model) values (?, ?, ?, ?)",
            (pid, json.dumps(labs) if labs else None, err, bunny.MODEL_ID),
        )
        con.commit()
        if n % 100 == 0:
            print(n, "done", flush=True)
    print("finished", "(stopped on 429)" if bunny.STOP.is_set() else "", flush=True)


if __name__ == "__main__":
    main()
