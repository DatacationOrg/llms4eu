"""Temporary: agent batches for the question-data cleaning pass (new rules + question_en).
Source: teacher items in questions_pool2.db (gen_questions.py output, ids = urls).
Writes fix2/<src>_NN/batch.jsonl rows {url, lang, title, country, text, items}; skips urls already batched.
Usage: fix2_batches.py [--size 40]"""

import argparse
import glob
import json
import os
import sqlite3

S = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    pool = {
        r["url"]: r
        for f in ("label_pool.json", "label_pool2.json", "label_pool3.json")
        for r in json.load(open(f"{S}/{f}"))
    }
    db = sqlite3.connect(f"{S}/questions_pool2.db")
    items = {
        u: json.loads(i)
        for u, i in db.execute("select id, items from questions where error is null")
    }
    batched = {json.loads(line)["url"] for line in open(f"{S}/qg_clean.jsonl")} | {
        json.loads(line)["url"]
        for p in glob.glob(f"{S}/fix2/*/batch.jsonl")
        for line in open(p)
    }
    todo = [u for u in items if u not in batched]
    n0 = len(glob.glob(f"{S}/fix2/batch_*"))
    for k in range(0, len(todo), a.size):
        d = f"{S}/fix2/batch_{n0 + k // a.size:02d}"
        os.makedirs(d)
        with open(f"{d}/batch.jsonl", "w") as f:
            for u in todo[k : k + a.size]:
                r = pool[u]
                row = {
                    "url": u,
                    **{x: r[x] for x in ("lang", "title", "country", "text")},
                    "items": items[u],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(d, len(todo[k : k + a.size]))


if __name__ == "__main__":
    main()
