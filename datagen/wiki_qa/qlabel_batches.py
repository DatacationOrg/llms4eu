"""Temporary: sample 600 non-empty pages from wiki_qa.db (both models, every language, sqrt-balanced) for agent
quality labelling: 500 train + 100 val pages. Writes qlabel/b_NN/batch.jsonl rows
{id, split, model, lang, title, country, text, items}; text = what the model saw (Qwen 6000, teacher 20000 chars)."""

import collections
import json
import os
import random
import sqlite3

S = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(5)
db = sqlite3.connect(f"{S}/wiki_qa.db", timeout=120)
got = {
    pid: (m, json.loads(i))
    for pid, m, i in db.execute(
        "select id, model, items from questions where error is null"
    )
}
by = collections.defaultdict(list)
for line in open("/data/llms4eu/wiki/pages.jsonl"):
    r = json.loads(line)
    if r["id"] in got and got[r["id"]][1]:
        m, its = got[r["id"]]
        m = "qwen" if m.startswith("qwen") else "teacher"
        cut = 6000 if m == "qwen" else 20000
        by[(m, r["in_language"])].append(
            {
                "id": r["id"],
                "model": m,
                "lang": r["in_language"],
                "title": r["title"],
                "country": r["country"],
                "text": r["text"][:cut],
                "items": its,
            }
        )
w = {k: len(v) ** 0.5 for k, v in by.items()}
tot = sum(w.values())
picked = []
for k, v in by.items():
    picked += rng.sample(v, min(len(v), max(3, round(620 * w[k] / tot))))
rng.shuffle(picked)
picked = picked[:600]
for n, r in enumerate(picked):
    r["split"] = "val" if n < 100 else "train"
for k in range(0, len(picked), 35):
    d = f"{S}/qlabel/b_{k // 35:02d}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/batch.jsonl", "w") as f:
        f.writelines(
            json.dumps(r, ensure_ascii=False) + "\n" for r in picked[k : k + 35]
        )
print(
    len(picked),
    "pages,",
    sum(len(r["items"]) for r in picked),
    "items,",
    (len(picked) + 34) // 35,
    "batches",
)
print(collections.Counter((r["model"], r["lang"]) for r in picked).most_common())
