"""Temporary: blind pairwise folders for the Qwen-vs-Bunny language comparison: langcmp/<n>/page.json with the
article and two item sets A/B in random order. The key (which set is which model) goes to langcmp_key.json, outside
the agents' folders. Usage: langcmp_prep.py"""

import json
import os
import random
import sqlite3

from bunny import PAGES

qwen = sqlite3.connect("wiki_qa.db")
regen = sqlite3.connect("bunny_regen.db")
ids = [i.strip() for i in open("langcmp_ids.txt")]
got = dict(regen.execute("select id, items from questions where error is null"))
rng = random.Random(3)
key = {}
n = 0
for line in open(PAGES):
    p = json.loads(line)
    if p["id"] not in got:
        continue
    sets = {
        "qwen": json.loads(
            qwen.execute(
                "select items from questions where id = ?", (p["id"],)
            ).fetchone()[0]
        ),
        "bunny": json.loads(got[p["id"]]),
    }
    order = rng.sample(["qwen", "bunny"], 2)
    d = f"langcmp/{n:03d}"
    os.makedirs(d, exist_ok=True)
    page = {k: p[k] for k in ("title", "in_language", "country")}  # no id: stays blind
    page |= {"text": p["text"][:8000], "A": sets[order[0]], "B": sets[order[1]]}
    json.dump(page, open(f"{d}/page.json", "w"), ensure_ascii=False, indent=1)
    key[d] = {"id": p["id"], "lang": p["in_language"], "A": order[0], "B": order[1]}
    n += 1
json.dump(key, open("langcmp_key.json", "w"), indent=1)
print(n, "folders")
