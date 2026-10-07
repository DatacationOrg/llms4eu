"""Temporary: class distribution of each quality label in a label table, per generating model.
Usage: label_dist.py [table=bunny_labels]"""

import collections
import json
import sqlite3
import sys

from qspec import IDS

t = sys.argv[1] if len(sys.argv) > 1 else "bunny_labels"
con = sqlite3.connect("wiki_qa.db")
model = dict(con.execute("select id, model from questions"))
cnt = collections.defaultdict(collections.Counter)
n = collections.Counter()
for i, its in con.execute(f"select id, items from {t} where items is not null"):
    g = "qwen" if model[i].startswith("qwen") else "bunny"
    for lab in json.loads(its):
        n[g] += 1
        for k in IDS:
            cnt[(g, k)][str(lab[k])] += 1
print("items", dict(n))
for k in IDS:
    print(
        f"{k:21}",
        " | ".join(
            f"{g}: "
            + " ".join(f"{c} {v / n[g]:.0%}" for c, v in sorted(cnt[(g, k)].items()))
            for g in n
        ),
    )
