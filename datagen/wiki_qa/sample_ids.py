"""Temporary: stratified page sample for teacher labels: up to N pages per (language, generating model), half of
them rule-flagged pages (table checks) so the rarer bad classes show up, excluding the gold val pages.
Usage: sample_ids.py <out.txt> [per_cell=80]"""

import collections
import glob
import json
import random
import sqlite3
import sys

con = sqlite3.connect("wiki_qa.db")
val = {
    json.loads(open(f"{d}/batch.jsonl").readline())["id"]
    for d in glob.glob("qlabel2/val_a_*")
}
flagged = {
    i for i, its in con.execute("select id, items from checks") if any(json.loads(its))
}
cells = collections.defaultdict(lambda: ([], []))
for i, m in con.execute("select id, model from questions where error is null"):
    if i not in val:
        cells[(i.split("wiki/")[0], m)][i in flagged].append(i)

per = int(sys.argv[2]) if len(sys.argv) > 2 else 80
rng = random.Random(11)
out = []
for key in sorted(cells):
    clean, bad = (sorted(x) for x in cells[key])
    nb = min(len(bad), per // 2)
    nc = min(len(clean), per - nb)
    nb = min(len(bad), per - nc)
    out += rng.sample(bad, nb) + rng.sample(clean, nc)
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(len(out), "pages")
