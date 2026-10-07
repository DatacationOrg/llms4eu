"""Temporary: priority order of corpus pages for Bunny label+repair (teacher sample and gold val first excluded):
1 rule-flagged pages (not counting padded), 2 Qwen pages outside Swedish lakes, 3 Bunny pages, 4 Swedish-lake
pages (huge, thin, low value). Inside a tier: shuffled, so every language advances at once.
Usage: make_queue.py <out.txt>"""

import glob
import json
import random
import sqlite3
import sys

from bunny import PAGES

con = sqlite3.connect("wiki_qa.db")
skip = {i.strip() for i in open("teacher_ids.txt")} | {
    json.loads(open(f"{d}/batch.jsonl").readline())["id"]
    for d in glob.glob("qlabel2/val_a_*")
}
model = dict(con.execute("select id, model from questions where error is null"))
flagged = {
    i
    for i, its in con.execute("select id, items from checks")
    if any(set(f) - {"padded"} for f in json.loads(its))
}
tiers = [[], [], [], []]
for line in open(PAGES):
    r = json.loads(line)
    i = r["id"]
    if i not in model or i in skip:
        continue
    se_lake = r["country"] == "SE" and "lake" in r["categories"]
    t = 0 if i in flagged else 3 if se_lake else 1 if model[i].startswith("qwen") else 2
    tiers[t].append(i)
rng = random.Random(1)
for t in tiers:
    rng.shuffle(t)
open(sys.argv[1], "w").write("\n".join(i for t in tiers for i in t) + "\n")
print([len(t) for t in tiers])
