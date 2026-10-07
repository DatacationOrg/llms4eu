"""Temporary: compare two labellers on the gold val pages (qlabel2/val_a_*, one-article Sonnet A): macro-F1 of each
against gold, their agreement, and on every disagreement which one gold sides with.
Usage: compare_labellers.py qlabel_bunny_val.jsonl qlabel_ling_val.jsonl"""

import glob
import json
import sys
from collections import Counter

from qspec import IDS, options


def load(f):
    out = {}
    for r in map(json.loads, open(f)):
        if r.get("labels"):
            out.update({(r["id"], j): lb for j, lb in enumerate(r["labels"])})
    return out


def macro_f1(pred, gold, k):
    f1s = []
    for c in options(k):
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


gold = {}
for d in glob.glob("qlabel2/val_a_*"):
    r = json.loads(open(f"{d}/labels.jsonl").readline())
    gold.update({(r["id"], j): lb for j, lb in enumerate(r["labels"])})
a_name, b_name = sys.argv[1], sys.argv[2]
a, b = load(a_name), load(b_name)
keys = [k for k in gold if k in a and k in b]
print(f"{len(keys)} items labelled by both ({a_name} vs {b_name})")
print(
    f"{'label':21} {'A F1':>5} {'B F1':>5} {'agree':>6}  on disagreement gold sides with A / B / neither"
)
for k in IDS:
    g = [gold[x][k] for x in keys]
    pa, pb = [a[x][k] for x in keys], [b[x][k] for x in keys]
    dis = Counter(
        "A" if va == vg else "B" if vb == vg else "neither"
        for va, vb, vg in zip(pa, pb, g)
        if va != vb
    )
    agree = sum(va == vb for va, vb in zip(pa, pb)) / len(keys)
    print(
        f"{k:21} {macro_f1(pa, g, k):5.2f} {macro_f1(pb, g, k):5.2f} {agree:6.0%}  "
        f"{dis['A']:3} / {dis['B']:3} / {dis['neither']:3}"
    )
