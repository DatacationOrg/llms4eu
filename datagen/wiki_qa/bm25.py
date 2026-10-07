"""Temporary: BM25 over all wiki pages (text, lowercased \\w+ tokens, no stemming, all languages in one index), to
measure how hard a question is for keyword search: the rank of its own page. Build once (bm25_index/), then
rank(questions, page_ids) -> ranks (1 = top; RANK_CAP + 1 when not in the top RANK_CAP).
Also distinctive(text): the article's highest-idf frequent words, what BM25 would match on.
Usage: bm25.py build | bm25.py test"""

import json
import math
import re
import sys

import bm25s
import numpy as np

from bunny import PAGES

DIR = "bm25_index"
RANK_CAP = 100


def tok(s):
    return re.findall(r"\w+", s.lower())


def build():
    ids, docs, df = [], [], {}
    for line in open(PAGES):
        p = json.loads(line)
        ids.append(p["id"])
        docs.append(tok(p["title"] + " " + p["text"]))
        for w in set(docs[-1]):
            df[w] = df.get(w, 0) + 1
    r = bm25s.BM25()
    r.index(docs)
    r.save(DIR)
    json.dump(ids, open(f"{DIR}/ids.json", "w"))
    json.dump({w: n for w, n in df.items() if n > 1}, open(f"{DIR}/df.json", "w"))
    print(len(ids), "pages indexed")


_r, _ids, _pos, _df = None, None, None, None


def load():
    global _r, _ids, _pos, _df
    if _r is None:
        _r = bm25s.BM25.load(DIR)
        _ids = json.load(open(f"{DIR}/ids.json"))
        _pos = {i: n for n, i in enumerate(_ids)}
        _df = json.load(
            open(f"{DIR}/df.json")
        )  # words in a single page are left out: df 1
    return _r


def rank(questions, page_ids):
    r = load()
    docs, _ = r.retrieve(
        [tok(q) or ["_"] for q in questions], k=RANK_CAP, show_progress=False
    )
    return [
        int(np.where(row == _pos[pid])[0][0]) + 1 if _pos[pid] in row else RANK_CAP + 1
        for row, pid in zip(docs, page_ids)
    ]


def distinctive(text, n=25):
    """Frequent words of the article that are rare in the corpus (high tf * idf): what keyword search keys on."""
    load()
    tf = {}
    for w in tok(text):
        if len(w) > 2 and not w.isdigit():
            tf[w] = tf.get(w, 0) + 1

    def idf(w):
        return math.log(len(_ids) / _df.get(w, 1))

    # only words in < 200 pages: function words of a smaller language are "rare" in a mostly Swedish corpus
    rare = [w for w in tf if _df.get(w, 1) < 200]
    return sorted(rare, key=lambda w: -idf(w) * math.log(1 + tf[w]))[:n]


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        load()
        pid = _ids[0]
        p = next(json.loads(line) for line in open(PAGES))
        print(
            pid,
            p["title"],
            rank([p["title"], "castle"], [pid, pid]),
            distinctive(p["text"])[:10],
        )
