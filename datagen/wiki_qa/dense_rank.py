"""Temporary: question -> page check for challenging items. Embeds every page ("passage: " + title + first 1000 chars)
and every challenging question ("query: ") with nvidia/Nemotron-3-Embed-1B-BF16 (first 512 dims, re-normalized),
then per question: dense rank of its own page among all pages and the margin (own cosine - best other page).
Validated against Sonnet's unique/ambiguous judgements (chcheck*_results.json), so we know whether dense rank (with
BM25 rank) predicts ambiguity. GPU only when free (others < 2 GB). Writes dense_rank.json {"pid|j": [rank, margin]}.
Usage: dense_rank.py pages | rank | rag (questions of table answers, plus BM25 -> rag_rank.json)"""

import json
import sqlite3
import sys

import numpy as np

from bunny import PAGES
from near_dups import gpu_busy

DIM = 512


def model():
    if gpu_busy():
        sys.exit("GPU in use by someone else: not starting")
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(
        "nvidia/Nemotron-3-Embed-1B-BF16",
        device="cuda",
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    m.max_seq_length = 512  # a few runaway generated questions are huge (OOM at 4 GB per batch otherwise)
    return m


def norm(e):
    e = e[:, :DIM].astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def pages():
    ids, texts = [], []
    for line in open(PAGES):
        p = json.loads(line)
        ids.append(p["id"])
        texts.append(f"passage: {p['title']}\n{p['text'][:1000]}")
    e = model().encode(texts, batch_size=128, show_progress_bar=False)
    np.savez("page_emb.npz", ids=np.array(ids), emb=norm(e).astype(np.float16))
    print("pages", len(ids), flush=True)


def ranks(keys, qs):
    """keys: (page id, tag) per question -> {"pid|tag": [dense rank of the page, margin to the best other page]}."""
    import torch

    d = np.load("page_emb.npz")
    pos = {i: n for n, i in enumerate(d["ids"])}
    q = torch.tensor(
        norm(model().encode(["query: " + x for x in qs], batch_size=256)), device="cuda"
    ).half()
    P = torch.tensor(d["emb"], device="cuda")
    out = {}
    for s in range(0, len(keys), 2048):
        sim = q[s : s + 2048] @ P.T
        own = torch.tensor([pos[pid] for pid, _ in keys[s : s + 2048]], device="cuda")
        own_sim = sim.gather(1, own[:, None])
        r = (sim > own_sim).sum(1) + 1
        sim.scatter_(1, own[:, None], -1)
        margin = own_sim[:, 0] - sim.max(1).values
        for (pid, j), rr, m in zip(keys[s : s + 2048], r.tolist(), margin.tolist()):
            out[f"{pid}|{j}"] = [rr, round(m, 4)]
    ranks = np.array([v[0] for v in out.values()])
    print(
        len(out),
        "questions; dense top1",
        f"{np.mean(ranks == 1):.0%}",
        "top10",
        f"{np.mean(ranks <= 10):.0%}",
    )
    return out


def rank():
    con = sqlite3.connect("wiki_qa.db")
    keys, qs = [], []
    for pid, its in con.execute(
        "select id, items from challenge_items where items is not null"
    ):
        for j, it in enumerate(json.loads(its)):
            keys.append((pid, j))
            qs.append(it["question"])
    out = ranks(keys, qs)
    json.dump(out, open("dense_rank.json", "w"))
    validate(out)


def rag():
    """Dense and BM25 rank of the own page for every question in table answers, original (n|o) and cross-lingual
    (n|x) -> rag_rank.json {"pid|n|o": [dense rank, margin, bm25 rank]}."""
    import bm25

    con = sqlite3.connect("wiki_qa.db")
    keys, qs = [], []
    for pid, its in con.execute(
        "select id, items from answers where items is not null"
    ):
        for n, it in enumerate(json.loads(its)):
            keys += [(pid, f"{n}|o"), (pid, f"{n}|x")]
            qs += [it["question"], it["question_x"]]
    out = ranks(keys, qs)
    for k, b in zip(out, bm25.rank(qs, [pid for pid, _ in keys])):
        out[k].append(b)
    json.dump(out, open("rag_rank.json", "w"))


def validate(out):
    """Does dense rank / margin predict Sonnet's 'ambiguous' on the 250 checked challenging items?"""
    rows = []
    for f, key_dir in (
        ("chcheck_results.json", "chcheck"),
        ("chcheck2_results.json", "chcheck2"),
    ):
        for d, items in json.load(open(f)):
            pid = json.load(open(f"{d}/key.json"))["id"]
            for j, it in enumerate(items):
                if f"{pid}|{j}" in out:
                    rows.append((out[f"{pid}|{j}"], it["points_to_place"] != "unique"))
    for name, pred in (
        ("dense rank > 1", lambda v: v[0] > 1),
        ("dense rank > 3", lambda v: v[0] > 3),
        ("margin < 0.02", lambda v: v[1] < 0.02),
    ):
        tp = sum(pred(v) and a for v, a in rows)
        fp = sum(pred(v) and not a for v, a in rows)
        fn = sum(not pred(v) and a for v, a in rows)
        print(
            f"{name:15} n={len(rows)} ambiguous={sum(a for _, a in rows)} precision {tp / max(tp + fp, 1):.2f} recall {tp / max(tp + fn, 1):.2f}"
        )


def bm25_only():
    """CPU refresh of rag_rank.json: BM25 rank for every current question (o, x); the dense rank and margin are kept
    only for answers rows unchanged since the last dense run (older than the file), else null (needs the GPU)."""
    import datetime
    import os

    import bm25

    old = json.load(open("rag_rank.json"))
    t0 = datetime.datetime.fromtimestamp(
        os.path.getmtime("rag_rank.json"), datetime.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect("wiki_qa.db")
    keys, qs, fresh = [], [], []
    for pid, its, t in con.execute(
        "select id, items, created_at from answers where items is not null"
    ):
        for n, it in enumerate(json.loads(its)):
            keys += [f"{pid}|{n}|o", f"{pid}|{n}|x"]
            qs += [it["question"], it["question_x"]]
            fresh += [t < t0] * 2
    pids = [k.split("|")[0] for k in keys]
    out = {}
    for s in range(0, len(qs), 20000):  # bm25s batches; progress per chunk
        for k, f, b in zip(
            keys[s : s + 20000],
            fresh[s : s + 20000],
            bm25.rank(qs[s : s + 20000], pids[s : s + 20000]),
        ):
            d = old.get(k) if f else None
            out[k] = [d[0], d[1], b] if d else [None, None, b]
        print(s + 20000, "ranked", flush=True)
    json.dump(out, open("rag_rank.json", "w"))
    print(len(out), "keys,", sum(v[0] is not None for v in out.values()), "with dense")


if __name__ == "__main__":
    {"pages": pages, "rank": rank, "rag": rag, "bm25": bm25_only}[sys.argv[1]]()
