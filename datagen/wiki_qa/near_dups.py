"""Temporary: near-duplicate questions across pages (a question that also fits another page cannot single out its
own). Embeds every final corpus question (build_clean.rows) and every challenging question with
nvidia/Nemotron-3-Embed-1B-BF16 ("query: " prefix, first 512 of 2048 dims, re-normalized), then exact blockwise
nearest neighbour among questions of OTHER pages. Writes near_dups.npz (keys, best other-page sim and its index) and
prints the similarity distribution plus sample pairs per band, to choose the flag threshold by eye. Before the
search the embeddings are mean-centred over the whole set and re-normalized (batch norm).
GPU only when free (refuses to start if another process uses > 2 GB). Usage: near_dups.py embed | near_dups.py knn"""

import json
import sqlite3
import subprocess
import sys

import numpy as np

import build_clean

OUT = "near_dups.npz"
DIM = 512


def questions():
    keys, texts = [], []
    for r in build_clean.rows():
        keys.append(f"q|{r['id']}|{r['j']}")
        texts.append(r["item"]["question"])
    con = sqlite3.connect("wiki_qa.db")
    for pid, its in con.execute(
        "select id, items from challenge_items where items is not null"
    ):
        for j, it in enumerate(json.loads(its)):
            keys.append(f"c|{pid}|{j}")
            texts.append(it["question"])
    return keys, texts


def gpu_busy():
    used = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    return sum(map(int, used)) > 2048


def embed():
    if gpu_busy():
        sys.exit("GPU in use by someone else: not starting")
    from sentence_transformers import SentenceTransformer

    keys, texts = questions()
    print(len(texts), "questions", flush=True)
    m = SentenceTransformer(
        "nvidia/Nemotron-3-Embed-1B-BF16",
        device="cuda",
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    e = m.encode(
        ["query: " + t for t in texts],
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    e = e[:, :DIM]
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.savez(OUT, keys=np.array(keys), emb=e.astype(np.float16))
    print("saved", e.shape, flush=True)


def knn(block=4096):
    import torch

    d = np.load(OUT)
    keys = d["keys"]
    page = np.array([k.split("|")[1] for k in keys])
    _, inv = np.unique(page, return_inverse=True)
    dev = "cuda" if torch.cuda.is_available() and not gpu_busy() else "cpu"
    # batch norm over the whole set: subtract the mean embedding and re-normalize, so the direction all these
    # place questions share ("which lake in ...") doesn't inflate every cosine
    x = d["emb"].astype(np.float32)
    x -= x.mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    e = torch.tensor(
        x, dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev
    )
    pg = torch.tensor(inv, device=dev)
    best_sim, best_idx = np.zeros(len(keys), np.float32), np.zeros(len(keys), np.int64)
    for s in range(0, len(keys), block):
        sim = e[s : s + block] @ e.T
        sim[pg[s : s + block, None] == pg[None, :]] = -1  # same page never counts
        v, i = sim.max(1)
        best_sim[s : s + block], best_idx[s : s + block] = (
            v.float().cpu().numpy(),
            i.cpu().numpy(),
        )
    np.savez(OUT, keys=keys, emb=d["emb"], best_sim=best_sim, best_idx=best_idx)
    print("device", dev, "distribution of best other-page similarity:")
    for t in (0.8, 0.85, 0.9, 0.95, 0.98):
        print(f"  >= {t}: {np.mean(best_sim >= t):.2%}")
    texts = dict(zip(*questions()))
    rng = np.random.default_rng(0)
    for lo, hi in ((0.85, 0.9), (0.9, 0.95), (0.95, 1.01)):
        idx = np.where((best_sim >= lo) & (best_sim < hi))[0]
        print(f"\n--- band {lo}-{hi}: {len(idx)} questions")
        for n in rng.choice(idx, min(8, len(idx)), replace=False):
            a, b = keys[n], keys[best_idx[n]]
            print(
                f"{best_sim[n]:.3f} {a.split('|')[1]} | {texts[a]}\n      {b.split('|')[1]} | {texts[b]}"
            )


if __name__ == "__main__":
    {"embed": embed, "knn": knn}[sys.argv[1]]()
