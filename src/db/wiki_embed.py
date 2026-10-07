"""Embed every chunk (title + breadcrumb + text) into
`embeddings/<model>/<size>.npy`: float16 unit vectors, row i = chunk i of
`load(Chunk, size=size)`, NaN until embedded. Resumable: only NaN rows are (re)done.

    uv run python -m src.db.wiki_embed qwen3-embedding-0.6b    # local GPU
    uv run python -m src.db.wiki_embed nemotron-3-embed-1b     # OpenRouter, free
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.db.schemas.chunk import represent
from src.db.wiki_chunks import SIZES
from src.db.wiki_qa import ROOT, Chunk, load
from src.shared.env import load_local_env

# name -> (model, dimensions, document prefix)
MODELS = {
    "qwen3-embedding-0.6b": ("Qwen/Qwen3-Embedding-0.6B", 1024, ""),
    "nemotron-3-embed-1b": ("nvidia/nemotron-3-embed-1b:free", 2048, "passage: "),
}
WINDOW = 8192  # rows embedded between flushes to disk


def vectors_file(name: str, size: int) -> np.ndarray:
    """The memory-mapped vectors file, created full of NaN on first use."""
    path = ROOT / "embeddings" / name / f"{size}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r+")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load(Chunk, ["id"], size=size).num_rows
    out = np.lib.format.open_memmap(path, "w+", np.float16, (rows, MODELS[name][1]))
    out[:] = np.nan
    return out


def local(model_id: str):
    """Batches of texts -> unit vectors on the GPU, longest first so batches pad little."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    )
    model.max_seq_length = 4096  # 2048-token chunks plus title and breadcrumb

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=32, normalize_embeddings=True)

    return embed


def openrouter(model_id: str, workers: int = 8):
    """Batches of texts -> unit vectors from OpenRouter, `workers` requests at a time."""
    import httpx

    client = httpx.Client(
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=300,
    )

    def request(texts: list[str]) -> np.ndarray:
        attempt = 0
        while attempt < 8:
            try:
                r = client.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    json={"model": model_id, "input": texts},
                )
                if (
                    "free-models-per-day" in r.text
                ):  # 1000 requests a day: wait for the reset
                    reset = int(
                        r.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"]
                    )
                    print(
                        "daily limit, waiting until",
                        time.ctime(reset / 1000),
                        flush=True,
                    )
                    time.sleep(max(60.0, reset / 1000 - time.time() + 60))
                    continue
                if r.status_code in (400, 413) and len(texts) > 1:  # too big: halve it
                    half = len(texts) // 2
                    return np.concatenate(
                        [request(texts[:half]), request(texts[half:])]
                    )
                r.raise_for_status()
                v = np.array([d["embedding"] for d in r.json()["data"]], np.float32)
                return v / np.linalg.norm(v, axis=1, keepdims=True)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                print("retry", attempt, type(e).__name__, str(e)[:120], flush=True)
                attempt += 1
                time.sleep(30 * attempt)
        raise RuntimeError("OpenRouter kept failing; rerun to resume")

    pool = ThreadPoolExecutor(workers)

    def embed(texts: list[str]) -> np.ndarray:
        # The endpoint takes at most 256 inputs and between 1 and 2 MB of body per request.
        batches, batch, used = [], [], 0
        for text in texts:
            cost = len(json.dumps(text).encode()) + 1
            if batch and (len(batch) == 256 or used + cost > 1_000_000):
                batches.append(batch)
                batch, used = [], 0
            batch.append(text)
            used += cost
        return np.concatenate(list(pool.map(request, [*batches, batch])))

    return embed


def run(name: str, sizes: list[int]) -> None:
    model_id, _, prefix = MODELS[name]
    embed = openrouter(model_id) if model_id.endswith(":free") else local(model_id)
    for size in sizes:
        out = vectors_file(name, size)
        todo = np.flatnonzero(
            np.isnan(out[:, 0]) | np.isnan(out[:, -1])
        )  # -1: cut mid-row
        print(
            f"{name} {size}: {len(out) - len(todo)} done, {len(todo)} to do", flush=True
        )
        if not len(todo):
            continue
        chunks = load(Chunk, ["title", "breadcrumb", "text"], size=size).to_pydict()
        texts = [prefix + represent(*c) for c in zip(*chunks.values())]
        todo = todo[np.argsort([-len(texts[i]) for i in todo], kind="stable")]
        for start in range(0, len(todo), WINDOW):
            rows = todo[start : start + WINDOW]
            out[rows] = embed([texts[i] for i in rows]).astype(np.float16)
            out.flush()
            print(f"{name} {size}: {start + len(rows)}/{len(todo)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()
    load_local_env()
    run(args.model, args.sizes)


if __name__ == "__main__":
    main()
