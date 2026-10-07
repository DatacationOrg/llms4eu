"""Embed the chunks (title + breadcrumb + text) of every size with one provider into
`embeddings/<provider>/<size>.npy`: float16 unit vectors, row i = chunk i of
`load(Chunk, size=size)`, NaN until embedded. Resumable: only NaN rows are (re)done, so
after a rechunk only the changed chunks are embedded again.

    uv run python -m src.indexing --method qwen           # local GPU
    uv run python -m src.indexing --method nemotron --api # same model, free on OpenRouter
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import numpy as np

from src.db.dataset import SIZES, Chunk, load, missing, vectors_path
from src.db.schemas.chunk import represent
from src.indexing.embedders import CONFIG, load_embedder
from src.shared.env import load_local_env

WINDOW = 8192  # rows embedded between flushes to disk
# OpenRouter's embedding endpoint takes at most this many inputs, and 1-2 MB, per request.
API_MAX_INPUTS = 256
API_MAX_BYTES = 1_000_000


def vectors_file(provider: str, size: int, dims: int) -> np.memmap:
    """The memory-mapped vectors file, created full of NaN on first use."""
    path = vectors_path(provider, size)
    rows = load(Chunk, ["id"], size=size).num_rows
    if path.exists():
        out = cast(np.memmap, np.load(path, mmap_mode="r+"))
        if out.shape != (rows, dims):
            raise SystemExit(f"{path} is {out.shape}, chunks need {(rows, dims)}")
        return out
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path, "w+", np.float16, (rows, dims))
    out[:] = np.nan
    return out


def local(provider: str, batch_size: int):
    """(document prefix, texts -> unit vectors) on the GPU, the provider's own settings."""
    settings = CONFIG["providers"][provider]
    model = load_embedder(
        settings["model_name"],
        dtype=settings.get("dtype", "bfloat16"),
        attn_implementation=settings.get("attn_implementation", "sdpa"),
        revision=settings.get("revision"),
    )
    model.max_seq_length = 2 * max(
        SIZES
    )  # the largest chunks plus title and breadcrumb
    prefix = (model.prompts or {}).get("document") or ""

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    return prefix, embed


def openrouter(provider: str, workers: int):
    """(document prefix, texts -> unit vectors) from OpenRouter's free endpoint."""
    import httpx

    model_id, prefix = CONFIG["api"][provider]
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
                if "free-models-per-day" in r.text:  # 1000 requests a day: wait
                    reset = r.json()["error"]["metadata"]["headers"][
                        "X-RateLimit-Reset"
                    ]
                    print(
                        "daily limit, waiting until",
                        time.ctime(int(reset) / 1000),
                        flush=True,
                    )
                    time.sleep(max(60.0, int(reset) / 1000 - time.time() + 60))
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
        batches, batch, used = [], [], 0
        for text in texts:
            cost = len(json.dumps(text).encode()) + 1
            if batch and (len(batch) == API_MAX_INPUTS or used + cost > API_MAX_BYTES):
                batches.append(batch)
                batch, used = [], 0
            batch.append(text)
            used += cost
        return np.concatenate(list(pool.map(request, [*batches, batch])))

    return prefix, embed


def run(
    provider: str, sizes: list[int], api: bool, batch_size: int, workers: int
) -> None:
    prefix, embed = (
        openrouter(provider, workers) if api else local(provider, batch_size)
    )
    dims = embed([prefix + "dimensions"]).shape[1]
    for size in sizes:
        out = vectors_file(provider, size, dims)
        todo = np.flatnonzero(missing(out))
        print(
            f"{provider} {size}: {len(out) - len(todo)} done, {len(todo)} to do",
            flush=True,
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
            print(f"{provider} {size}: {start + len(rows)}/{len(todo)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(CONFIG["providers"]), default="qwen")
    parser.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--api", action="store_true", help="OpenRouter instead of GPU")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU batch")
    parser.add_argument("--workers", type=int, default=8, help="parallel API requests")
    args = parser.parse_args()
    load_local_env()
    run(args.method, args.sizes, args.api, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
