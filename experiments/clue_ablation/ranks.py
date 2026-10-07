"""Fresh gold-page ranks for the clue sample: dense (qwen, chunks of CHUNK_SIZE) and sparse (BM25, same chunks).

    uv run python -m experiments.clue_ablation.ranks --n 1000

The stored `rank_o` dense ranks came from a title + first 1,000 chars index on a third of the rows, so they are
not used here. A page's rank is the position of its first chunk among the top `--depth` chunks (depth + 1 = not
found). Uses the pipeline's own retrievers. Writes out/ranks-<size>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.clue_ablation.extract import sample
from src.db.dataset import chunk_size
from src.retrieval.methods import build_retriever
from src.retrieval.retrievers.sparse import SparseRetriever
from src.retrieval.retrievers.vector import VectorChunkRetriever

OUT = Path(__file__).parent / "out"


def page_rank(chunk_ids: list[str], page: str, depth: int) -> int:
    """1-based position of the page's first chunk; depth + 1 when it is not in the list."""
    pages = [c.rsplit(":", 2)[0] for c in chunk_ids]
    return pages.index(page) + 1 if page in pages else depth + 1


def main() -> None:
    # the query-embedding cache lives under LLMS4EU_DATA, which is read-only on the shared disk: keep ours local
    os.environ.setdefault("LLMS4EU_DATA", str(OUT / "artifacts"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--methods", default="dense_qwen,bm25,hybrid_qwen")
    args = ap.parse_args()

    df = sample(args.n)
    queries = df.question.tolist()
    ranks = {}
    retrievers = {
        "dense_qwen": lambda: VectorChunkRetriever(name="qwen", provider="qwen"),
        "bm25": SparseRetriever,
        "hybrid_qwen": lambda: build_retriever("qwen_hybrid"),
    }
    for name in args.methods.split(","):
        retriever = retrievers[name]()
        found = retriever.retrieve_batch(queries, args.depth)
        ranks[name] = [
            page_rank([c.id for c in found[i]], page, args.depth)
            for i, page in enumerate(df.id)
        ]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"ranks-{chunk_size()}.jsonl"
    with path.open("w") as fh:
        for i, r in enumerate(df.itertuples()):
            fh.write(
                json.dumps(
                    {"key": r.key, "lang": r.lang} | {k: v[i] for k, v in ranks.items()}
                )
                + "\n"
            )
    for name, v in ranks.items():
        print(
            f"{name}: hit@10 {sum(x <= 10 for x in v) / len(v):.0%} over {len(v)} questions"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
