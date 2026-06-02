from __future__ import annotations

import argparse
from pathlib import Path

from _cli import run_cli
from src.eval.evaluate import format_eval_report, run_eval
from src.shared.env import load_yaml

METHODS = [
    "qwen",
    "qwen_hybrid",
    "qwen_rerank",
    "qwen_hybrid_rerank",
]
CONFIG = load_yaml(Path("experiments/indexing/config.yaml"))


def main() -> None:
    args = _parse_args()
    run = run_eval(
        METHODS,
        limit=args.limit,
        category=args.category,
        warmup=args.warmup,
    )
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Warmup: {run.warmup_count} queries")
    print()
    print(format_eval_report(run, include_categories=args.category is None))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare qwen vector, hybrid, rerank, and hybrid+rerank.",
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=CONFIG["default_warmup"])
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(main)
