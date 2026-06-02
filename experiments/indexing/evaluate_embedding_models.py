from __future__ import annotations

import argparse
from pathlib import Path

from _cli import run_cli
from src.eval.evaluate import format_eval_report, run_eval
from src.shared.env import load_yaml

INDEX_CONFIG = load_yaml(Path("src/indexing/config.yaml"))
CONFIG = load_yaml(Path("experiments/indexing/config.yaml"))


def main() -> None:
    args = _parse_args()
    providers = _providers(args.providers)
    run = run_eval(
        providers,
        limit=args.limit,
        category=args.category,
        warmup=args.warmup,
    )
    print(f"Providers: {', '.join(providers)}")
    print(f"Warmup: {run.warmup_count} queries")
    print()
    print(format_eval_report(run, include_categories=args.category is None))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chunk embedding providers on approved eval questions.",
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=CONFIG["default_warmup"])
    parser.add_argument(
        "--providers",
        default="all",
        help="Comma-separated providers, or all from src/indexing/config.yaml.",
    )
    return parser.parse_args()


def _providers(value: str) -> list[str]:
    if value == "all":
        return list(INDEX_CONFIG["providers"])
    return [name.strip() for name in value.split(",") if name.strip()]


if __name__ == "__main__":
    run_cli(main)
