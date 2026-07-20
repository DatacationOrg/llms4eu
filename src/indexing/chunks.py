from __future__ import annotations

import argparse

from src.vector_store.chunks import DEFAULT_INDEXING_PROVIDER, INDEXING_PROVIDERS
from src.vector_store.chunks import rebuild_chunk_collection


def rebuild_chunk_vector_index(
    method: str = DEFAULT_INDEXING_PROVIDER,
    assume_yes: bool = False,
) -> None:
    if method == "azure" and not assume_yes:
        from src.shared.cli import confirm

        if not confirm(
            "Do you want to run Azure embeddings and spend quota? [yes/no]: "
        ):
            raise RuntimeError("Azure indexing cancelled.")
    rebuild_chunk_collection(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=INDEXING_PROVIDERS,
        default=DEFAULT_INDEXING_PROVIDER,
    )
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    rebuild_chunk_vector_index(args.method, assume_yes=args.yes)


if __name__ == "__main__":
    main()
