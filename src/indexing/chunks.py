from __future__ import annotations

import argparse

from src.vector_store.chunks import DEFAULT_INDEXING_PROVIDER, INDEXING_PROVIDERS
from src.vector_store.chunks import rebuild_chunk_collection


def rebuild_chunk_vector_index(method: str = DEFAULT_INDEXING_PROVIDER) -> None:
    rebuild_chunk_collection(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=INDEXING_PROVIDERS,
        default=DEFAULT_INDEXING_PROVIDER,
    )
    args = parser.parse_args()
    rebuild_chunk_vector_index(args.method)


if __name__ == "__main__":
    main()
