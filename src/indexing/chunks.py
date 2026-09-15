from __future__ import annotations

import argparse

from src.vector_store.chunks import (
    DEFAULT_INDEXING_PROVIDER,
    INDEXING_PROVIDERS,
    rebuild_chunk_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=INDEXING_PROVIDERS,
        default=DEFAULT_INDEXING_PROVIDER,
    )
    rebuild_chunk_collection(parser.parse_args().method)


if __name__ == "__main__":
    main()
