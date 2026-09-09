from __future__ import annotations

import argparse

from src.indexing.chunk_text import CHUNK_VERSIONS
from src.preprocess.chunks import BASE_CHUNK_VARIANT
from src.vector_store.chunks import (
    DEFAULT_CHUNK_VERSION,
    DEFAULT_INDEXING_PROVIDER,
    INDEXING_PROVIDERS,
    rebuild_chunk_collection,
)


def rebuild_chunk_vector_index(
    method: str = DEFAULT_INDEXING_PROVIDER,
    chunk_version: str = DEFAULT_CHUNK_VERSION,
    variant: str = BASE_CHUNK_VARIANT,
) -> None:
    rebuild_chunk_collection(method, chunk_version, variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=INDEXING_PROVIDERS,
        default=DEFAULT_INDEXING_PROVIDER,
    )
    parser.add_argument(
        "--chunk-version",
        choices=CHUNK_VERSIONS,
        default=DEFAULT_CHUNK_VERSION,
    )
    parser.add_argument(
        "--variant",
        default=BASE_CHUNK_VARIANT,
        help="Chunk variant to index. The base variant keeps its collection name.",
    )
    args = parser.parse_args()
    rebuild_chunk_vector_index(
        args.method,
        chunk_version=args.chunk_version,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
