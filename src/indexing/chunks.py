from __future__ import annotations

import argparse

from src.indexing.chunk_text import BASE_CHUNK_VARIANT, CHUNK_VERSIONS
from src.vector_store.chunks import (
    DEFAULT_CHUNK_VERSION,
    DEFAULT_INDEXING_PROVIDER,
    INDEXING_PROVIDERS,
)
from src.vector_store.chunks import rebuild_chunk_collection


def rebuild_chunk_vector_index(
    method: str = DEFAULT_INDEXING_PROVIDER,
    chunk_version: str = DEFAULT_CHUNK_VERSION,
    chunk_variant: str = BASE_CHUNK_VARIANT,
) -> None:
    rebuild_chunk_collection(method, chunk_version, chunk_variant)


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
        "--chunk-variant",
        default=BASE_CHUNK_VARIANT,
        help="Chunking variant to index, as built by src.preprocess.chunks",
    )
    args = parser.parse_args()
    rebuild_chunk_vector_index(
        args.method,
        chunk_version=args.chunk_version,
        chunk_variant=args.chunk_variant,
    )


if __name__ == "__main__":
    main()
