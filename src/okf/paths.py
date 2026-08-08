from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

CONCEPT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)+$")
RESERVED_NAMES = {"index", "log"}


def parse_concept_id(value: str) -> tuple[str, ...]:
    normalized = value.removesuffix(".md").strip("/")
    if not CONCEPT_ID_RE.fullmatch(normalized):
        raise ValueError(f"Invalid concept id: {value!r}")
    parts = tuple(PurePosixPath(normalized).parts)
    if any(part in {".", ".."} for part in parts) or parts[-1] in RESERVED_NAMES:
        raise ValueError(f"Invalid concept id: {value!r}")
    return parts


def concept_path(bundle_root: Path, concept_id: str) -> Path:
    parts = parse_concept_id(concept_id)
    root = bundle_root.resolve()
    path = (root.joinpath(*parts)).with_suffix(".md").resolve()
    if root not in path.parents:
        raise ValueError("Concept path escapes bundle root")
    return path


def concept_id_for(bundle_root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(bundle_root.resolve())
    return relative.with_suffix("").as_posix()
