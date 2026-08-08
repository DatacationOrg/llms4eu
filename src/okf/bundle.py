from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.okf.document import OKFDocument, OKFDocumentError
from src.okf.paths import concept_id_for, concept_path

LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")
MAX_INDEX_ENTRIES = 20


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def write_concept(bundle_root: Path, concept_id: str, document: OKFDocument) -> Path:
    path = concept_path(bundle_root, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = document.serialize()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_concepts(bundle_root: Path) -> dict[str, OKFDocument]:
    concepts = {}
    if not bundle_root.exists():
        return concepts
    for path in sorted(bundle_root.rglob("*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        concepts[concept_id_for(bundle_root, path)] = OKFDocument.parse(
            path.read_text(encoding="utf-8")
        )
    return concepts


def regenerate_indexes(bundle_root: Path) -> list[Path]:
    concepts = load_concepts(bundle_root)
    entries: dict[Path, list[tuple[str, str, str, str]]] = defaultdict(list)
    for concept_id, document in concepts.items():
        path = concept_path(bundle_root, concept_id)
        fm = document.frontmatter
        entries[path.parent].append(
            (
                str(fm["type"]),
                str(fm["title"]),
                path.name,
                str(fm["description"]),
            )
        )

    written = []
    directories = set(entries)
    for directory in tuple(directories):
        parent = directory.parent
        while bundle_root.resolve() in (parent.resolve(), *parent.resolve().parents):
            directories.add(parent)
            if parent.resolve() == bundle_root.resolve():
                break
            parent = parent.parent

    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        if not entries[directory]:
            continue
        direct_entries = entries[directory]
        _remove_browse_indexes(directory)
        text, browse_paths = _directory_index(directory, direct_entries)
        path = directory / "index.md"
        path.write_text(text, encoding="utf-8")
        written.extend([path, *browse_paths])
        if directory != bundle_root:
            parent = directory.parent
            entries[parent].append(
                (
                    "Collections",
                    directory.name.replace("_", " ").title(),
                    f"{directory.name}/index.md",
                    _collection_description(direct_entries),
                )
            )
    return written


def validate_bundle(bundle_root: Path) -> ValidationReport:
    report = ValidationReport()
    if not bundle_root.exists():
        report.errors.append(f"Bundle does not exist: {bundle_root}")
        return report
    for path in sorted(bundle_root.rglob("*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        try:
            document = OKFDocument.parse(path.read_text(encoding="utf-8"))
            document.validate()
        except (OSError, OKFDocumentError) as exc:
            report.errors.append(f"{path.relative_to(bundle_root)}: {exc}")
            continue
        for target in LINK_RE.findall(document.body):
            if target.startswith(("http://", "https://")):
                continue
            resolved = (path.parent / target).resolve()
            if bundle_root.resolve() not in resolved.parents:
                report.errors.append(
                    f"{path.relative_to(bundle_root)}: link escapes bundle: {target}"
                )
            elif not resolved.exists():
                report.warnings.append(
                    f"{path.relative_to(bundle_root)}: broken link: {target}"
                )
    return report


def _directory_index(
    directory: Path,
    entries: list[tuple[str, str, str, str]],
) -> tuple[str, list[Path]]:
    if len(entries) <= MAX_INDEX_ENTRIES:
        return _index_text(_sections(entries)), []

    browse_paths = []
    browse_entries = []
    ordered = sorted(entries, key=lambda entry: (entry[1].casefold(), entry[2]))
    for offset in range(0, len(ordered), MAX_INDEX_ENTRIES):
        group = ordered[offset : offset + MAX_INDEX_ENTRIES]
        browse_dir = directory / f"_browse-{offset // MAX_INDEX_ENTRIES + 1:02d}"
        browse_dir.mkdir(parents=True, exist_ok=True)
        browse_path = browse_dir / "index.md"
        relative_group = [
            (type_name, title, f"../{link}", description)
            for type_name, title, link, description in group
        ]
        browse_path.write_text(_index_text(_sections(relative_group)), encoding="utf-8")
        browse_paths.append(browse_path)
        titles = [entry[1] for entry in group]
        browse_entries.append(
            (
                "Browse groups",
                f"{titles[0]} – {titles[-1]}",
                f"{browse_dir.name}/index.md",
                f"Contains {len(group)} entries including {', '.join(titles[:4])}.",
            )
        )
    return _index_text(_sections(browse_entries)), browse_paths


def _sections(
    entries: list[tuple[str, str, str, str]],
) -> dict[str, list[tuple[str, str, str]]]:
    sections = defaultdict(list)
    for type_name, title, link, description in entries:
        sections[type_name].append((title, link, description))
    return sections


def _index_text(
    sections: dict[str, list[tuple[str, str, str]]],
) -> str:
    lines = ["---", 'okf_version: "0.1"', "---", ""]
    for type_name in sorted(sections):
        lines.extend([f"# {type_name}", ""])
        for title, link, description in sorted(sections[type_name]):
            lines.append(f"* [{title}]({link}) - {description}")
        lines.append("")
    return "\n".join(lines)


def _collection_description(
    entries: list[tuple[str, str, str, str]],
) -> str:
    titles = [title for _type, title, _link, _description in entries]
    examples = ", ".join(sorted(titles, key=str.casefold)[:8])
    return f"Contains {len(entries)} entries, including {examples}."


def _remove_browse_indexes(directory: Path) -> None:
    for path in directory.glob("_browse-*"):
        if path.is_dir():
            shutil.rmtree(path)
