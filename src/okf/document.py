from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

REQUIRED_KEYS = ("type", "title", "description", "timestamp")
DELIMITER = "---"


class OKFDocumentError(ValueError):
    """Raised when an OKF document is malformed."""


@dataclass(frozen=True)
class OKFDocument:
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @classmethod
    def parse(cls, text: str) -> OKFDocument:
        lines = text.splitlines()
        if not lines or lines[0].strip() != DELIMITER:
            raise OKFDocumentError("Concept documents require YAML frontmatter")
        try:
            end = next(
                i
                for i, line in enumerate(lines[1:], start=1)
                if line.strip() == DELIMITER
            )
        except StopIteration as exc:
            raise OKFDocumentError("Unterminated YAML frontmatter") from exc
        try:
            frontmatter = yaml.safe_load("\n".join(lines[1:end])) or {}
        except yaml.YAMLError as exc:
            raise OKFDocumentError(f"Invalid YAML frontmatter: {exc}") from exc
        if not isinstance(frontmatter, dict):
            raise OKFDocumentError("Frontmatter must be a YAML mapping")
        body = "\n".join(lines[end + 1 :]).lstrip("\n")
        return cls(frontmatter=frontmatter, body=body)

    def validate(self) -> None:
        missing = [key for key in REQUIRED_KEYS if not self.frontmatter.get(key)]
        if missing:
            raise OKFDocumentError(f"Missing frontmatter keys: {', '.join(missing)}")
        if not self.body.strip():
            raise OKFDocumentError("Concept body must not be empty")

    def serialize(self) -> str:
        self.validate()
        frontmatter = yaml.safe_dump(
            self.frontmatter,
            sort_keys=False,
            allow_unicode=True,
        ).rstrip()
        return f"{DELIMITER}\n{frontmatter}\n{DELIMITER}\n\n{self.body.rstrip()}\n"
