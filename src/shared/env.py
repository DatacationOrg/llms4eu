from typing import Any
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


__all__ = [
    "ROOT",
    "chroma_path",
    "data_path",
    "load_local_env",
    "load_yaml",
    "pages_db",
]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = "/data/llms4eu"


def load_local_env() -> None:
    """Load the root .env file before reading service connection settings."""
    load_dotenv(ROOT / ".env")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a tiny per-service YAML config file."""
    return yaml.safe_load(path.read_text()) or {}


def data_path(*parts: str) -> Path:
    """Resolve a path in the shared artifact store, creating its parent."""
    path = Path(os.getenv("LLMS4EU_DATA", DEFAULT_DATA_ROOT)).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pages_db() -> Path:
    """Canonical raw-page SQLite database."""
    return data_path("db", "pages.db")


def chroma_path() -> Path:
    path = data_path("chroma")
    path.mkdir(parents=True, exist_ok=True)
    return path
