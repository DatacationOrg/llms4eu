import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


__all__ = ["ROOT", "chroma_path", "load_local_env", "load_yaml", "sqlite_path"]

ROOT = Path(__file__).resolve().parents[2]


def load_local_env() -> None:
    """Load the root .env file before reading service connection settings."""
    load_dotenv(ROOT / ".env")


def load_yaml(path: Path) -> dict:
    """Read a tiny per-service YAML config file."""
    return yaml.safe_load(path.read_text()) or {}


def sqlite_path() -> Path:
    path = ROOT / os.getenv("SQLITE_PATH", ".local/places.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def chroma_path() -> Path:
    path = ROOT / os.getenv("CHROMA_PATH", "data/cache/chroma")
    path.mkdir(parents=True, exist_ok=True)
    return path
