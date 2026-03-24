import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

STRONG_MARKERS = [".env", "pyproject.toml"]
OPTIONAL_MARKERS = [".git", ".venv", "uv.lock"]  # fallback if strong markers not found


def find_project_root() -> Path:
    """
    Find the project root folder.
    - Primary markers: .env, pyproject.toml
    - Fallback markers: .git, .venv, uv.lock
    - Starts from multiple likely locations (cwd, script, this file)
    """
    candidates = [
        Path(__file__).resolve().parent,
        Path(sys.argv[0]).resolve().parent,
        Path.cwd(),
    ]

    for start in candidates:
        for parent in [start, *start.parents]:
            if any((parent / marker).exists() for marker in STRONG_MARKERS):
                return parent

    # fallback to optional markers
    for start in candidates:
        for parent in [start, *start.parents]:
            if any((parent / marker).exists() for marker in OPTIONAL_MARKERS):
                return parent

    # ultimate fallback: cwd
    return Path.cwd()


# Optional override via environment variable (actual environment variable, not one in .env)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", find_project_root()))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
INTERIM_DATA = DATA_DIR / "interim"
PROCESSED_DATA = DATA_DIR / "processed"


class Settings(BaseSettings):
    """
    Settings configuration class for managing environment variables.

    This class defines all application environment variables with type annotations
    and optional default values. Environment variables are loaded from a .env file
    located at the project root, or from the environment when running in e.g. Docker.

    Attributes:
        model_config: Configuration dictionary that specifies the .env file path
                     and forbids extra attributes not explicitly defined.

    Example:
        To add a new environment variable, define it as a class attribute with
        a type annotation and optional default value:

            example_variable: str = "default_value"
    """

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]


env = Settings()
