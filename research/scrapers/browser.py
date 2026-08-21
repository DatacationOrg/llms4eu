"""Playwright launch helper.

This WSL2 box has no `libasound.so.2`, which Chromium links against, and
installing it system-wide needs sudo. `snapshot.py` therefore extracts the
library into `.local/scraper-arena/lib/` and this module puts it on the loader
path before any browser starts. Chromium is a child process, so setting the
environment variable from Python is enough -- the child inherits it.
"""

from __future__ import annotations

import os

from research.scrapers import store


def prepare_library_path() -> str | None:
    """Prepend the vendored library directory to LD_LIBRARY_PATH, if present."""
    library_dir = store.data_dir() / "lib"
    if not library_dir.is_dir():
        return None
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(library_dir) not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(library_dir), *parts])
    return os.environ["LD_LIBRARY_PATH"]
