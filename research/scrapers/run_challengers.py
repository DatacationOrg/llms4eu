"""Run crawl4ai / markitdown / docling over the corpus, each in its own interpreter.

Separate from `run_extractors` for one reason: dependency isolation. `crawl4ai` pulls
Playwright, nltk, tiktoken and openai; `docling` pulls a document-AI stack. Neither can
join `arena-env` without making it unresolvable for the other nine entrants. Each tool
therefore runs under `uv run --no-project --with <tool>` and returns JSON.

Every challenger reads the same saved `raw.html` as the other twelve entrants. This
matters most for `crawl4ai`, which is a fetcher *and* an extractor: letting it fetch its
own copy would confound the two layers, and the fetch-layer result already showed that
confound is worth 44% of pages.

Timings are wall-clock inside the worker and are NOT comparable to the `--timing-only`
numbers for the in-process entrants: a subprocess boundary and a cold import are
included. They are recorded for order-of-magnitude only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from research.scrapers import store
from research.scrapers.extractors import CHALLENGERS


WORKER = Path(__file__).parent / "extractors" / "challenger_worker.py"


def jobs_for(pages: list[dict]) -> list[dict]:
    """One job per page that actually has a raw snapshot on disk."""
    out = []
    for page in pages:
        path = store.data_dir() / "snapshots" / page["id"] / "raw.html"
        if path.exists():
            out.append({"id": page["id"], "path": str(path)})
    return out


def run_tool(tool: str, jobs: list[dict]) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        out_file = Path(tmp) / "out.json"
        job_file.write_text(json.dumps(jobs), encoding="utf-8")
        command = [
            "uv",
            "run",
            "--no-project",
            "--with",
            tool,
            "python",
            str(WORKER),
            tool,
            str(job_file),
            str(out_file),
        ]
        print(f"  $ {' '.join(command[:7])} …", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip().splitlines()[-6:]
            raise SystemExit(f"{tool} worker failed:\n  " + "\n  ".join(tail))
        return json.loads(out_file.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", default=",".join(CHALLENGERS))
    parser.add_argument("--limit", type=int, default=None, help="first N pages only")
    parser.add_argument("--wiki-only", action="store_true")
    arguments = parser.parse_args()

    store.initialize()
    pages = store.load_pages()
    if arguments.wiki_only:
        pages = [page for page in pages if page["is_wiki"]]
    if arguments.limit:
        pages = pages[: arguments.limit]

    jobs = jobs_for(pages)
    print(f"{len(jobs)} pages with a raw snapshot")

    for tool in arguments.tools.split(","):
        tool = tool.strip()
        if tool not in CHALLENGERS:
            raise SystemExit(f"unknown challenger {tool!r}; expected {CHALLENGERS}")
        print(f"\n=== {tool} ===", flush=True)
        results = run_tool(tool, jobs)

        modes: dict[str, int] = {}
        ok = empty = failed = 0
        for page_id, result in results.items():
            text = result["text"]
            if result["error"]:
                status, failed = "error", failed + 1
            elif not text.strip():
                status, empty = "empty", empty + 1
            else:
                status, ok = "ok", ok + 1
            modes[result.get("mode") or "-"] = (
                modes.get(result.get("mode") or "-", 0) + 1
            )
            store.record_run(
                page_id,
                tool,
                variant="raw",
                output=text,
                output_chars=len(text),
                output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                extract_ms=result["ms"],
                status=status,
                error=result["error"],
            )
        print(f"  ok={ok} empty={empty} error={failed}  modes={modes}", flush=True)

    print("\ndone; regenerate the report to pick these up")


if __name__ == "__main__":
    sys.exit(main())
