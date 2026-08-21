"""Run every entrant over every applicable page, timed, and persist the output.

Loop order is page-outer/entrant-inner so each snapshot is read from disk once
and all entrants see byte-identical input.

Timing protocol: one discarded warmup call, then the median of N timed calls.
The warmup matters -- resiliparse is Cython and goose3 lazily loads stopword
data, so a cold first call would measure import cost rather than extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from collections import defaultdict

from loguru import logger

from research.scrapers import store
from research.scrapers.extractors import Entrant, enabled, reset_extractor_state


CONFIG = store.config()
TIMING = CONFIG["timing"]


def _snapshot_costs() -> dict[tuple[str, str], float]:
    return {
        (row["page_id"], row["variant"]): float(row["fetch_ms"] or 0.0)
        for row in store.load_snapshots()
    }


def run_one(
    entrant: Entrant,
    inputs: dict[str, str | None],
    page: dict,
    warmup: int | None = None,
    repeats: int | None = None,
) -> dict:
    """Warmup + timed repeats for a single (entrant, page) cell."""
    warmup = int(TIMING["warmup_runs"]) if warmup is None else warmup
    repeats = int(TIMING["timed_runs"]) if repeats is None else repeats
    try:
        for _ in range(warmup):
            reset_extractor_state()
            entrant.run(inputs, page)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}

    samples: list[float] = []
    outcome = None
    try:
        for index in range(repeats):
            reset_extractor_state()
            started = time.perf_counter()
            current = entrant.run(inputs, page)
            samples.append((time.perf_counter() - started) * 1000)
            # Keep the FIRST timed result, not the last. With the state reset above
            # every repeat should be identical, but storing the first means any
            # residual cross-call contamination cannot reach the stored output.
            if index == 0:
                outcome = current
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}

    text = outcome.text if outcome else ""
    return {
        "status": "ok" if text.strip() else "empty",
        "variant": outcome.used_variant if outcome else entrant.variants[0],
        "output": text,
        "output_chars": len(text),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "extract_ms": statistics.median(samples),
        "extract_ms_all": samples,
    }


def _cpu_busy_percent(interval: float = 0.4, cpu: int | None = None) -> float:
    """Busy percentage sampled from /proc/stat, aggregate or for one core.

    `cpu` matters more than it looks. The aggregate figure is close to useless as a
    gate for a *pinned* measurement: on a 16-core machine one process saturating a
    single core reads as 6% busy and sails past a 25% limit, yet if it lands on the
    core we pinned to, our timings are halved. Pinning our process to a core does
    not reserve that core -- other work is still scheduled onto it.
    """
    line_prefix = "cpu " if cpu is None else f"cpu{cpu} "

    def snapshot() -> tuple[int, int]:
        with open("/proc/stat", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(line_prefix):
                    fields = [int(value) for value in line.split()[1:]]
                    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
                    return sum(fields), idle
        raise RuntimeError(f"no {line_prefix.strip()!r} line in /proc/stat")

    total_a, idle_a = snapshot()
    time.sleep(interval)
    total_b, idle_b = snapshot()
    total_delta = max(total_b - total_a, 1)
    return 100.0 * (1.0 - (idle_b - idle_a) / total_delta)


def _stratified(pages: list[dict], limit: int) -> list[dict]:
    """Round-robin across buckets, so a subset stays balanced by page type.

    Taking the first N pages would hand the whole sample to one bucket, and the
    buckets differ enormously in page size -- the wiki pages are ~600 KB against
    ~50 KB for many others -- so an unbalanced subset would misstate every timing.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_bucket[page["bucket"]].append(page)

    picked: list[dict] = []
    index = 0
    while len(picked) < limit and any(index < len(rows) for rows in by_bucket.values()):
        for bucket in sorted(by_bucket):
            if index < len(by_bucket[bucket]) and len(picked) < limit:
                picked.append(by_bucket[bucket][index])
        index += 1
    return picked


def timing_pass(pages_limit: int | None = None) -> None:
    """Re-measure extraction time only, as cleanly as this machine allows.

    Three things make the default pass unsuitable for reporting speed: it shares
    the machine with whatever else is running, it uses only three repeats, and it
    may migrate between cores mid-measurement. This pass pins to a single CPU,
    takes `precise_runs` repeats after `precise_warmup` warmups, and refuses to
    run on a busy machine -- CPU contention inflates timings by a factor that
    varies per entrant, which would silently reorder the speed table.

    Outputs and statuses are left exactly as the main pass recorded them; only the
    timing columns are rewritten.
    """
    warmup = int(TIMING["precise_warmup"])
    repeats = int(TIMING["precise_runs"])
    limit = float(TIMING["max_cpu_busy_percent"])

    cpu = int(TIMING["pin_cpu"])

    def check_load(context: str) -> float:
        """Gate on the pinned core's utilisation, not on load average or aggregate.

        Load average is the wrong instrument: under WSL2 it is inflated by I/O wait
        and decays slowly, so it read 6.7 on a machine where nothing was running.

        The aggregate figure is also wrong for a pinned run. One competing process
        saturating a single core on a 16-core box reads as ~6% and passes any
        sensible limit, while halving our throughput if it happens to share our
        core. So the gate is the pinned core; the aggregate is reported alongside
        because broad contention still costs memory bandwidth and cache.
        """
        core = _cpu_busy_percent(cpu=cpu)
        overall = _cpu_busy_percent()
        if core > limit:
            raise SystemExit(
                f"CPU {cpu} is {core:.0f}% busy (limit {limit:.0f}%, machine "
                f"{overall:.0f}%) {context}. Timings taken under contention are not "
                "comparable -- contention inflates them by an amount that varies "
                "per entrant, which silently reorders the speed table. Wait for the "
                "machine to go idle and re-run; results already written are kept."
            )
        return core

    load = check_load("at startup")
    try:
        os.sched_setaffinity(0, {cpu})
        pinned = f"pinned to CPU {cpu}"
    except (AttributeError, OSError) as exc:
        pinned = f"could not pin CPU ({exc}); timings will be noisier"

    entrants = [entrant for entrant in enabled(None) if entrant.layer != "wiki"]
    pages = store.load_pages()
    if pages_limit and pages_limit < len(pages):
        pages = _stratified(pages, pages_limit)
        logger.info(
            f"timing a stratified subset of {len(pages)} pages. The speed table "
            "will flag a mixed repeat count, because the pages not covered here keep "
            "their coarser timings from the main extraction pass."
        )
    existing = {(row["page_id"], row["entrant"]): row for row in store.load_runs()}
    logger.info(
        f"timing pass: {len(entrants)} entrants x {len(pages)} pages, "
        f"{warmup} warmup + {repeats} timed, {pinned}, core {load:.0f}% busy"
    )

    for index, page in enumerate(pages, start=1):
        inputs = {
            variant: store.snapshot_text(page["id"], variant)
            for variant in ("raw", "rendered", "wikitext")
        }
        for entrant in entrants:
            previous = existing.get((page["id"], entrant.name))
            if not previous or previous["status"] not in ("ok", "empty"):
                continue
            result = run_one(entrant, inputs, page, warmup=warmup, repeats=repeats)
            if result["status"] == "error":
                logger.warning(
                    f"{entrant.name} errored while re-timing "
                    f"{page['url'][:50]}; keeping previous timing"
                )
                continue
            samples = result["extract_ms_all"]
            store.record_run(
                page["id"],
                entrant.name,
                variant=previous["variant"],
                output=previous["output"],
                output_chars=previous["output_chars"],
                output_sha256=previous["output_sha256"],
                extract_ms=statistics.median(samples),
                extract_ms_all=samples,
                input_ms=previous["input_ms"],
                status=previous["status"],
                error=previous["error"],
            )
        if index % 5 == 0:
            # Re-check rather than trusting the startup reading: a job starting
            # mid-run would otherwise quietly contaminate the remaining pages.
            current = check_load(f"after {index} pages")
            logger.info(f"  timed {index}/{len(pages)} pages (core {current:.0f}%)")

    logger.info("--- median extract ms (single pinned CPU) ---")
    runs = store.load_run_index()
    for entrant in sorted(entrants, key=lambda e: e.name):
        values = [
            row["extract_ms"]
            for row in runs
            if row["entrant"] == entrant.name and row["extract_ms"]
        ]
        if values:
            logger.info(
                f"{entrant.name:24s} median {statistics.median(values):8.2f} ms"
                f"  mean {statistics.mean(values):8.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrants", nargs="*", help="subset of entrant names")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="one page only, verifying every entrant imports and produces output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute cells that already have a result",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="re-measure timings on one pinned CPU with more repeats; "
        "outputs are left untouched",
    )
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="extract each cell once instead of warmup + 3 timed runs, and leave "
        "extract_ms null. Roughly 4x faster. Use whenever the pass only needs to "
        "refresh *outputs* -- re-extracting after a corpus change, adding an "
        "entrant -- since the speed table is a separate, already-settled result "
        "that a correctness pass does not improve.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="with --timing-only, time a stratified subset of this many pages. "
        "The full 100-page pass is ~13,200 extractions and takes about an hour; "
        "the entrants differ by 50x in speed, so 20-30 pages rank them just as "
        "confidently in about ten minutes.",
    )
    arguments = parser.parse_args()

    if arguments.timing_only:
        timing_pass(arguments.pages)
        return

    store.initialize()
    entrants = enabled(arguments.entrants)
    # wikiextractor-v2 needs its own interpreter; see wiki_pass.py.
    entrants = [entrant for entrant in entrants if entrant.layer != "wiki"]

    pages = store.load_pages()
    if arguments.dry_run:
        pages = pages[:1]
        logger.info(f"dry run over {pages[0]['url']}")

    existing = {
        (row["page_id"], row["entrant"])
        for row in store.load_runs()
        if row["status"] != "error"
    }
    costs = _snapshot_costs()
    tally: dict[str, dict[str, int]] = {
        entrant.name: {"ok": 0, "empty": 0, "error": 0, "n/a": 0}
        for entrant in entrants
    }

    for index, page in enumerate(pages, start=1):
        inputs = {
            variant: store.snapshot_text(page["id"], variant)
            for variant in ("raw", "rendered", "wikitext")
        }
        for entrant in entrants:
            if not entrant.applies_to(page):
                store.record_run(
                    page["id"],
                    entrant.name,
                    variant=entrant.variants[0],
                    status="n/a",
                    error="entrant does not apply to this page",
                )
                tally[entrant.name]["n/a"] += 1
                continue
            if not arguments.force and (page["id"], entrant.name) in existing:
                continue

            if arguments.no_timing:
                result = run_one(entrant, inputs, page, warmup=0, repeats=1)
                # Null the timing rather than storing the single cold sample. A
                # no-warmup measurement is import cost, not extraction cost, and
                # leaving it in place would quietly corrupt the speed table --
                # `--timing-only` is the pass that produces those numbers.
                if result["status"] != "error":
                    result["extract_ms"] = None
                    result["extract_ms_all"] = None
            else:
                result = run_one(entrant, inputs, page)
            variant = result.get("variant", entrant.variants[0])
            result["input_ms"] = costs.get((page["id"], variant), 0.0)
            store.record_run(page["id"], entrant.name, **result)
            tally[entrant.name][result["status"]] += 1

            if result["status"] == "error":
                logger.warning(
                    f"[{index}/{len(pages)}] {entrant.name} FAILED on "
                    f"{page['url'][:50]}: {result['error'][:90]}"
                )
        logger.info(f"[{index}/{len(pages)}] {page['bucket']:9s} {page['url'][:64]}")

    logger.info("--- per-entrant coverage ---")
    for name, counts in tally.items():
        logger.info(
            f"{name:24s} ok={counts['ok']:3d} empty={counts['empty']:3d} "
            f"error={counts['error']:3d} n/a={counts['n/a']:3d}"
        )


if __name__ == "__main__":
    main()
