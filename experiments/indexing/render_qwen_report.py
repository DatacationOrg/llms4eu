"""Render a `compare_chunkings.py` checkpoint in `compare_qwen_modes.py` layout.

The two harnesses answer different questions and print different reports: the
sweep is a variant grid, the qwen comparison is one method-per-row table that
every earlier retrieval report in `docs/` used. A single-variant sweep is
already exactly the qwen shape, so this reprojects it rather than re-running
two hours of GPU to produce the same numbers in another format.

What it cannot invent, it leaves as `-`: the sweep checkpoint deliberately
stores aggregates and not per-question rankings (a stored grid ran to 96 MB),
so any column that needs the rankings back -- `judge_hit@K`, and the
action-log half of the agentic diagnostics -- is absent rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.eval.agentic_diagnostics import format_agentic_diagnostics
from src.eval.evaluate import cost_metrics
from src.eval.metrics import bold_best_table, plain_table

# Column order of the published reports, so a rendered table drops straight into
# them. Anything a run measured that is not named here is appended after.
PREFERRED_SCORES = (
    "hit@1",
    "hit@5",
    "hit@10",
    "recall@10",
    "mrr@10",
    "hit@20",
    "recall@20",
    "mrr@20",
)


def render(
    checkpoint_paths: list[Path],
    variant: str | None = None,
    compare: list[Path] | None = None,
) -> str:
    """Merge one or more sweep checkpoints into a single qwen-format report.

    Several checkpoints are needed because the sweep invalidates a checkpoint
    whose `--methods` list changed, so adding one method to a finished run means
    a second checkpoint rather than an extra cell. Cells merge only when their
    question count and design agree; a duplicated method keeps the first cell
    seen and is reported, never silently averaged.
    """
    signature: dict = {}
    runs: dict = {}
    duplicates: list[str] = []
    for path in checkpoint_paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        signature = signature or state["signature"]
        for key, cell in state["runs"].items():
            if key in runs:
                duplicates.append(key)
                continue
            runs[key] = cell

    cells = _cells_for_variant(runs, variant)
    if not cells:
        names = ", ".join(str(path) for path in checkpoint_paths)
        raise ValueError(f"No cells in {names} for variant {variant!r}")
    chosen = cells[0][0]
    methods = [method for _, method, _ in cells]
    questions = max(cell["questions"] for _, _, cell in cells)

    counts = sorted({cell["questions"] for _, _, cell in cells})
    if len(counts) > 1:
        raise ValueError(
            f"Cells were scored on different question counts {counts}; merging "
            "them would publish a row that is not comparable. Re-run the smaller "
            "cell at the same --limit."
        )

    lines = [
        f"Methods: {', '.join(methods)}",
        "Scoring unit: chunk",
        f"Warmup: {signature.get('warmup', 0)} queries",
        f"Variant: {chosen}",
        f"Design: {signature.get('design', 'unknown')}",
        "Source checkpoints: " + ", ".join(str(path) for path in checkpoint_paths),
        f"Rendered: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Progress",
        *(
            f"- {method}: {cell['questions']}/{questions} timed queries"
            for _, method, cell in cells
        ),
        "",
        f"Evaluating {questions} questions",
        "",
        _overall(cells),
        "",
        "Speed",
        _speed(cells),
        "",
        "hit@5 by category",
        _categories(cells),
    ]

    diagnostics = _diagnostics(cells)
    if diagnostics:
        lines.extend(["", diagnostics])

    history = {}
    for path in compare or []:
        markdown = path.read_text(encoding="utf-8")
        parsed = _parse_overall(markdown)
        if parsed:
            count = next(
                (
                    line.split()[1]
                    for line in markdown.splitlines()
                    if line.startswith("Evaluating ")
                ),
                "?",
            )
            history[path.stem] = (count, parsed)
    delta = _delta_table(cells, history, "hit@5")
    if delta:
        lines.extend(["", delta])
    notes = _notes(cells, duplicates)
    if notes:
        lines.extend(["", notes])
    return "\n".join(lines)


def _cells_for_variant(runs: dict, variant: str | None) -> list[tuple[str, str, dict]]:
    parsed = []
    for key, cell in runs.items():
        cell_variant, _, method = key.partition("|")
        if variant is None or cell_variant == variant:
            parsed.append((cell_variant, method, cell))
    if variant is None and parsed:
        # Default to whichever variant the checkpoint holds most of, so a
        # single-variant run needs no flag and a grid still renders one table.
        counts: dict[str, int] = {}
        for cell_variant, _, _ in parsed:
            counts[cell_variant] = counts.get(cell_variant, 0) + 1
        best = max(counts, key=lambda name: counts[name])
        parsed = [row for row in parsed if row[0] == best]
    return parsed


def _score_names(cells: list[tuple[str, str, dict]]) -> list[str]:
    measured = {name for _, _, cell in cells for name in cell["scores"]}
    # store_share is a sweep-only cost column and has no place in this layout.
    measured = {name for name in measured if not name.startswith("store_share")}
    ordered = [name for name in PREFERRED_SCORES if name in measured]
    ordered.extend(sorted(measured - set(ordered)))
    return ordered


def _overall(cells: list[tuple[str, str, dict]]) -> str:
    score_names = _score_names(cells)
    rows = [
        [method, *(_score(cell["scores"], name) for name in score_names)]
        for _, method, cell in cells
    ]
    return "\n".join(
        [
            "Overall",
            bold_best_table(
                ["method", *score_names],
                rows,
                lower_is_better=cost_metrics(score_names),
            ),
        ]
    )


def _score(scores: dict, name: str):
    value = scores.get(name)
    return "-" if value is None else value


def _speed(cells: list[tuple[str, str, dict]]) -> str:
    rows = []
    for _, method, cell in cells:
        queries_per_query = cell.get("queries_per_query", 1.0)
        rows.append(
            [
                method,
                f"{cell['seconds']:.2f}",
                f"{cell['ms_per_query']:.1f}",
                f"{queries_per_query:.2f}",
                int(round(cell["questions"] * queries_per_query)),
            ]
        )
    return plain_table(
        ["method", "seconds", "ms/query", "queries/query", "queries"], rows
    )


def _categories(cells: list[tuple[str, str, dict]]) -> str:
    types = sorted(
        {name for _, _, cell in cells for name in cell.get("category_scores", {})}
    )
    if not types:
        return "(no category scores in this checkpoint)"
    rows = [
        [method, *(_score(cell.get("category_scores", {}), name) for name in types)]
        for _, method, cell in cells
    ]
    return bold_best_table(["method", *types], rows)


def _diagnostics(cells: list[tuple[str, str, dict]]) -> str:
    """Merge every cell's stored diagnostics into one paired table."""
    summaries: dict = {}
    cutoff = None
    for _, _, cell in cells:
        stored = cell.get("diagnostics") or {}
        if stored.get("summaries"):
            summaries.update(stored["summaries"])
            cutoff = stored.get("cutoff", cutoff)
    if not summaries:
        return ""
    return format_agentic_diagnostics(
        {"cutoff": cutoff, "summaries": summaries, "details": {}}
    )


def _notes(
    cells: list[tuple[str, str, dict]], duplicates: list[str] | None = None
) -> str:
    """State what this layout cannot show, so a blank column is never read as a zero."""
    notes = []
    for key in sorted(set(duplicates or [])):
        notes.append(
            f"- `{key}` was measured in more than one checkpoint; the first cell "
            "seen was kept and the later one ignored."
        )
    if any((cell.get("diagnostics") or {}).get("summaries") for _, _, cell in cells):
        notes.append(
            "- The `retried` / `rewritten` / `expanded` / `retry precision` / "
            "`retry ms` columns read 0 or `-` because the sweep harness does not "
            "persist the per-question agent action log its diagnostics are "
            "computed from. `recovered` and `lost` come from the rankings and are "
            "measured. A zero in the action-log columns is missing data, not an "
            "agent that never acted -- `queries/query` above shows it acted."
        )
    if not any("judge_hit" in name for name in _score_names(cells)):
        notes.append(
            "- No `judge_hit@K` column: equivalence judging was not run "
            "(`--judge-equivalence`)."
        )
    if not notes:
        return ""
    return "\n".join(["Not measured in this run", *notes])


def _parse_overall(markdown: str) -> dict[str, dict[str, str]]:
    """Read the `Overall` table out of a published qwen-format report.

    Older results exist only as their rendered markdown -- the 2026-07-27 run
    left no checkpoint -- so the published table is the only source for them.
    """
    scores: dict[str, dict[str, str]] = {}
    headers: list[str] = []
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == "Overall":
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            if headers:
                break
            continue
        cells = [cell.strip().strip("*") for cell in stripped.strip("|").split("|")]
        if not headers:
            headers = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        scores[cells[0]] = dict(zip(headers[1:], cells[1:]))
    return scores


def _delta_table(
    cells: list[tuple[str, str, dict]],
    history: dict[str, tuple[str, dict[str, dict[str, str]]]],
    metric: str,
) -> str:
    """Every agent against its own non-agentic baseline, across runs.

    Absolute scores from different runs are not comparable -- different question
    samples, embedders and judge models -- but each agent's delta against the
    baseline measured beside it in the same run is. That delta is the only
    number that answers "does the agent earn its latency".
    """
    from src.eval.agentic_diagnostics import baseline_for_agent

    rows = []
    for label, (questions, scores) in history.items():
        for method in scores:
            baseline = baseline_for_agent(method)
            if baseline is None or baseline not in scores:
                continue
            agent_value = scores[method].get(metric)
            base_value = scores[baseline].get(metric)
            if not agent_value or not base_value or "-" in (agent_value, base_value):
                continue
            delta = float(agent_value) - float(base_value)
            rows.append(
                [
                    label,
                    questions,
                    method,
                    baseline,
                    base_value,
                    agent_value,
                    f"{100 * delta:+.1f}pp",
                ]
            )

    for _, method, cell in cells:
        baseline = baseline_for_agent(method)
        if baseline is None:
            continue
        base_cell = next((c for _, m, c in cells if m == baseline), None)
        if base_cell is None:
            continue
        agent_value = cell["scores"].get(metric)
        base_value = base_cell["scores"].get(metric)
        if agent_value is None or base_value is None:
            continue
        rows.append(
            [
                "this run",
                str(cell["questions"]),
                method,
                baseline,
                f"{base_value:.3f}",
                f"{agent_value:.3f}",
                f"{100 * (agent_value - base_value):+.1f}pp",
            ]
        )

    if not rows:
        return ""
    return "\n".join(
        [
            f"Agent versus its own baseline, across runs ({metric})",
            "",
            plain_table(
                ["run", "questions", "agent", "baseline", "baseline", "agent", "delta"],
                rows,
            ),
            "",
            "Absolute scores across runs are NOT comparable: different question "
            "samples, embedders and judge models. The delta column is, because "
            "each agent is differenced against the baseline measured beside it in "
            "the same run.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument(
        "--variant",
        default=None,
        help="Which variant to render. Defaults to the best-represented one.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--compare",
        type=Path,
        action="append",
        default=None,
        help=(
            "A previously published qwen-format report to difference against. "
            "Repeatable. Only each agent's delta over its own baseline is "
            "carried across, never absolute scores."
        ),
    )
    args = parser.parse_args()

    report = render(args.checkpoint, args.variant, args.compare)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
