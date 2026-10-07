from __future__ import annotations

from collections import defaultdict
from typing import Any


def score_rankings(
    rows: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    ks: tuple[int, ...],
    mrr_k: int,
    recall_k: int | None = None,
) -> dict[str, float]:
    relevant_by_question: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    question_ids = list(relevant_by_question)
    recall_k = recall_k or max(ks, default=mrr_k)
    scores = {f"hit@{k}": 0.0 for k in ks}
    scores[f"recall@{recall_k}"] = 0.0
    scores[f"mrr@{mrr_k}"] = 0.0

    for question_id in question_ids:
        relevant = relevant_by_question[question_id]
        ranked = rankings.get(question_id, [])
        for k in ks:
            if relevant.intersection(ranked[:k]):
                scores[f"hit@{k}"] += 1
        scores[f"recall@{recall_k}"] += len(
            relevant.intersection(ranked[:recall_k])
        ) / len(relevant)
        first_rank = first_relevant_rank(ranked[:mrr_k], relevant)
        if first_rank:
            scores[f"mrr@{mrr_k}"] += 1 / first_rank

    total = len(question_ids)
    if total == 0:
        return scores
    return {name: value / total for name, value in scores.items()}


def first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    return next(
        (i for i, chunk_id in enumerate(ranked, 1) if chunk_id in relevant), None
    )


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table whose raw text stays column-aligned."""
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    row_line = lambda cells: (
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"
    )
    return "\n".join(
        [
            row_line(headers),
            "|-" + "-|-".join("-" * width for width in widths) + "-|",
            *(row_line(row) for row in rows),
        ]
    )


def bold_best_table(headers: list[str], rows: list[list[str | float]]) -> str:
    best_by_column = {}
    for index in range(1, len(headers)):
        values = [row[index] for row in rows if isinstance(row[index], int | float)]
        if values:
            best_by_column[index] = max(values)

    rendered_rows = []
    for row in rows:
        rendered = [str(row[0])]
        for index, value in enumerate(row[1:], start=1):
            text = f"{value:.3f}" if isinstance(value, float) else str(value)
            if best_by_column.get(index) == value:
                text = f"**{text}**"
            rendered.append(text)
        rendered_rows.append(rendered)

    return _render_table(headers, rendered_rows)


def plain_table(headers: list[str], rows: list[list[str | int]]) -> str:
    return _render_table(headers, [[str(cell) for cell in row] for row in rows])
