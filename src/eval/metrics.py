from __future__ import annotations

from collections import defaultdict


def score_rankings(
    rows: list[dict],
    rankings: dict[str, list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    relevant_by_question: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        relevant_by_question[row["question_id"]].add(row["chunk_id"])

    question_ids = list(relevant_by_question)
    scores = {f"hit@{k}": 0.0 for k in ks}
    scores["mrr@10"] = 0.0

    for question_id in question_ids:
        relevant = relevant_by_question[question_id]
        ranked = rankings.get(question_id, [])
        for k in ks:
            if relevant.intersection(ranked[:k]):
                scores[f"hit@{k}"] += 1
        first_rank = _first_relevant_rank(ranked[:10], relevant)
        if first_rank:
            scores["mrr@10"] += 1 / first_rank

    total = len(question_ids)
    return {name: value / total for name, value in scores.items()}


def _first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return index
    return None


def bold_best_table(headers: list[str], rows: list[list[str | float]]) -> str:
    numeric_columns = []
    for index in range(1, len(headers)):
        values = [row[index] for row in rows if isinstance(row[index], int | float)]
        if values:
            numeric_columns.append((index, max(values)))

    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered = [str(row[0])]
        for index, value in enumerate(row[1:], start=1):
            text = f"{value:.3f}" if isinstance(value, float) else str(value)
            if any(
                index == best_index and value == best
                for best_index, best in numeric_columns
            ):
                text = f"**{text}**"
            rendered.append(text)
        rendered_rows.append(rendered)

    widths = [
        max(len(_plain(row[index])) for row in [headers, *rendered_rows])
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(headers))
        + " |",
        "|-" + "-|-".join("-" * width for width in widths) + "-|",
    ]
    lines.extend(
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rendered_rows
    )
    return "\n".join(lines)


def plain_table(headers: list[str], rows: list[list[str | int]]) -> str:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(row[index]) for row in [headers, *rendered_rows])
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(headers))
        + " |",
        "|-" + "-|-".join("-" * width for width in widths) + "-|",
    ]
    lines.extend(
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rendered_rows
    )
    return "\n".join(lines)


def _plain(text: str) -> str:
    return text.replace("**", "")
