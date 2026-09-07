from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def score_rankings(
    rows: list[dict],
    rankings: dict[str, list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
    mrr_k: int = 10,
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
        first_rank = _first_relevant_rank(ranked[:mrr_k], relevant)
        if first_rank:
            scores[f"mrr@{mrr_k}"] += 1 / first_rank

    total = len(question_ids)
    if total == 0:
        return scores
    return {name: value / total for name, value in scores.items()}


def _first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return index
    return None


Span = tuple[str, int, int]
"""A page id and a half-open character range inside that page."""


def score_span_rankings(
    anchors: dict[str, Span],
    chunk_spans: dict[str, Span],
    rankings: dict[str, list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
    budgets: tuple[int, ...] = (),
) -> dict[str, float]:
    """Retrieval quality in characters of answer text, not in whole chunks.

    `hit@k` and `recall@k` count chunks, which hands a systematic advantage to
    large ones: a 4,000-character chunk is four times likelier to contain any
    given answer than a 1,000-character chunk at identical retrieval quality. Rank
    chunkings on those and the ranking largely reproduces chunk size.

    Character overlap against the answer span removes that. Recall is how much of
    the answer the retrieved text covers, precision how much of the retrieved text
    is answer, IoU the two together. A large chunk still earns its recall, and
    pays for the characters it made the reader wade through to deliver it.

    `budgets` scores the same rankings under a fixed character allowance, filled in
    rank order. That is the comparison that decides an indexing choice, because
    what is actually scarce downstream is the generator's context window rather
    than k.

    Averaged over questions that have both an anchor and a ranking. A question
    with no anchor has no ground truth to score against; one that was not run
    would otherwise count as a silent miss.
    """
    question_ids = [
        question_id
        for question_id, (_, start, end) in anchors.items()
        if question_id in rankings and end > start
    ]
    scores = {
        name: 0.0
        for name in [
            *(f"char_recall@{k}" for k in ks),
            *(f"char_precision@{k}" for k in ks),
            *(f"iou@{k}" for k in ks),
            *(f"budget_recall@{budget}" for budget in budgets),
        ]
    }
    if not question_ids:
        return scores

    for question_id in question_ids:
        anchor = anchors[question_id]
        anchor_length = anchor[2] - anchor[1]
        ranked = rankings[question_id]
        for k in ks:
            covered, retrieved = _overlap(ranked[:k], chunk_spans, anchor)
            union = retrieved + anchor_length - covered
            scores[f"char_recall@{k}"] += covered / anchor_length
            scores[f"char_precision@{k}"] += covered / retrieved if retrieved else 0.0
            scores[f"iou@{k}"] += covered / union if union else 0.0
        for budget in budgets:
            selected = _fill_budget(ranked, chunk_spans, budget)
            covered, _ = _overlap(selected, chunk_spans, anchor)
            scores[f"budget_recall@{budget}"] += covered / anchor_length

    total = len(question_ids)
    return {name: value / total for name, value in scores.items()}


def span_coverage(anchors: dict[str, Span], rankings: dict[str, list[str]]) -> int:
    """How many questions the span metrics are actually averaged over."""
    return sum(
        1
        for question_id, (_, start, end) in anchors.items()
        if question_id in rankings and end > start
    )


def score_store_share(
    chunk_spans: dict[str, Span],
    rankings: dict[str, list[str]],
    store_chars: int,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """What share of the whole index one query pulls back, in percent.

    Every quality metric here rewards a variant for cutting large. `hit@k` and
    `recall@k` do it by making the target chunk a bigger target; `char_recall@k`
    does it more honestly but still hands k slots of 1,024 tokens four times the
    text that k slots of 256 tokens get. This is the other side of that trade,
    priced in the store's own units: at k=10 `tok1024` returns about 2.0% of its
    index and `tok256` about 0.5%, so equal recall out of the two is not equal
    work.

    Chunk lengths are summed *unmerged*, against the summed length of everything
    the variant stores. Both sides then count stored text the same way, which is
    what makes an overlapping cut pay for the copies it keeps: merging the
    numerator while the denominator still held every copy would report overlap as
    free. Characters of *reading*, where overlap genuinely is deduplicated, are
    what `char_precision@k` already measures.

    Averaged over questions that have a ranking. Needs no anchors, so it covers
    the whole labelled set rather than the anchored subset.
    """
    scores = {f"store_share@{k}": 0.0 for k in ks}
    if store_chars <= 0 or not rankings:
        return scores
    for ranked in rankings.values():
        for k in ks:
            retrieved = sum(
                max(chunk_spans[chunk_id][2] - chunk_spans[chunk_id][1], 0)
                for chunk_id in ranked[:k]
                if chunk_id in chunk_spans
            )
            scores[f"store_share@{k}"] += 100.0 * retrieved / store_chars
    return {name: value / len(rankings) for name, value in scores.items()}


def score_retrieval_efficiency(
    scores: dict[str, float], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """`recall_per_share@k`: answer coverage earned per percent of index read.

    The metric that stops chunk size deciding the ranking on its own. A cut can
    buy `char_recall` with size — retrieve four times the text and cover four
    times as much of any answer — and this charges it for that text. Two cuts
    reaching the same recall are separated by what each had to return to get
    there.

    A ratio of the two means rather than the mean of per-question ratios. The
    per-question ratio is undefined whenever a query returns nothing and its
    distribution is long-tailed enough that a handful of near-zero denominators
    would decide the average; the ratio of aggregates is bounded by construction
    and is the number a reader can recompute from the two columns beside it.

    Only pairs where both halves are present are returned, so a run without span
    metrics gains no column of real-looking zeroes.
    """
    efficiency = {}
    for k in ks:
        recall = scores.get(f"char_recall@{k}")
        share = scores.get(f"store_share@{k}")
        if recall is None or share is None or share <= 0:
            continue
        efficiency[f"recall_per_share@{k}"] = recall / share
    return efficiency


def _overlap(
    chunk_ids: list[str], chunk_spans: dict[str, Span], anchor: Span
) -> tuple[int, int]:
    """Characters of `anchor` covered, and characters retrieved in total.

    Chunks are merged per page first. Overlapping variants would otherwise be
    charged twice for text the reader only sees once, which would read as a
    precision penalty for using overlap at all.
    """
    page_id, start, end = anchor
    by_page: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk_id in chunk_ids:
        span = chunk_spans.get(chunk_id)
        if span is not None:
            by_page[span[0]].append((span[1], span[2]))

    covered = 0
    retrieved = 0
    for page, intervals in by_page.items():
        merged = _merge(intervals)
        retrieved += sum(stop - begin for begin, stop in merged)
        if page == page_id:
            covered += sum(
                max(0, min(stop, end) - max(begin, start)) for begin, stop in merged
            )
    return covered, retrieved


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Non-overlapping cover of `intervals`, in order."""
    merged: list[tuple[int, int]] = []
    for begin, stop in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((begin, stop))
    return merged


def _fill_budget(
    ranked: list[str], chunk_spans: dict[str, Span], budget: int
) -> list[str]:
    """Chunks taken in rank order while they fit a character budget.

    A chunk too large for the room left is skipped rather than ending the fill,
    which is what a context assembler does and keeps one oversized chunk near the
    top of the ranking from deciding the score.
    """
    selected: list[str] = []
    used = 0
    for chunk_id in ranked:
        span = chunk_spans.get(chunk_id)
        if span is None:
            continue
        length = max(span[2] - span[1], 0)
        if used + length > budget:
            continue
        selected.append(chunk_id)
        used += length
    return selected


def bold_best_table(
    headers: list[str],
    rows: list[list[str | float]],
    lower_is_better: bool | Iterable[str] = (),
) -> str:
    """Markdown table with the best cell of each numeric column in bold.

    `lower_is_better` names the columns where "best" is the smallest value, and
    takes `True` for a table whose every number is a cost. Cost columns exist
    now that `store_share@k` is reported beside recall, and bolding the largest
    share would recommend the most expensive cut in the grid.
    """
    lower = (
        set(headers)
        if lower_is_better is True
        else set(() if lower_is_better is False else lower_is_better)
    )
    numeric_columns = []
    for index in range(1, len(headers)):
        values = [row[index] for row in rows if isinstance(row[index], int | float)]
        if values:
            best = min(values) if headers[index] in lower else max(values)
            numeric_columns.append((index, best))

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
    
    def _plain(text: str) -> str:
        return text.replace("**", "")
    
    widths = [
        max(len(_plain(row[index]) ) for row in [headers, *rendered_rows])
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
