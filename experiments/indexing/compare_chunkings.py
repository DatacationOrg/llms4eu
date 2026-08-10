"""Compare chunking variants for the same retrieval methods.

Every published retrieval number so far was measured on one chunking
(`chunk_target_chars: 1800`, no overlap, character sized). This runner varies the
chunking instead of the retriever, so chunk size, overlap and size unit become
measured choices rather than assumptions.

Three tables come out of a run:

- **Chunk audit** — per variant, how the indexed documents sit against each
  provider's sequence limit. This is the pre-flight check: a variant whose chunks
  exceed the limit is being silently truncated, and its retrieval scores describe
  a corpus the embedder never fully saw.
- **Per-variant metrics** — each variant scored on its own generated question set.
- **Projected metrics** — every variant scored against the *base* variant's
  approved labels, mapped through character spans. Per-variant question sets are
  not comparable to each other; this column is, and it is reported beside the
  strict numbers rather than merged into them.

Usage:

    uv run python experiments/indexing/compare_chunkings.py \\
        --variants base,tok512 --methods nemotron_hybrid_rerank --limit 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from _cli import run_cli
from src.db.pages import connect_pages, initialize_page_artifacts_db
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import load_eval_rows
from src.eval.metrics import bold_best_table, plain_table, score_rankings
from src.eval.spans import project_relevance
from src.indexing.chunk_text import (
    BASE_CHUNK_VARIANT,
    PageChunk,
    chunk_text_representation,
)
from src.retrieval.methods import build_retriever, missing_retriever_indexes
from src.vector_store.chunks import enabled_provider_names
from src.shared.env import ROOT, load_local_env, load_yaml

DEFAULT_METHODS = ("sparse_rerank", "nemotron_hybrid_rerank", "qwen4b_hybrid_rerank")
DEFAULT_VARIANTS = (BASE_CHUNK_VARIANT,)
DEFAULT_OUTPUT = Path("docs/retrieval-results-chunking.md")
CONFIG = load_yaml(ROOT / "experiments" / "indexing" / "config.yaml")


def audit_providers() -> tuple[str, ...]:
    """Every enabled provider, so no configured embedder escapes the audit.

    A variant is cut with one provider's tokenizer, but it gets indexed by
    others too, and their tokenizers disagree. Auditing all of them is what
    catches a variant that fits the embedder it was built for and overflows
    another.
    """
    return tuple(enabled_provider_names())


def main() -> None:
    load_local_env()
    args = _parse_args()
    initialize_page_artifacts_db()

    variants = _split(args.variants) or list(DEFAULT_VARIANTS)
    methods = _split(args.methods) or list(DEFAULT_METHODS)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )

    _require_variants(variants)
    audit = _chunk_audit(variants, args.chunk_version) if not args.no_audit else []
    if args.audit_only:
        print(_audit_table(audit))
        return

    _require_indexes(methods, variants)
    state = _load_state(checkpoint_path, variants, methods, resume=not args.no_resume)
    results = _run(
        variants=variants,
        methods=methods,
        limit=args.limit,
        category=args.category,
        warmup=_resolve_warmup(args.warmup, args.limit),
        state=state,
        checkpoint_path=checkpoint_path,
        log_every=max(1, args.log_every),
    )

    report = _format_report(
        results=results,
        audit=audit,
        variants=variants,
        methods=methods,
        limit=args.limit,
        category=args.category,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    if checkpoint_path.exists() and not args.keep_checkpoint:
        checkpoint_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="",
        help="Comma separated chunking variants, e.g. base,tok512,tok256",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma separated retrieval methods to run for every variant",
    )
    parser.add_argument("--limit", type=int, help="Questions per variant")
    parser.add_argument("--category", help="Restrict to one question_type")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--chunk-version", default="v1")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only print the token audit; run no retrieval",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the token audit, which loads tokenizers",
    )
    return parser.parse_args()


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_warmup(requested: int | None, limit: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    default = int(CONFIG.get("default_warmup", 5))
    if limit is None:
        return default
    return min(default, max(0, limit - 1))


def _require_variants(variants: list[str]) -> None:
    with connect_pages() as conn:
        stored = {
            row["variant"]
            for row in conn.execute("select distinct variant from page_chunks")
        }
    missing = [variant for variant in variants if variant not in stored]
    if missing:
        commands = "\n".join(
            f"  uv run python -m src.preprocess.chunks --variant {variant} ..."
            for variant in missing
        )
        raise RuntimeError(
            f"No chunks stored for variants: {', '.join(missing)}\n"
            f"Build them first:\n{commands}"
        )


def _require_indexes(methods: list[str], variants: list[str]) -> None:
    missing: dict[str, str] = {}
    for variant in variants:
        for method, requirement in missing_retriever_indexes(methods, variant).items():
            missing[f"{method}@{variant}"] = requirement
    if missing:
        commands = sorted(
            {
                "  uv run python -m src.indexing.chunks --method "
                f"{requirement.split('@')[0].split('#')[0]}"
                f" --chunk-variant {key.rsplit('@', 1)[1]}"
                for key, requirement in missing.items()
            }
        )
        raise RuntimeError(
            "Missing vector indexes:\n"
            + "\n".join(f"  {key}" for key in sorted(missing))
            + "\nBuild them first:\n"
            + "\n".join(commands)
        )


def _chunk_audit(variants: list[str], chunk_version: str) -> list[dict[str, Any]]:
    """Measure indexed document length against each provider's token limit.

    This is the check that exposes silent truncation: a chunk longer than the
    sequence limit is embedded from its prefix only, so part of the corpus is
    unreachable by dense retrieval no matter how good the retriever is.
    """
    from src.shared.tokenizers import provider_token_limit, token_counter

    representation = chunk_text_representation(chunk_version)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        documents = _indexed_documents(variant, representation)
        for provider in audit_providers():
            try:
                count = token_counter(provider)
                limit = provider_token_limit(provider)
            except Exception as error:  # tokenizer unavailable offline
                print(f"skipping audit for {provider}: {error}", flush=True)
                continue
            tokens = sorted(count(text) for text in documents)
            if not tokens:
                continue
            over = sum(1 for value in tokens if value > limit)
            discarded = sum(max(0, value - limit) for value in tokens)
            total = sum(tokens)
            rows.append(
                {
                    "variant": variant,
                    "provider": provider,
                    "chunks": len(tokens),
                    "limit": limit,
                    "median": int(statistics.median(tokens)),
                    "p95": tokens[min(int(len(tokens) * 0.95), len(tokens) - 1)],
                    "max": tokens[-1],
                    "over_limit": f"{100 * over / len(tokens):.1f}%",
                    "text_discarded": f"{100 * discarded / total:.1f}%"
                    if total
                    else "-",
                }
            )
    return rows


def _indexed_documents(variant: str, representation) -> list[str]:
    with connect_pages() as conn:
        rows = conn.execute(
            """
            select c.id, c.page_id, c.chunk_index, c.heading_path, c.text,
                   m.title, m.source, m.page_kind, s.language
            from page_chunks c
            join page_metadata m on m.id = c.page_id
            left join page_sources s on s.source = m.source
            where c.variant = ?
            order by c.id
            """,
            (variant,),
        ).fetchall()
    return [
        representation.text_for_embedding(
            PageChunk(
                id=row["id"],
                page_id=row["page_id"],
                chunk_index=row["chunk_index"],
                heading_path=row["heading_path"],
                text=row["text"],
                title=row["title"],
                source=row["source"],
                language=row["language"],
                page_kind=row["page_kind"],
                variant=variant,
            )
        )
        for row in rows
    ]


def _run(
    *,
    variants: list[str],
    methods: list[str],
    limit: int | None,
    category: str | None,
    warmup: int,
    state: dict[str, Any],
    checkpoint_path: Path,
    log_every: int,
) -> dict[str, Any]:
    base_relevance = _base_relevance()
    results: dict[str, Any] = {"variants": {}}

    for variant in variants:
        questions, relevance = _variant_questions(variant, limit, category)
        if not questions:
            print(f"no questions for variant {variant}, skipping", flush=True)
            continue
        warmup_questions = questions[:warmup]
        timed = questions[warmup:]
        # Both label sets must cover exactly the questions that were run:
        # score_rankings averages over every question it sees relevance for, so
        # a label row for an unrun question silently counts as a miss.
        timed_ids = {str(row["id"]) for row in timed}
        relevance = [row for row in relevance if str(row["question_id"]) in timed_ids]
        projected = [
            row
            for row in project_relevance(base_relevance, BASE_CHUNK_VARIANT, variant)
            if str(row["question_id"]) in timed_ids
        ]
        variant_result = {
            "questions": len(timed),
            "own_labels": len({row["question_id"] for row in relevance}),
            "projected_labels": len({row["question_id"] for row in projected}),
            "methods": {},
        }

        for method in methods:
            key = f"{variant}|{method}"
            method_state = state["runs"].setdefault(
                key, {"rankings": {}, "next_index": 0, "elapsed_seconds": 0.0}
            )
            retriever = build_retriever(method, variant)
            for question in warmup_questions[: max(0, warmup)]:
                retriever.retrieve(question["question"], EVAL_CONFIG["result_limit"])

            for index in range(int(method_state["next_index"]), len(timed)):
                row = timed[index]
                started = time.perf_counter()
                chunks = retriever.retrieve(
                    row["question"], EVAL_CONFIG["result_limit"]
                )
                method_state["elapsed_seconds"] += time.perf_counter() - started
                method_state["rankings"][str(row["id"])] = [
                    chunk.id for chunk in chunks
                ]
                method_state["next_index"] = index + 1
                if (index + 1) % log_every == 0 or index + 1 == len(timed):
                    print(
                        f"{variant}/{method}: {index + 1}/{len(timed)}",
                        flush=True,
                    )
                    _write_state(checkpoint_path, state)
            _write_state(checkpoint_path, state)

            rankings = method_state["rankings"]
            elapsed = float(method_state["elapsed_seconds"])
            variant_result["methods"][method] = {
                "own": _scores(relevance, rankings),
                "projected": _scores(projected, rankings),
                "seconds": elapsed,
                "ms_per_query": (elapsed / len(rankings) * 1000) if rankings else 0.0,
            }
        results["variants"][variant] = variant_result
    return results


def _scores(
    relevance: list[dict],
    rankings: dict[str, list[str]],
) -> dict[str, float] | None:
    """Score one ranking set, or None when there are no labels to score against.

    Returning None keeps a variant without its own generated questions out of
    the table as `-` instead of showing a real-looking 0.000.
    """
    if not relevance or not rankings:
        return None
    metric_ks = tuple(EVAL_CONFIG["metric_ks"])
    return score_rankings(
        relevance,
        rankings,
        ks=metric_ks,
        mrr_k=EVAL_CONFIG["mrr_k"],
        recall_k=max(metric_ks),
    )


def _variant_questions(
    variant: str,
    limit: int | None,
    category: str | None,
) -> tuple[list[dict], list[dict]]:
    """A variant's own generated question set, falling back to the base set.

    A variant with no generated questions yet is still worth running: it is
    scored on the projected base labels alone.
    """
    questions, relevance = load_eval_rows(limit, category, variant)
    if questions:
        return questions, relevance
    if variant == BASE_CHUNK_VARIANT:
        return load_eval_rows(limit, category)
    print(
        f"variant {variant} has no generated questions; "
        "reusing the base question texts and scoring by projection only",
        flush=True,
    )
    questions, _ = load_eval_rows(limit, category, BASE_CHUNK_VARIANT)
    if not questions:
        questions, _ = load_eval_rows(limit, category)
    return questions, []


def _base_relevance() -> list[dict]:
    with connect_pages() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                select r.question_id, r.chunk_id
                from eval_relevant_chunks r
                join page_chunks c on c.id = r.chunk_id
                join eval_questions q on q.id = r.question_id
                where c.variant = ? and q.approved = 1
                """,
                (BASE_CHUNK_VARIANT,),
            )
        ]


def _load_state(
    checkpoint_path: Path,
    variants: list[str],
    methods: list[str],
    resume: bool,
) -> dict[str, Any]:
    signature = {"variants": variants, "methods": methods}
    if resume and checkpoint_path.exists():
        stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if stored.get("signature") == signature:
            print(f"resuming from {checkpoint_path}", flush=True)
            return stored
        print("checkpoint does not match this run, starting fresh", flush=True)
    return {"signature": signature, "runs": {}}


def _write_state(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")


def _format_report(
    *,
    results: dict[str, Any],
    audit: list[dict[str, Any]],
    variants: list[str],
    methods: list[str],
    limit: int | None,
    category: str | None,
) -> str:
    metric_ks = tuple(EVAL_CONFIG["metric_ks"])
    standard_k = max(metric_ks)
    score_names = [
        *(f"hit@{k}" for k in metric_ks),
        f"recall@{standard_k}",
        f"mrr@{EVAL_CONFIG['mrr_k']}",
        f"ndcg@{EVAL_CONFIG['mrr_k']}",
    ]

    lines = [
        "# Chunking comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Variants: {', '.join(variants)}",
        f"Methods: {', '.join(methods)}",
        f"Questions per variant: {limit if limit else 'all'}"
        + (f" (category {category})" if category else ""),
        "",
    ]

    if audit:
        lines += [
            "## Chunk audit",
            "",
            "Indexed document length against each provider's sequence limit. A"
            " non-zero `over limit` means part of the corpus is silently"
            " truncated before it is ever embedded.",
            "",
            _audit_table(audit),
            "",
        ]

    lines += ["## Per-variant metrics", "", _metric_note("own"), ""]
    lines.append(_metric_table(results, methods, score_names, "own"))
    lines += [
        "",
        "## Projected metrics (base labels via character spans)",
        "",
        _metric_note("projected"),
        "",
        _metric_table(results, methods, score_names, "projected"),
        "",
        "## Speed",
        "",
        _speed_table(results, methods),
    ]
    return "\n".join(lines)


def _metric_note(kind: str) -> str:
    if kind == "own":
        return (
            "Each variant scored on its own generated questions and gold chunk"
            " ids. Comparable across methods within a variant, not across"
            " variants: the question sets differ."
        )
    return (
        "Every variant scored against the base variant's approved labels,"
        " mapped through chunk character spans. A hit means a retrieved chunk"
        " covers the gold span. This is the cross-variant comparison; it is a"
        " span proxy and never replaces the strict numbers above."
    )


def _metric_table(
    results: dict[str, Any],
    methods: list[str],
    score_names: list[str],
    kind: str,
) -> str:
    rows: list[list[Any]] = []
    for variant, variant_result in results["variants"].items():
        for method in methods:
            scores = variant_result["methods"].get(method)
            if not scores:
                continue
            values = scores[kind]
            rows.append(
                [
                    f"{method} @ {variant}",
                    *(values.get(name, "-") for name in score_names),
                ]
            )
    if not rows:
        return "_No scored questions._"
    return bold_best_table(["method @ variant", *score_names], rows)


def _speed_table(results: dict[str, Any], methods: list[str]) -> str:
    rows: list[list[Any]] = []
    for variant, variant_result in results["variants"].items():
        for method in methods:
            scores = variant_result["methods"].get(method)
            if not scores:
                continue
            rows.append(
                [
                    f"{method} @ {variant}",
                    f"{scores['seconds']:.1f}",
                    f"{scores['ms_per_query']:.1f}",
                    variant_result["questions"],
                    variant_result["own_labels"],
                    variant_result["projected_labels"],
                ]
            )
    if not rows:
        return "_No timings._"
    return plain_table(
        [
            "method @ variant",
            "seconds",
            "ms/query",
            "questions",
            "own labels",
            "projected labels",
        ],
        rows,
    )


def _audit_table(audit: list[dict[str, Any]]) -> str:
    if not audit:
        return "_No audit rows._"
    headers = [
        "variant",
        "provider",
        "chunks",
        "limit",
        "median tok",
        "p95 tok",
        "max tok",
        "over limit",
        "text discarded",
    ]
    rows = [
        [
            row["variant"],
            row["provider"],
            row["chunks"],
            row["limit"],
            row["median"],
            row["p95"],
            row["max"],
            row["over_limit"],
            row["text_discarded"],
        ]
        for row in audit
    ]
    return plain_table(headers, rows)


if __name__ == "__main__":
    run_cli(main)
