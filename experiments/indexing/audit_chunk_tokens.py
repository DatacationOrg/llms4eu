from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

from _cli import run_cli
from src.indexing.chunk_text import CHUNK_VERSIONS, chunk_text_representation
from src.indexing.token_audit import (
    PairMeasurement,
    ProviderTokenizer,
    TextMeasurement,
    compare_language_pairs,
    format_corpus_table,
    format_pair_table,
    load_language_pairs,
    load_provider_tokenizer,
    measure_text,
    summarize_measurements,
)
from src.preprocess.chunks import CONFIG as CHUNK_CONFIG
from src.shared.env import ROOT, load_local_env, load_yaml
from src.vector_store.chunks import load_chunks

INDEX_CONFIG = load_yaml(ROOT / "src" / "indexing" / "config.yaml")
DEFAULT_PROVIDERS = ("qwen", "qwen4b", "nemotron")
DEFAULT_VERSIONS = ("v1", "v2")
DEFAULT_PAIRS = ROOT / "experiments" / "indexing" / "data" / "sl_en_tourism_pairs.jsonl"
DEFAULT_OUTPUT = ROOT / "docs" / f"chunk-token-audit-{date.today().isoformat()}.md"


def main() -> None:
    load_local_env()
    args = _parse_args()
    providers = _selected_values(args.providers, DEFAULT_PROVIDERS, "provider")
    versions = _selected_values(args.versions, DEFAULT_VERSIONS, "version")
    _validate_selections(providers, versions)
    languages = _optional_values(args.languages)
    chunks = [
        chunk for chunk in load_chunks() if not languages or chunk.language in languages
    ]
    if not chunks:
        raise RuntimeError("No SQLite chunks matched the requested filters")

    runtimes = [_load_runtime(provider) for provider in providers]
    measurements = _measure_corpus(runtimes, versions, chunks)
    pair_rows, pair_status = _measure_pairs(runtimes, Path(args.pairs))
    report = _format_report(
        runtimes=runtimes,
        versions=versions,
        measurements=measurements,
        pair_rows=pair_rows,
        pair_status=pair_status,
        command=" ".join(sys.argv),
        chunk_count=len(chunks),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nAudit report saved to: {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current SQLite chunks with locally cached embedding tokenizers. "
            "The command never computes embeddings or downloads models."
        )
    )
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help="Comma-separated providers (default: qwen,qwen4b,nemotron).",
    )
    parser.add_argument(
        "--versions",
        default=",".join(DEFAULT_VERSIONS),
        help="Comma-separated chunk representations (default: v1,v2).",
    )
    parser.add_argument(
        "--languages",
        help="Optional comma-separated source-language filter.",
    )
    parser.add_argument(
        "--pairs",
        default=str(DEFAULT_PAIRS),
        help="Reviewed 30-pair Slovenian/English JSONL fixture.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def _selected_values(value: str, defaults: tuple[str, ...], label: str) -> list[str]:
    values = _optional_values(value)
    if not values:
        raise ValueError(
            f"At least one {label} is required; defaults are {','.join(defaults)}"
        )
    return values


def _optional_values(value: str | None) -> list[str]:
    return list(
        dict.fromkeys(part.strip() for part in (value or "").split(",") if part.strip())
    )


def _validate_selections(providers: list[str], versions: list[str]) -> None:
    available_providers = set(INDEX_CONFIG["providers"])
    unknown_providers = sorted(set(providers) - available_providers)
    if unknown_providers:
        raise ValueError(f"Unknown providers: {', '.join(unknown_providers)}")
    unknown_versions = sorted(set(versions) - set(CHUNK_VERSIONS))
    if unknown_versions:
        raise ValueError(f"Unknown versions: {', '.join(unknown_versions)}")


def _load_runtime(provider: str) -> ProviderTokenizer:
    try:
        return load_provider_tokenizer(provider, INDEX_CONFIG)
    except Exception as error:
        raise RuntimeError(
            f"Could not load locally cached tokenizer/model for {provider}. "
            "The audit does not download model files; cache the configured model "
            f"explicitly and retry. Original error: {error}"
        ) from error


def _measure_corpus(
    runtimes: list[ProviderTokenizer], versions: list[str], chunks: list
) -> list[TextMeasurement]:
    rows = []
    for runtime in runtimes:
        for version in versions:
            representation = chunk_text_representation(version)
            for chunk in chunks:
                rows.append(
                    measure_text(
                        runtime,
                        item_id=chunk.id,
                        version=version,
                        text=representation.text_for_embedding(chunk),
                        canonical_text=chunk.text,
                        language=chunk.language,
                    )
                )
    return rows


def _measure_pairs(
    runtimes: list[ProviderTokenizer], path: Path
) -> tuple[list[PairMeasurement], str]:
    try:
        pairs = load_language_pairs(path)
    except ValueError as error:
        return [], str(error)
    return [
        row for runtime in runtimes for row in compare_language_pairs(runtime, pairs)
    ], "available"


def _format_report(
    *,
    runtimes: list[ProviderTokenizer],
    versions: list[str],
    measurements: list[TextMeasurement],
    pair_rows: list[PairMeasurement],
    pair_status: str,
    command: str,
    chunk_count: int,
) -> str:
    lines = [
        f"# Chunk Token Audit {date.today().isoformat()}",
        "",
        "## Run Configuration",
        "",
        f"- Command: `{command}`",
        f"- SQLite chunks scanned: {chunk_count}",
        f"- Representations: {', '.join(versions)}",
        "- Token accounting: document prompt + representation text + special tokens, "
        "with truncation disabled",
        f"- Chunk target/max/min characters: {CHUNK_CONFIG['chunk_target_chars']}/"
        f"{CHUNK_CONFIG['chunk_max_chars']}/{CHUNK_CONFIG['chunk_min_chars']}",
        "- Models: "
        + "; ".join(
            f"{runtime.provider}={runtime.model_name} "
            f"(configured {runtime.configured_limit}, effective {runtime.effective_limit})"
            for runtime in runtimes
        ),
        "",
        "## Corpus Results",
        "",
        format_corpus_table(summarize_measurements(measurements)),
        "",
        "Character percentiles use canonical chunk text. Token percentiles use the "
        "effective embedding input. Overhead includes representation metadata, the "
        "model's document prompt, and special tokens.",
        "",
        "## Worst Inputs",
        "",
        _format_worst_inputs(measurements),
        "",
        "## Slovenian/English Pairs",
        "",
    ]
    if pair_rows:
        lines.extend(
            [
                format_pair_table(pair_rows),
                "",
                "Pairs contain passage text only; representation metadata is excluded. "
                "Fewer characters per token means more tokens for a fixed character budget.",
            ]
        )
    else:
        lines.append(
            "Blocked pending a trustworthy 30-pair reviewed translation fixture. "
            f"No review status was inferred or fabricated. Detail: {pair_status}"
        )
    lines.extend(["", "## Recommendation", "", _recommendation(measurements)])
    return "\n".join(lines)


def _format_worst_inputs(measurements: list[TextMeasurement]) -> str:
    rows = sorted(
        measurements,
        key=lambda row: (
            -row.lost_tokens,
            -row.token_count,
            row.item_id,
            row.provider,
            row.version,
        ),
    )[:10]
    lines = [
        "| Provider | Version | Chunk ID | Tokens | Limit | Lost | Overhead |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.provider} | {row.version} | {row.item_id} | {row.token_count} | "
        f"{row.limit} | {row.lost_tokens} | {row.overhead_tokens} |"
        for row in rows
    )
    return "\n".join(lines)


def _recommendation(measurements: list[TextMeasurement]) -> str:
    over_limit = [row for row in measurements if row.lost_tokens]
    rate = len(over_limit) / len(measurements)
    if not over_limit:
        return (
            "Keep current chunking for tokenizer capacity: no measured embedding inputs "
            "exceed the selected providers' effective limits. Evaluate no-overlap "
            "separately as a boundary-recall issue and short-fragment dropping as content loss."
        )
    if rate <= 0.001 and max(row.lost_tokens for row in over_limit) <= 8:
        return (
            f"Keep current chunking: {len(over_limit)}/{len(measurements)} inputs "
            f"({rate:.3%}) exceed a limit and the lost-token tail is at most 8 tokens. "
            "Treat no-overlap and short-fragment dropping as separate retrieval/content risks."
        )
    return (
        f"Do not change stable chunk IDs from this audit alone: {len(over_limit)}/"
        f"{len(measurements)} inputs ({rate:.2%}) exceed a provider limit. Run a future "
        "versioned, provider-aware token-budget experiment unless a lower shared character "
        "target is shown to satisfy every provider."
    )


if __name__ == "__main__":
    run_cli(main)
