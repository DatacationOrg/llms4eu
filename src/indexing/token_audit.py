from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol

from src.shared.indexers import build_indexer


class Tokenizer(Protocol):
    model_max_length: int

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ProviderTokenizer:
    provider: str
    model_name: str
    configured_limit: int
    effective_limit: int
    document_prompt: str
    tokenizer: Tokenizer


@dataclass(frozen=True)
class TextMeasurement:
    item_id: str
    provider: str
    version: str
    language: str | None
    character_count: int
    embedding_character_count: int
    canonical_tokens: int
    token_count: int
    prompt_tokens: int
    special_tokens: int
    overhead_tokens: int
    limit: int
    lost_tokens: int

    @property
    def tokens_per_character(self) -> float:
        return self.token_count / self.character_count if self.character_count else 0.0


@dataclass(frozen=True)
class Distribution:
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float


@dataclass(frozen=True)
class GroupSummary:
    provider: str
    version: str
    language: str
    count: int
    characters: Distribution
    tokens: Distribution
    tokens_per_character: float
    over_limit_count: int
    over_limit_rate: float
    total_lost_tokens: int
    max_lost_tokens: int
    mean_overhead_tokens: float
    max_overhead_tokens: int
    worst_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class LanguagePair:
    id: str
    sl: str
    en: str
    source: str
    reviewed_by: str


@dataclass(frozen=True)
class PairMeasurement:
    pair_id: str
    provider: str
    sl_characters: int
    en_characters: int
    sl_tokens: int
    en_tokens: int

    @property
    def sl_tokens_per_character(self) -> float:
        return self.sl_tokens / self.sl_characters

    @property
    def en_tokens_per_character(self) -> float:
        return self.en_tokens / self.en_characters

    @property
    def sl_characters_per_token(self) -> float:
        return self.sl_characters / self.sl_tokens

    @property
    def en_characters_per_token(self) -> float:
        return self.en_characters / self.en_tokens

    @property
    def token_ratio_sl_to_en(self) -> float:
        return self.sl_tokens / self.en_tokens


def load_provider_tokenizer(provider: str, config: dict) -> ProviderTokenizer:
    indexer = build_indexer(provider, config)
    if not getattr(indexer, "local_files_only", True):
        raise RuntimeError(f"Token audit requires local_files_only for {provider}")
    model = indexer._model()
    tokenizer = model.tokenizer
    configured_limit = getattr(indexer, "max_seq_length", None)
    if configured_limit is None:
        configured_limit = int(model.max_seq_length)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", configured_limit))
    credible_limits = [int(configured_limit), int(model.max_seq_length)]
    if tokenizer_limit < 1_000_000_000:
        credible_limits.append(tokenizer_limit)
    prompts = model.prompts or {}
    return ProviderTokenizer(
        provider=provider,
        model_name=indexer.model_name,
        configured_limit=int(configured_limit),
        effective_limit=min(credible_limits),
        document_prompt=prompts.get("document", ""),
        tokenizer=tokenizer,
    )


def measure_text(
    runtime: ProviderTokenizer,
    *,
    item_id: str,
    version: str,
    text: str,
    canonical_text: str | None = None,
    language: str | None = None,
) -> TextMeasurement:
    canonical_text = text if canonical_text is None else canonical_text
    canonical_tokens = _token_count(
        runtime.tokenizer, canonical_text, add_special_tokens=False
    )
    content_tokens = _token_count(runtime.tokenizer, text, add_special_tokens=False)
    prompted_tokens = _token_count(
        runtime.tokenizer,
        runtime.document_prompt + text,
        add_special_tokens=False,
    )
    token_count = _token_count(
        runtime.tokenizer,
        runtime.document_prompt + text,
        add_special_tokens=True,
    )
    return TextMeasurement(
        item_id=item_id,
        provider=runtime.provider,
        version=version,
        language=language,
        character_count=len(canonical_text),
        embedding_character_count=len(text),
        canonical_tokens=canonical_tokens,
        token_count=token_count,
        prompt_tokens=prompted_tokens - content_tokens,
        special_tokens=token_count - prompted_tokens,
        overhead_tokens=token_count - canonical_tokens,
        limit=runtime.effective_limit,
        lost_tokens=max(0, token_count - runtime.effective_limit),
    )


def summarize_measurements(
    measurements: list[TextMeasurement],
) -> list[GroupSummary]:
    groups: dict[tuple[str, str, str], list[TextMeasurement]] = defaultdict(list)
    for measurement in measurements:
        key = (
            measurement.provider,
            measurement.version,
            measurement.language or "unknown",
        )
        groups[key].append(measurement)

    summaries = []
    for (provider, version, language), rows in sorted(groups.items()):
        over_limit = [row for row in rows if row.lost_tokens]
        worst = sorted(
            rows,
            key=lambda row: (-row.lost_tokens, -row.token_count, row.item_id),
        )[:5]
        summaries.append(
            GroupSummary(
                provider=provider,
                version=version,
                language=language,
                count=len(rows),
                characters=_distribution([row.character_count for row in rows]),
                tokens=_distribution([row.token_count for row in rows]),
                tokens_per_character=_safe_ratio(
                    sum(row.token_count for row in rows),
                    sum(row.character_count for row in rows),
                ),
                over_limit_count=len(over_limit),
                over_limit_rate=len(over_limit) / len(rows),
                total_lost_tokens=sum(row.lost_tokens for row in rows),
                max_lost_tokens=max(row.lost_tokens for row in rows),
                mean_overhead_tokens=sum(row.overhead_tokens for row in rows)
                / len(rows),
                max_overhead_tokens=max(row.overhead_tokens for row in rows),
                worst_item_ids=tuple(row.item_id for row in worst),
            )
        )
    return summaries


def compare_language_pairs(
    runtime: ProviderTokenizer,
    pairs: list[LanguagePair],
) -> list[PairMeasurement]:
    rows = []
    for pair in sorted(pairs, key=lambda item: item.id):
        sl_tokens = _token_count(runtime.tokenizer, pair.sl, add_special_tokens=False)
        en_tokens = _token_count(runtime.tokenizer, pair.en, add_special_tokens=False)
        rows.append(
            PairMeasurement(
                pair_id=pair.id,
                provider=runtime.provider,
                sl_characters=len(pair.sl),
                en_characters=len(pair.en),
                sl_tokens=sl_tokens,
                en_tokens=en_tokens,
            )
        )
    return rows


def load_language_pairs(path: Path, expected_count: int = 30) -> list[LanguagePair]:
    if not path.exists():
        raise ValueError(f"Reviewed language-pair fixture is unavailable: {path}")
    pairs = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on fixture line {line_number}: {error}"
            ) from error
        required = {"id", "sl", "en", "source", "reviewed_by"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(
                f"Fixture line {line_number} is missing: {', '.join(missing)}"
            )
        if not all(
            isinstance(row[field], str) and row[field].strip() for field in required
        ):
            raise ValueError(
                f"Fixture line {line_number} has an empty or non-string field"
            )
        pairs.append(LanguagePair(**{field: row[field] for field in required}))
    ids = [pair.id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Language-pair fixture IDs must be unique")
    if len(pairs) != expected_count:
        raise ValueError(
            f"Language-pair fixture must contain {expected_count} pairs; found {len(pairs)}"
        )
    return sorted(pairs, key=lambda pair: pair.id)


def format_corpus_table(summaries: list[GroupSummary]) -> str:
    lines = [
        "| Provider | Version | Language | N | Chars p50/p90/p95/p99/max | "
        "Tokens p50/p90/p95/p99/max | Tok/char | Over limit | Lost total/max | "
        "Overhead mean/max | Worst IDs |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        summaries, key=lambda item: (item.provider, item.version, item.language)
    ):
        lines.append(
            f"| {row.provider} | {row.version} | {row.language} | {row.count} | "
            f"{_format_distribution(row.characters)} | {_format_distribution(row.tokens)} | "
            f"{row.tokens_per_character:.4f} | {row.over_limit_count} "
            f"({row.over_limit_rate:.2%}) | {row.total_lost_tokens}/{row.max_lost_tokens} | "
            f"{row.mean_overhead_tokens:.1f}/{row.max_overhead_tokens} | "
            f"{', '.join(row.worst_item_ids)} |"
        )
    return "\n".join(lines)


def format_pair_table(rows: list[PairMeasurement]) -> str:
    lines = [
        "| Provider | Pairs | SL tok/char | EN tok/char | SL char/tok | "
        "EN char/tok | SL/EN token ratio p50/p90/p95/p99/max |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    by_provider: dict[str, list[PairMeasurement]] = defaultdict(list)
    for row in rows:
        by_provider[row.provider].append(row)
    for provider, provider_rows in sorted(by_provider.items()):
        sl_characters = sum(row.sl_characters for row in provider_rows)
        en_characters = sum(row.en_characters for row in provider_rows)
        sl_tokens = sum(row.sl_tokens for row in provider_rows)
        en_tokens = sum(row.en_tokens for row in provider_rows)
        ratios = _distribution([row.token_ratio_sl_to_en for row in provider_rows])
        lines.append(
            f"| {provider} | {len(provider_rows)} | {sl_tokens / sl_characters:.4f} | "
            f"{en_tokens / en_characters:.4f} | {sl_characters / sl_tokens:.4f} | "
            f"{en_characters / en_tokens:.4f} | {_format_distribution(ratios)} |"
        )
    return "\n".join(lines)


def _token_count(tokenizer: Tokenizer, text: str, *, add_special_tokens: bool) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def _distribution(values: list[int | float]) -> Distribution:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    ordered = sorted(values)
    return Distribution(
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


def _percentile(ordered: list[int | float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _format_distribution(distribution: Distribution) -> str:
    values = (
        distribution.p50,
        distribution.p90,
        distribution.p95,
        distribution.p99,
        distribution.maximum,
    )
    return "/".join(f"{value:g}" for value in values)
