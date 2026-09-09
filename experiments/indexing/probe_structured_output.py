"""Measure whether a judge model can emit each agent's schema on real prompts.

Every agentic method depends on a local model returning one Pydantic shape, and a
failure is silent in the benchmark: the retriever falls back (`expand`, or the
BM25 shortlist) and the cell scores something other than the method under test.
The 2026-09-01 run reported 7.1% failures after `judge_retries: 3`; because
temperature is 0 those retries are correlated, so the first-attempt rate is the
number that actually describes the model, and it is much worse.

The probe has two prompt sources. The synthetic one (`--samples N`) repeats one
short hand-written prompt per schema and is a smoke test only: gemma4:31b scored
10/10 on it and then failed 136 times in the first real cell. `--from-eval N`
takes the first N approved questions from `PAGES_DB_PATH` in benchmark order,
runs the real first-stage retriever, and builds each schema's prompt exactly as
the retriever would on its first attempt. That is the number to gate a rerun on.

For every call it records the prompt length, the stop reason and the size of the
thinking block, so a failure can be tied to a cause (context, output cap, model)
rather than a folklore comment in config.yaml.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time

from src.eval.evaluate import load_eval_rows
from src.retrieval.base import RankedChunk
from src.retrieval.methods import build_retriever
from src.retrieval.retrievers.agentic import ChunkSufficiency, _sufficiency_prompt
from src.retrieval.retrievers.agentic_tools import ToolAction, _action_prompt
from src.retrieval.retrievers.dci import (
    CorpusAction,
    _corpus_prompt,
    _shortlist_documents,
)
from src.preprocess.locations import PageForLocating, PagePlaces, page_places_prompt
from src.retrieval.retrievers.page_tools import PageRef, page_refs_for_chunks
from src.shared.geo_resolver import QUERY_LOCATION_PROMPT, ExtractedQueryLocation
from src.retrieval.workspace import PageDocument, load_workspace
from src.shared.env import ROOT, load_yaml
from src.shared.llm import structured_local_model

CONFIG = load_yaml(ROOT / "src" / "retrieval" / "config.yaml")
METHODS = ("function_calling", "json_schema")
SCHEMAS = (
    "ChunkSufficiency",
    "ToolAction",
    "CorpusAction",
    "ExtractedQueryLocation",
    "PagePlaces",
)
# Prompt-length buckets in tokens, as counted by Ollama for the judge model.
BUCKETS = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class Case:
    schema: type
    prompts: list[str]
    num_predict: int
    source: str


@dataclass
class Attempt:
    ok: bool
    error: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    done_reason: str | None
    thinking_chars: int
    seconds: float
    # What the model decided, so two configurations that both parse can be
    # compared on their verdicts (a judge that always says sufficient parses fine).
    parsed: dict | None = None


def _chunks() -> list[RankedChunk]:
    return [
        RankedChunk(
            id="1e691d23-4040-5f7f-9b5c-f8103f086722:5",
            score=0.91,
            text=(
                "Ašič je imel svoj laboratorij v samostanu, kjer je pripravljal "
                "zdravilne pripravke iz zelišč."
            ),
        ),
        RankedChunk(
            id="1e691d23-4040-5f7f-9b5c-f8103f086722:1",
            score=0.72,
            text="Pater Simon Ašič se je rodil leta 1906 v Prevaljah.",
        ),
    ]


def _synthetic_cases(samples: int) -> dict[str, Case]:
    """One short hand-written prompt per schema, repeated. A smoke test only."""
    query = "Kje je imel Ašič svoj laboratorij?"
    chunks = _chunks()
    pages = {
        chunk.id: PageRef(
            page_id="1e691d23-4040-5f7f-9b5c-f8103f086722",
            title="Simon Ašič",
            url="https://x.test/asic",
            source="svn_biography",
            chunk_count=9,
        )
        for chunk in chunks
    }
    documents = [
        PageDocument(
            page_id="1e691d23-4040-5f7f-9b5c-f8103f086722",
            path="svn_biography/1e691d23-4040-5f7f-9b5c-f8103f086722.md",
            title="Simon Ašič",
            source="svn_biography",
            url="https://x.test/asic",
            line_count=64,
            spans=(),
        )
    ]
    return {
        "ChunkSufficiency": Case(
            ChunkSufficiency,
            [_sufficiency_prompt(query, chunks)] * samples,
            CONFIG["agentic_judge_num_predict"],
            "synthetic",
        ),
        "ToolAction": Case(
            ToolAction,
            [_action_prompt(query, chunks, pages, [])] * samples,
            CONFIG["agentic_judge_num_predict"],
            "synthetic",
        ),
        "CorpusAction": Case(
            CorpusAction,
            [_corpus_prompt(query, documents, [])] * samples,
            CONFIG["dci_num_predict"],
            "synthetic",
        ),
        # The geo schemas are small; 512 output tokens is what the resolver and
        # the locate pass hand the model.
        "ExtractedQueryLocation": Case(
            ExtractedQueryLocation,
            [QUERY_LOCATION_PROMPT.format(query="Kateri gradovi so blizu Brestanice?")]
            * samples,
            512,
            "synthetic",
        ),
        "PagePlaces": Case(
            PagePlaces,
            [
                page_places_prompt(
                    PageForLocating(
                        id="fixture",
                        source="castle_rajhenburg",
                        url="https://x.test/grad",
                        title="Grad Rajhenburg",
                        markdown=(
                            "Grad Rajhenburg stoji na skalnem pomolu nad Brestanico "
                            "v občini Krško. Je najstarejši ohranjeni grad v Sloveniji."
                        ),
                    )
                )
            ]
            * samples,
            512,
            "synthetic",
        ),
    }


def _eval_cases(
    count: int,
    schema_names: list[str],
    first_stage: str,
    variant: str,
) -> dict[str, Case]:
    """Prompts the retrievers would issue on their first attempt for real questions.

    Question order is `load_eval_rows`' order, which is the benchmark's, so
    `--from-eval 50` probes the first 50 questions the sweep would score.
    """
    questions, _ = load_eval_rows(limit=count, variant=variant)
    if len(questions) < count:
        print(f"warning: only {len(questions)} labelled questions for {variant}")
    queries = [row["question"] for row in questions]
    cases: dict[str, Case] = {}

    if {"ChunkSufficiency", "ToolAction"} & set(schema_names):
        retriever = build_retriever(first_stage, variant=variant)
        # The agents' first call sees `agentic_initial_limit` chunks.
        batches = retriever.retrieve_batch(queries, CONFIG["agentic_initial_limit"])
        ranked = [batches[index] for index in range(len(queries))]
        if "ChunkSufficiency" in schema_names:
            cases["ChunkSufficiency"] = Case(
                ChunkSufficiency,
                [_sufficiency_prompt(q, c) for q, c in zip(queries, ranked)],
                CONFIG["agentic_judge_num_predict"],
                f"eval:{first_stage}",
            )
        if "ToolAction" in schema_names:
            prompts = []
            for query, chunks in zip(queries, ranked):
                pages = page_refs_for_chunks([chunk.id for chunk in chunks])
                prompts.append(_action_prompt(query, chunks, pages, []))
            cases["ToolAction"] = Case(
                ToolAction,
                prompts,
                CONFIG["agentic_judge_num_predict"],
                f"eval:{first_stage}",
            )

    if "ExtractedQueryLocation" in schema_names:
        cases["ExtractedQueryLocation"] = Case(
            ExtractedQueryLocation,
            [QUERY_LOCATION_PROMPT.format(query=query) for query in queries],
            512,
            "eval:questions",
        )

    if "CorpusAction" in schema_names:
        sparse = build_retriever("sparse", variant=variant)
        workspace = load_workspace(variant=variant)
        batches = sparse.retrieve_batch(queries, CONFIG["dci_shortlist_k"])
        prompts = []
        for index, query in enumerate(queries):
            documents = _shortlist_documents(
                workspace, batches[index], CONFIG["dci_max_documents"]
            )
            prompts.append(_corpus_prompt(query, documents, []))
        cases["CorpusAction"] = Case(
            CorpusAction, prompts, CONFIG["dci_num_predict"], "eval:sparse"
        )
    return cases


def _attempt(model, prompt: str) -> Attempt:
    """One first attempt. Retries at temperature 0 repeat the same failure."""
    started = time.perf_counter()
    try:
        outcome = model.invoke(prompt)
    except Exception as exc:  # transport failures escape include_raw
        return Attempt(
            False,
            type(exc).__name__,
            None,
            None,
            None,
            0,
            time.perf_counter() - started,
        )
    seconds = time.perf_counter() - started
    raw = outcome.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}
    meta = getattr(raw, "response_metadata", None) or {}
    thinking = (getattr(raw, "additional_kwargs", None) or {}).get(
        "reasoning_content"
    ) or ""
    error = outcome.get("parsing_error")
    parsed = outcome.get("parsed")
    if error is not None:
        name = type(error).__name__
    elif parsed is None:
        # function_calling returns None when the model answers in prose
        # instead of calling the tool.
        name = "NoToolCall"
    else:
        name = None
    return Attempt(
        ok=name is None,
        error=name,
        prompt_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        done_reason=meta.get("done_reason"),
        thinking_chars=len(thinking),
        seconds=seconds,
        parsed=_compact(parsed),
    )


def _compact(parsed) -> dict | None:
    if parsed is None:
        return None
    data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    return {
        key: (value[:120] if isinstance(value, str) else value)
        for key, value in data.items()
        if value not in (None, "", [])
    }


def _parse_reasoning(value: str) -> bool | str | None:
    """`--reasoning` values: gpt-oss effort levels, or on/off for boolean models.

    Gemma treats `think` as a switch, so `false` is the only way to probe it
    without a scratchpad, and that is the configuration that works with
    `function_calling` (thinking on gives 0% there, thinking off gives 100%).
    """
    lowered = value.strip().lower()
    if lowered in ("false", "off", "no"):
        return False
    if lowered in ("true", "on", "yes"):
        return True
    if lowered == "none":
        return None
    return value.strip()


def _bucket(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    for edge in BUCKETS:
        if tokens < edge:
            return f"<{edge}"
    return f">={BUCKETS[-1]}"


def probe(
    model_id: str,
    cases: dict[str, Case],
    schema_names: list[str],
    methods: list[str],
    reasonings: list[str],
    num_predict: int | None,
) -> list[dict]:
    rows = []
    for schema_name in schema_names:
        case = cases[schema_name]
        for method in methods:
            for reasoning in reasonings:
                model = structured_local_model(
                    model_id,
                    case.schema,
                    reasoning=_parse_reasoning(reasoning),
                    num_ctx=CONFIG["agentic_judge_num_ctx"],
                    num_predict=num_predict or case.num_predict,
                    method=method,
                    include_raw=True,
                )
                attempts = [_attempt(model, prompt) for prompt in case.prompts]
                rows.append(
                    _row(
                        model_id,
                        schema_name,
                        method,
                        reasoning,
                        case,
                        num_predict or case.num_predict,
                        attempts,
                    )
                )
                print(_line(rows[-1]), flush=True)
    return rows


def _row(model_id, schema_name, method, reasoning, case, num_predict, attempts):
    errors = Counter(a.error for a in attempts if a.error)
    reasons = Counter(a.done_reason or "unknown" for a in attempts)
    by_bucket: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        by_bucket[_bucket(attempt.prompt_tokens)].append(attempt)
    buckets = {
        bucket: {"n": len(group), "ok": sum(a.ok for a in group)}
        for bucket, group in sorted(
            by_bucket.items(), key=lambda item: _bucket_order(item[0])
        )
    }
    known = [a.prompt_tokens for a in attempts if a.prompt_tokens is not None]
    failed_thinking = [a.thinking_chars for a in attempts if not a.ok]
    ok_thinking = [a.thinking_chars for a in attempts if a.ok]
    samples = len(attempts)
    ok = sum(a.ok for a in attempts)
    decisions = Counter(
        str((a.parsed or {}).get("action", (a.parsed or {}).get("sufficient")))
        for a in attempts
        if a.parsed is not None
    )
    return {
        "decisions": dict(decisions),
        "model": model_id,
        "schema": schema_name,
        "method": method,
        "reasoning": reasoning,
        "source": case.source,
        "num_predict": num_predict,
        "samples": samples,
        "ok": ok,
        "rate": ok / samples if samples else 0.0,
        "s_per_call": sum(a.seconds for a in attempts) / samples if samples else 0.0,
        "errors": dict(errors),
        "done_reasons": dict(reasons),
        "prompt_tokens": {
            "min": min(known) if known else None,
            "median": sorted(known)[len(known) // 2] if known else None,
            "max": max(known) if known else None,
        },
        "buckets": buckets,
        "thinking_chars": {
            "ok_mean": sum(ok_thinking) / len(ok_thinking) if ok_thinking else None,
            "failed_mean": (
                sum(failed_thinking) / len(failed_thinking) if failed_thinking else None
            ),
        },
        "attempts": [a.__dict__ for a in attempts],
    }


def _bucket_order(label: str) -> int:
    if label == "unknown":
        return 10**9
    return int(label.lstrip("<>="))


def _line(row: dict) -> str:
    top = ", ".join(f"{k}={v}" for k, v in Counter(row["errors"]).most_common(3)) or "-"
    reasons = ", ".join(f"{k}={v}" for k, v in row["done_reasons"].items())
    tokens = row["prompt_tokens"]
    return (
        f"{row['model']:16s} {row['schema']:17s} {row['method']:17s} "
        f"{str(row['reasoning']):6s} {row['ok']:>3d}/{row['samples']} "
        f"({100 * row['rate']:5.1f}%) {row['s_per_call']:5.1f}s/call  "
        f"prompt_tok={tokens['min']}/{tokens['median']}/{tokens['max']}  "
        f"done=[{reasons}]  decisions={row.get('decisions', {})}  {top}"
    )


def table(rows: list[dict]) -> str:
    header = (
        "| model | schema | method | reasoning | source | num_predict | "
        "first-attempt | s/call | prompt tokens (min/med/max) | stop reasons | "
        "top failure |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        errors = Counter(row["errors"])
        top = errors.most_common(1)[0][0] if errors else "—"
        tokens = row["prompt_tokens"]
        reasons = ", ".join(f"{k}={v}" for k, v in row["done_reasons"].items())
        lines.append(
            f"| `{row['model']}` | `{row['schema']}` | {row['method']} | "
            f"{row['reasoning']} | {row['source']} | {row['num_predict']} | "
            f"**{100 * row['rate']:.0f}%** ({row['ok']}/{row['samples']}) | "
            f"{row['s_per_call']:.1f} | {tokens['min']}/{tokens['median']}/"
            f"{tokens['max']} | {reasons} | {top} |"
        )
    return "\n".join(lines)


def bucket_table(rows: list[dict]) -> str:
    """Success by prompt length, one line per cell; the length hypothesis test."""
    lines = [
        "| model | schema | method | bucket (prompt tokens) | ok / n |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        for bucket, counts in row["buckets"].items():
            lines.append(
                f"| `{row['model']}` | `{row['schema']}` | {row['method']} | "
                f"{bucket} | {counts['ok']} / {counts['n']} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=CONFIG["agentic_judge_model"])
    parser.add_argument("--schemas", default=",".join(SCHEMAS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--reasoning", default="low")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Repeats of the synthetic prompt. Ignored with --from-eval.",
    )
    parser.add_argument(
        "--from-eval",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Build prompts from the first N approved questions in PAGES_DB_PATH "
            "(benchmark order) with real first-stage retrieval, one call each."
        ),
    )
    parser.add_argument(
        "--first-stage",
        default="qwen_hybrid_rerank",
        help="First-stage retriever behind ChunkSufficiency and ToolAction prompts.",
    )
    parser.add_argument("--variant", default="base")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Override the output cap for every schema (default: config per schema).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    schema_names = [s.strip() for s in args.schemas.split(",") if s.strip()]
    if args.from_eval:
        cases = _eval_cases(
            args.from_eval, schema_names, args.first_stage, args.variant
        )
    else:
        cases = _synthetic_cases(args.samples)

    rows: list[dict] = []
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows.extend(
            probe(
                model_id,
                cases,
                schema_names,
                [m.strip() for m in args.methods.split(",") if m.strip()],
                [r.strip() for r in args.reasoning.split(",") if r.strip()],
                args.num_predict,
            )
        )
    print()
    print(table(rows))
    print()
    print(bucket_table(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            table(rows)
            + "\n\n"
            + bucket_table(rows)
            + "\n\n```json\n"
            + json.dumps(rows, indent=1, default=str)
            + "\n```\n",
            encoding="utf-8",
        )
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
