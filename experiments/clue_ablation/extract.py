"""Split hard questions into clues: one attribute-value constraint each, quoted literally from the question.

    uv run python -m experiments.clue_ablation.extract --n 30            # pilot
    uv run python -m experiments.clue_ablation.extract --n 1000

The sample is dev, answer_ok, hard questions on `balanced` pages (<= 400 pages per language, so Swedish lake stubs do
not dominate), in a fixed hash order; `--n 30` is the first 30 of the same order as `--n 1000`. Each clue's quote is
checked against the question by exact string match (`exact`), so a paraphrased clue is visible, not trusted.
Writes out/clues-<model>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import duckdb
import httpx
from pydantic import BaseModel, ValidationError

from src.db.dataset import ROOT
from src.shared.env import load_local_env
from src.shared.llm import LocalOllamaStructuredLlm, run_structured_outputs
from src.shared.prompts import render

OUT = Path(__file__).parent / "out"
SEED = 7
Attribute = Literal[
    "name", "location", "type", "feature", "quantity", "date", "event", "other"
]


class Clue(BaseModel):
    quote: str
    attribute: Attribute


class Clues(BaseModel):
    clues: list[Clue]


def sample(n: int):
    """The first n dev hard questions on balanced pages, in a fixed hash order."""
    rag = ROOT / "qa" / "wiki_qa_rag.parquet"
    return duckdb.sql(
        f"""select 'rag|' || id || '|' || n as "key", id, lang, question from '{rag}'
        where answer_ok and split = 'dev' and kind = 'challenge' and balanced
        order by hash(id || n || {SEED}) limit {int(n)}"""  # nosec B608 - built from constants and an int
    ).df()


LING = (
    "inclusionai/ling-3.1-flash-free"  # free on the Vercel AI Gateway, as in datagen/
)
GATEWAY = "https://ai-gateway.vercel.sh/v1/chat/completions"
# Clues.model_json_schema() puts Clue under $defs; the provider behind the gateway rejects $ref with a 400
TOOL = {
    "type": "function",
    "function": {
        "name": "clues",
        "parameters": {
            "type": "object",
            "properties": {
                "clues": {"type": "array", "items": Clue.model_json_schema()}
            },
            "required": ["clues"],
        },
    },
}


def ask_ling(client: httpx.Client, prompt: str) -> Clues | None:
    """One request; structured output through a forced tool call (Ling has no response_format). None on failure."""
    body = {
        "model": LING,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "clues"}},
    }
    try:
        reply = client.post(GATEWAY, json=body).raise_for_status().json()
        call = reply["choices"][0]["message"]["tool_calls"][0]
        return Clues.model_validate_json(call["function"]["arguments"])
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValidationError):
        return None


def run_ling(
    keys: list[str], prompts: list[str], workers: int, passes: int
) -> list[Clues]:
    """All prompts through Ling. The free gateway answers 503 in bursts, so every answer is cached as it comes
    in (out/ling-cache.jsonl, by question key) and the rest is retried in passes with a growing pause; a rerun
    only asks what is still missing."""
    load_local_env()
    key = os.environ["VERCEL_API_KEY"]
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)
    OUT.mkdir(exist_ok=True)
    cache_path = OUT / "ling-cache.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {
            r["key"]: Clues.model_validate(r["clues"])
            for r in map(json.loads, cache_path.open())
        }
    with ThreadPoolExecutor(workers) as pool, cache_path.open("a") as cache_file:
        for attempt in range(passes):
            todo = [i for i, k in enumerate(keys) if k not in cache]
            if not todo:
                break
            if attempt:
                time.sleep(min(10 * 2 ** (attempt - 1), 120))
            print(f"pass {attempt + 1}: {len(todo)} to ask", flush=True)
            for i, answer in zip(
                todo,
                pool.map(lambda i: ask_ling(client, prompts[i]), todo),
                strict=True,
            ):
                if answer is not None:
                    cache[keys[i]] = answer
                    cache_file.write(
                        json.dumps(
                            {"key": keys[i], "clues": answer.model_dump()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    cache_file.flush()
    if missing := [k for k in keys if k not in cache]:
        raise RuntimeError(
            f"no answer yet for {len(missing)} questions after {passes} passes; rerun to continue"
        )
    return [cache[k] for k in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--backend", choices=["ollama", "ling"], default="ollama")
    ap.add_argument(
        "--model", default="gpt-oss:20b", help="Ollama model (--backend ollama)"
    )
    ap.add_argument(
        "--reasoning", default="low", help="gpt-oss effort: low, medium or high"
    )
    # Ollama otherwise loads gpt-oss with its full 131k context: slow and memory-hungry on the shared GPU
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--passes", type=int, default=12, help="Ling: retry passes before giving up"
    )
    args = ap.parse_args()

    df = sample(args.n)
    prompts = [
        render("clue_extract", lang=r.lang, question=r.question)
        for r in df.itertuples()
    ]
    start = time.time()
    if args.backend == "ling":
        model, results = (
            "ling-3.1-flash",
            run_ling(df.key.tolist(), prompts, args.workers, args.passes),
        )
    else:
        llm = LocalOllamaStructuredLlm(
            args.model, reasoning=args.reasoning, num_ctx=args.num_ctx
        )
        model, results = (
            args.model,
            run_structured_outputs(llm, prompts, Clues, workers=args.workers),
        )
    print(f"{len(results)} questions in {time.time() - start:.0f} s")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"clues-{model.replace(':', '_').replace('/', '_')}.jsonl"
    with path.open("w") as fh:
        for r, res in zip(df.itertuples(), results, strict=True):
            clues = [
                c.model_dump() | {"exact": c.quote in r.question} for c in res.clues
            ]
            fh.write(
                json.dumps(
                    {
                        "key": r.key,
                        "lang": r.lang,
                        "question": r.question,
                        "clues": clues,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    rows = [json.loads(line) for line in path.open()]
    quotes = [c for row in rows for c in row["clues"]]
    print(
        f"wrote {path}: {len(quotes)} clues, {sum(c['exact'] for c in quotes) / max(len(quotes), 1):.0%} quoted exactly"
    )


if __name__ == "__main__":
    main()
