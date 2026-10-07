"""Temporary: shared free-model client for the cleaning scripts: structured output, <= RPM requests started per
minute across threads, and a STOP event on the first 429 (rate limit or daily quota). The model is chosen per
process with CLEAN_MODEL=bunny|ling (default bunny); every writer stores MODEL_ID with its rows. Needs
OPENROUTER_API_KEY (bunny) or VERCEL_API_KEY (ling). Also loads pages (article rows) and their generated items."""

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PAGES = "/data/llms4eu/wiki/pages.jsonl"
DB = "wiki_qa.db"
RPM = 50
STOP = threading.Event()
_lock, _next = threading.Lock(), [0.0]

# name -> (model id, base url, key env var, structured-output method)
MODELS = {
    "bunny": (
        "stealth/space-bunny-alpha",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "json_schema",
    ),
    # Vercel AI Gateway (no rate limits per the user; OpenRouter has limits and Bunny needs them). Needs a credit
    # card on the Vercel account even for the free model (403 otherwise). No response_format: tool calling
    "ling": (
        "inclusionai/ling-3.1-flash-free",
        "https://ai-gateway.vercel.sh/v1",
        "VERCEL_API_KEY",
        "function_calling",
    ),
}
MODEL_ID, _BASE, _KEY, _METHOD = MODELS[os.environ.get("CLEAN_MODEL", "bunny")]
if MODEL_ID.startswith("inclusionai/"):
    RPM = 300  # Vercel, account with credits: 99/100 concurrent requests went through (5 rpm before credits)


def llm(
    schema, temperature=0.3, max_tokens=32768, effort=None
):  # max_tokens caps runaway loops (65k); reasoning alone can exceed 8k. effort: OpenRouter reasoning effort
    from langchain_openai import (
        ChatOpenAI,
    )  # imported here: page loading works without it

    return ChatOpenAI(
        model=MODEL_ID,
        base_url=_BASE,
        api_key=os.environ[_KEY],
        temperature=temperature,
        max_retries=0,
        timeout=300,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": effort}} if effort else None,
    ).with_structured_output(schema, method=_METHOD)


def call(chain, messages):
    """One paced request; None on failure (a 429 sets STOP, except for Ling: its per-minute limit is waited out)."""
    if STOP.is_set():
        return None
    with _lock:
        time.sleep(max(0.0, _next[0] - time.monotonic()))
        _next[0] = max(_next[0], time.monotonic()) + 60 / RPM
    try:
        return chain.invoke(messages)
    except Exception as e:
        print("error", type(e).__name__, str(e)[:160], flush=True)
        if MODEL_ID.startswith("inclusionai/") and "Error code: 503" in str(e):
            time.sleep(
                60
            )  # Vercel overloaded (many runners at once): back off longer than the 20 s below
            return None
        if "code: 429" in str(e) or "Error code: 429" in str(e):
            if MODEL_ID.startswith(
                "inclusionai/"
            ):  # per-minute limit: wait it out instead of stopping the run
                time.sleep(60)
                return None
            STOP.set()
        time.sleep(
            20
        )  # mostly transient "provider returned an empty response" (502): don't hammer it
        return None


def pages(ids=None):
    """Stream (page, items) for pages with generated items, optionally only `ids`."""
    con = sqlite3.connect(DB, timeout=120)
    with open(PAGES) as fh:
        for line in fh:
            p = json.loads(line)
            if ids is not None and p["id"] not in ids:
                continue
            row = con.execute(
                "select items, model from questions where id = ? and error is null",
                (p["id"],),
            ).fetchone()
            if row:
                p["gen_model"] = row[1]
                yield p, json.loads(row[0])


def run_all(fn, items, workers):
    """Yield fn(item) results as they complete (one slow page must not hold back the saving of the rest)."""
    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(fn, t) for t in items]):
            yield f.result()
