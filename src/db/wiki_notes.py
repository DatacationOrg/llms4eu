"""Ling 3.1 Flash notes on the wiki pages, in the page's language, kept in `notes.db`
(`wiki_chunks` copies them into the Parquet files):

- `summaries`: each page in at most 3 sentences / 300 characters;
- `roles`: per chunk, one sentence on what kind of information it adds to its page,
  given the page summary and the chunk.

Resumable: done ids are skipped, failures are retried on the next run.

    uv run python -m src.db.wiki_notes summaries --workers 100
    uv run python -m src.db.wiki_notes roles --size 2048 --workers 100
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from src.db.dataset import Chunk, Page, load
from src.db.schemas.chunk import represent
from src.db.wiki_chunks import NOTES
from src.shared.env import load_local_env

MODEL = "inclusionai/ling-3.1-flash-free"  # free on the Vercel AI Gateway
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
MAX_CHARS = 300
ARTICLE_CHARS = 24_000  # ponytail: long pages are summarised from their first 24k chars

SUMMARY = """Summarise this Wikipedia article about a place in at most 3 short sentences \
(under 300 characters in total). Write in the article's language ({lang}). Answer with \
the summary only.

{text}"""

ROLE = """Below is a summary of a Wikipedia article about a place, then one chunk of that \
article. In one short sentence (under 200 characters), say what kind of information this \
chunk contributes to the article (for example its history, how to visit, a list of \
sources). Do not summarise the whole article. Write in the article's language ({lang}). \
Answer with the sentence only.

Article summary: {summary}

Chunk:
{chunk}"""

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def cap(text: str, sentences: int, chars: int = MAX_CHARS) -> str:
    """At most `sentences` sentences and `chars` characters, cut at a sentence end if any."""
    parts = _SENTENCE_END.split(" ".join(text.split()).strip("\"'“”„ "))[:sentences]
    while len(parts) > 1 and len(" ".join(parts)) > chars:
        parts.pop()
    out = " ".join(parts)
    return out if len(out) <= chars else out[: out.rfind(" ", 0, chars - 1)] + "…"


def ask(client: httpx.Client, prompt: str) -> str | None:
    try:
        r = client.post(
            URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        )
        if r.status_code in (429, 503):  # busy gateway: wait, retried next run
            time.sleep(30)
            return None
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or None
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print("error", type(e).__name__, str(e)[:120], flush=True)
        time.sleep(10)
        return None


def run(table: str, items: list[tuple[str, str]], sentences: int, workers: int) -> None:
    """Ask Ling for every (id, prompt) not yet in `table`, saving as answers come in."""
    db = sqlite3.connect(NOTES)
    db.execute(
        f"create table if not exists {table} (id text primary key, text text, raw text, "
        "model text, created_at text default current_timestamp)"
    )
    client = httpx.Client(
        headers={"Authorization": f"Bearer {os.environ['VERCEL_API_KEY']}"}, timeout=120
    )
    with ThreadPoolExecutor(workers) as pool:
        while True:  # passes until every item has an answer: the gateway drops many
            done = {i for (i,) in db.execute(f"select id from {table}")}
            todo = [(i, p) for i, p in items if i not in done]
            print(f"{table}: {len(done)} done, {len(todo)} to do", flush=True)
            if not todo:
                return
            for start in range(0, len(todo), 2000):  # bounded queue, commit per window
                window = todo[start : start + 2000]
                answers = pool.map(lambda item: ask(client, item[1]), window)
                rows = [
                    (i, cap(a, sentences), a, MODEL)
                    for (i, _), a in zip(window, answers)
                    if a
                ]
                db.executemany(
                    f"insert or ignore into {table} (id, text, raw, model) values (?, ?, ?, ?)",
                    rows,
                )
                db.commit()
                print(
                    f"{table}: +{len(rows)}, {start + len(window)}/{len(todo)} asked",
                    flush=True,
                )


def summaries(workers: int) -> None:
    pages = load(Page, ["id", "in_language", "text"]).to_pylist()
    items = [
        (p["id"], SUMMARY.format(lang=p["in_language"], text=p["text"][:ARTICLE_CHARS]))
        for p in pages
    ]
    run("summaries", items, 3, workers)


def roles(size: int, workers: int) -> None:
    pages = load(Page, ["id", "in_language"]).to_pylist()
    lang = {p["id"]: p["in_language"] for p in pages}
    with sqlite3.connect(NOTES) as db:
        summary = dict(db.execute("select id, text from summaries"))
    chunks = load(
        Chunk, ["id", "page_id", "title", "breadcrumb", "text"], size=size
    ).to_pylist()
    items = [
        (
            c["id"],
            ROLE.format(
                lang=lang[c["page_id"]],
                summary=summary[c["page_id"]],
                chunk=represent(c["title"], c["breadcrumb"], c["text"]),
            ),
        )
        for c in chunks
        if c["page_id"] in summary
    ]
    run("roles", items, 1, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["summaries", "roles"])
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args()
    load_local_env()
    if args.job == "summaries":
        summaries(args.workers)
    else:
        roles(args.size, args.workers)


if __name__ == "__main__":
    main()
