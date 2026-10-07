"""Not used by the pipeline: how the `summary` and `role` columns were made.

Ling 3.1 Flash notes on the wiki pages, in the page's language, kept in `notes.db`
until `export` writes them into `wikipages.parquet` and `chunks.parquet`:

- `summaries`: each page in at most 3 sentences / 300 characters;
- `roles`: per chunk, one sentence on what kind of information it adds to its page,
  given the page summary and the chunk.

Resumable: done ids are skipped, failures are retried on the next run.

    uv run python -m datagen.notes summaries
    uv run python -m datagen.notes roles   # every size, largest first
    uv run python -m datagen.notes export
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pyarrow as pa

from src.db.dataset import ROOT, SIZES, Chunk, Page, load, write
from src.db.schemas.chunk import represent
from src.shared.env import load_local_env

NOTES = ROOT / "notes.db"

MODEL = "inclusionai/ling-3.1-flash-free"  # free on the Vercel AI Gateway
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
MAX_SENTENCES = {"summaries": 3, "roles": 1}
MAX_CHARS = 300
WINDOW = 2000  # answers saved per commit
ARTICLE_CHARS = 24_000  # ponytail: long pages are summarised from their first 24k chars

SUMMARY = """Summarise this Wikipedia article about a place in at most {sentences} short \
sentences (under {chars} characters in total). Write in the article's language ({lang}). Answer with \
the summary only.

{text}"""

ROLE = """Below is a summary of a Wikipedia article about a place, then one chunk of that \
article. In one short sentence (under {chars} characters), say what kind of information this \
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


def run(table: str, items: list[tuple[str, str]], workers: int) -> None:
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
            done = {i for (i,) in db.execute(f"select id from {table}")}  # nosec B608 - fixed table name
            todo = [(i, p) for i, p in items if i not in done]
            print(f"{table}: {len(done)} done, {len(todo)} to do", flush=True)
            if not todo:
                return
            for start in range(0, len(todo), WINDOW):
                window = todo[start : start + WINDOW]
                answers = pool.map(lambda item: ask(client, item[1]), window)
                rows = [
                    (i, cap(a, MAX_SENTENCES[table]), a, MODEL)
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
        (
            p["id"],
            SUMMARY.format(
                sentences=MAX_SENTENCES["summaries"],
                chars=MAX_CHARS,
                lang=p["in_language"],
                text=p["text"][:ARTICLE_CHARS],
            ),
        )
        for p in pages
    ]
    run("summaries", items, workers)


def roles(size: int, workers: int) -> None:
    pages = load(Page, ["id", "in_language"]).to_pylist()
    lang = {p["id"]: p["in_language"] for p in pages}
    with sqlite3.connect(NOTES) as db:
        summary = dict(db.execute("select id, text from summaries"))
    columns = ["id", "page_id", "title", "breadcrumb", "text", "role"]
    chunks = load(Chunk, columns, size=size).to_pylist()
    items = [
        (
            c["id"],
            ROLE.format(
                chars=MAX_CHARS,
                lang=lang[c["page_id"]],
                summary=summary[c["page_id"]],
                chunk=represent(c["title"], c["breadcrumb"], c["text"]),
            ),
        )
        for c in chunks
        if c["page_id"] in summary and c["role"] is None  # kept over a rechunk
    ]
    run("roles", items, workers)


def export() -> None:
    """Fill `summary` / `role` in the Parquet files; notes for gone ids are skipped."""
    with sqlite3.connect(NOTES) as db:
        notes = {
            t: dict(db.execute(f"select id, text from {t}"))  # nosec B608 - fixed table names
            for t in ("summaries", "roles")
        }
    for model, table, column in (
        (Page, "summaries", "summary"),
        (Chunk, "roles", "role"),
    ):
        data = load(model)
        ids, current = data.column("id").to_pylist(), data.column(column).to_pylist()
        filled = [notes[table].get(i, old) for i, old in zip(ids, current)]
        index = data.schema.get_field_index(column)
        write(model, data.set_column(index, column, pa.array(filled, pa.string())))
        print(f"{column}: {sum(v is not None for v in filled)} of {len(ids)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["summaries", "roles", "export"])
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        choices=SIZES,
        default=sorted(SIZES, reverse=True),
    )
    parser.add_argument("--workers", type=int, default=60)
    args = parser.parse_args()
    load_local_env()
    if args.job == "summaries":
        summaries(args.workers)
    elif args.job == "roles":
        for size in args.sizes:
            roles(size, args.workers)
    else:
        export()


if __name__ == "__main__":
    main()
