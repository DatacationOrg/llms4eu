"""Temporary: generate up to 5 search questions per wiki article with stealth/space-bunny-alpha.
Items: question (article language), question_en, facts, query (English keywords).
Queue: short articles first; Swedish lakes, then machine-generated pages last. Resumable via the DB.
Needs OPENROUTER_API_KEY. Usage: gen_questions.py [--limit N] [--workers 10] [--rpm 50] [--max-chars 100000] [--db questions.db]
Stops at the first 429 (rate limit or daily free quota) instead of retrying into it."""

import argparse
import json
import os
import itertools
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

PAGES = "/data/llms4eu/wiki/pages.jsonl"
MODEL = os.environ.get("QG_MODEL", "stealth/space-bunny-alpha")
BASE_URL = os.environ.get("QG_BASE_URL", "https://openrouter.ai/api/v1")

PROMPT = """You write test data for a search system. The article below is one of ~100k Wikipedia
pages about places in Europe. Write up to 5 questions that a chat user might ask
without having seen this article. The user is looking for this information among all
the pages.

Only ask what a real person would want to know: what the place is like, what there is
to see or do, its history, nature, size or height, how to get there. Never ask about
codes, IDs, register or basin numbers, coordinates or other database fields. Match the
number of questions to the article: a short stub gets 1-2, a rich article up to 5.
Fewer good questions are better than five weak ones. Return an empty list if the
article has no facts worth asking about.

For each item:
- question: natural, self-contained and grammatical, in the article's language, one
  information need per question. Name the place or describe it (region, type, a
  distinguishing detail) so that only this place fits among all the pages. Never refer
  to "the article" or "the text" and never put the answer in the question. Paraphrase
  instead of copying the article's wording, so that keyword search alone would struggle.
- question_en: the question translated to English.
- facts: 1-3 short facts from the article that answer the question, not restate it.
- query: English search keywords an embedder would use to find this passage, using
  only information in the question, never the answer."""


class QA(BaseModel):
    question: str
    question_en: str
    facts: list[str] = Field(min_length=1, max_length=3)
    query: str


class Questions(BaseModel):
    items: list[QA] = Field(max_length=5)

    @model_validator(mode="before")
    @classmethod
    def wrap_bare_list(cls, v):  # the model often drops the {"items": ...} wrapper
        return {"items": v} if isinstance(v, list) else v


def queue(done, largest=False):
    """Byte offsets of pending rows, in priority order (one metadata pass, texts read later).
    largest: longest pages first, skipping SE lakes and machine-made pages (spend the API teacher there)."""
    keys = []
    with open(PAGES, "rb") as f:
        while line := f.readline():
            r = json.loads(line)
            if r["id"] in done:
                continue
            se_lake = r["country"] == "SE" and "lake" in r["categories"]
            if largest and (se_lake or r["machine_generated"]):
                continue
            keys.append(
                (
                    (-r["char_count"],)
                    if largest
                    else (r["machine_generated"], se_lake, r["char_count"]),
                    f.tell() - len(line),
                )
            )
    return [off for _, off in sorted(keys)]


def read_row(off):
    with open(PAGES, "rb") as f:
        f.seek(off)
        return json.loads(f.readline())


def user_msg(
    r, max_chars=6000
):  # corpus rows or pool rows; long articles are cut like the teacher pools
    lang = r.get("in_language") or r.get("lang")
    return f"Title: {r['title']}\nCountry: {r.get('country')}\nLanguage: {lang}\n\n{r['text'][:max_chars]}"


STOP = threading.Event()  # set on a 429: no more requests
_pace = threading.Lock()
_next = [0.0]


def paced(rpm):  # start requests at most rpm per minute, across all workers
    with _pace:
        wait = _next[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next[0] = max(_next[0], time.monotonic()) + 60 / rpm


def ask(llm, r, max_chars=6000, rpm=50):
    msg = user_msg(r, max_chars)
    err = "skipped: stopped on 429"
    for attempt in range(3):  # ponytail: every retry costs a free-tier request
        if STOP.is_set():
            break
        paced(rpm)
        try:
            t = time.time()
            res = llm.invoke([("system", PROMPT), ("user", msg)])
            print(f"{r['id']} {time.time() - t:.1f}s", flush=True)
            return r["id"], [qa.model_dump() for qa in res.items], None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print("retry", r["id"], err, flush=True)
            if "429" in err:
                STOP.set()
                print("429: stopping, no new requests", flush=True)
                break
            time.sleep(5 * (attempt + 1))
    return r["id"], None, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument(
        "--largest", action="store_true", help="longest non-SE-lake pages first"
    )
    p.add_argument("--redo", action="store_true", help="also redo pages already done")
    p.add_argument(
        "--rpm", type=int, default=50, help="max requests started per minute"
    )
    p.add_argument(
        "--max-chars", type=int, default=6000, help="article cut (100000 ~ 32k tokens)"
    )
    p.add_argument(
        "--db", default=os.path.join(os.path.dirname(__file__), "questions.db")
    )
    p.add_argument(
        "--pool",
        help="pool JSON (e.g. eval20.json): only these rows, keyed by url, no corpus scan",
    )
    p.add_argument(
        "--ids", help="file of page ids: only these corpus pages, file order ignored"
    )
    a = p.parse_args()

    db = sqlite3.connect(a.db)
    db.execute(
        "create table if not exists questions (id text primary key, model text, items text,"
        " error text, created_at text default current_timestamp)"
    )
    done = (
        set()
        if a.redo
        else {i for (i,) in db.execute("select id from questions where error is null")}
    )
    if a.pool:  # rows carry title/lang/text(/country); ids are urls
        rows = [
            {**r, "id": r["url"]}
            for r in json.load(open(a.pool))
            if r["url"] not in done
        ]
    elif a.ids:
        ids = {i.strip() for i in open(a.ids)} - done
        rows = (r for r in map(json.loads, open(PAGES)) if r["id"] in ids)
    else:
        rows = map(read_row, queue(done, a.largest))
    todo = itertools.islice(rows, a.limit)  # lazy: texts are read as workers need them

    llm = ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ.get("OPENROUTER_API_KEY", "local"),
        temperature=0,
        max_retries=0,
        timeout=240,
    )
    llm = llm.with_structured_output(Questions, method="json_schema")

    with ThreadPoolExecutor(a.workers) as ex:
        results = ex.map(lambda r: ask(llm, r, a.max_chars, a.rpm), todo)
        for n, (pid, items, err) in enumerate(results, 1):
            if STOP.is_set() and items is None:
                continue  # never overwrite with a quota error
            # an error never replaces an existing good row (matters with --redo)
            verb = "insert or ignore" if items is None else "insert or replace"
            db.execute(
                f"{verb} into questions (id, model, items, error) values (?, ?, ?, ?)",
                (
                    pid,
                    MODEL,
                    None if items is None else json.dumps(items, ensure_ascii=False),
                    err,
                ),
            )
            db.commit()
            if n % 50 == 0:
                print(n, "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
