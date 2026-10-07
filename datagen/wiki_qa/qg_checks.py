"""Temporary: cheap automatic checks on questions tables (ids = urls) against a pool file.
Usage: qg_checks.py <pool.json> <db>...   ('teacher' in place of a db scores qg_clean.jsonl)"""

import json
import os
import re
import sqlite3
import sys

from lingua import LanguageDetectorBuilder

S = os.path.dirname(os.path.abspath(__file__))
det = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)


def lang(s):
    got = det.detect_language_of(s)
    return got.iso_code_639_1.name.lower() if got else None


def leak(it):
    # answer tokens (numbers / words >=5 chars in facts but not in the question) that show up in the query
    q = set(re.findall(r"\w+", it["question"].lower()))
    ans = {w for f in it["facts"] for w in re.findall(r"\d+|\w{5,}", f.lower())} - q
    return bool(ans & set(re.findall(r"\w+", it["query"].lower())))


def score(rows, got):
    n = ok = items = q_lang = q_en = leaks = 0
    for r in rows:
        n += 1
        its = got.get(r["url"])
        if its is None:
            continue
        ok += 1
        for it in its:
            items += 1
            q_lang += lang(it["question"]) == r["lang"]
            # drop capitalized tokens (place names) so they don't decide the query's language
            q_en += (
                lang(" ".join(w for w in it["query"].split() if not w[:1].isupper()))
                == "en"
            )
            leaks += leak(it)
    d = max(items, 1)
    return {
        "valid_json": f"{ok}/{n}",
        "items/article": round(items / max(ok, 1), 1),
        "question_in_article_lang": f"{q_lang / d:.0%}",
        "query_english": f"{q_en / d:.0%}",
        "query_leaks_answer": f"{leaks / d:.0%}",
    }


if __name__ == "__main__":
    rows = json.load(open(sys.argv[1]))
    for db in sys.argv[2:]:
        if db == "teacher":
            got = {
                r["url"]: r["items"]
                for r in map(json.loads, open(f"{S}/qg_clean.jsonl"))
            }
        else:
            got = {
                u: json.loads(i)
                for u, i in sqlite3.connect(db).execute(
                    "select id, items from questions where error is null"
                )
            }
        print(f"{os.path.basename(db):24}", score(rows, got))
