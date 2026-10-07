"""Temporary: free rule checks on every generated item in wiki_qa.db -> table `checks` (id, items = per-item flag
lists, same order as questions.items). Streams pages.jsonl, so RAM stays flat. Flags:
  dup            question (near-)repeats an earlier one on the page (token Jaccard >= 0.8)
  q_lang         question not detected as the article's language (skipped for languages lingua lacks)
  en_lang        question_en not detected as English
  query_leak     query contains numbers from the facts (the answer) that are not in the question
  fact_unmatched a fact with a number not in the article, or most of its word stems not in it
  no_anchor      question shares no word with the article title (likely not unique to this place)
  padded         more items than the article can carry (< 400 chars of text per item)
Usage: rule_checks.py [--limit N]   |   rule_checks.py test"""

import json
import re
import sqlite3
import sys

from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

PAGES = "/data/llms4eu/wiki/pages.jsonl"
DB = "wiki_qa.db"
det = LanguageDetectorBuilder.from_all_languages().build()


def is_lang(
    s, code
):  # argmax is noisy on short keyword strings; only flag when the language is near-impossible
    return (
        det.compute_language_confidence(
            s, Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper()))
        )
        >= 0.05
    )


def known(code):
    return hasattr(IsoCode639_1, code.upper())


def words(s):
    return re.findall(r"\w+", s.lower())


def stems(
    s,
):  # crude prefix stems so inflected forms (Hungarian, Finnish, Slavic cases) still match
    return {w[:4] for w in words(s) if len(w) >= 4 and not w.isdigit()}


def jaccard(a, b):
    a, b = set(words(a)), set(words(b))
    return len(a & b) / max(len(a | b), 1)


def nums(s):  # "1 416" and "1416" are the same number
    return set(re.findall(r"\d+", re.sub(r"(?<=\d)[\s\u202f.,](?=\d{3}\b)", "", s)))


def leak(
    it,
):  # numbers from the facts (the answer) in the query that the question doesn't give
    return bool(
        nums(it["query"]) & set().union(*map(nums, it["facts"])) - nums(it["question"])
    )


def fact_ok(fact, text_stems, text_nums):
    n = nums(fact)
    st = stems(fact)
    if n - text_nums:
        return False
    return not st or len(st & text_stems) / len(st) >= 0.4


def flags(page, items):
    text = page["text"]
    text_stems, text_nums = stems(text), nums(text)
    title = stems(re.sub(r"\(.*?\)", "", page["title"])) or set(words(page["title"]))
    lg = page["in_language"]
    padded = len(items) > 1 and page["char_count"] / len(items) < 400
    names = stems(page["title"]) | set(words(page["title"]))

    def plain(
        s,
    ):  # drop names (capitalized or title words): they don't tell a sentence's language
        return " ".join(
            w
            for w in s.split()
            if not w[:1].isupper() and w.lower().strip("?,.")[:4] not in names
        )

    out = []
    for j, it in enumerate(items):
        f = []
        if any(jaccard(it["question"], items[k]["question"]) >= 0.8 for k in range(j)):
            f.append("dup")
        if known(lg) and not is_lang(it["question"], lg):
            f.append("q_lang")
        en = plain(it["question_en"])
        if lg != "en" and en and not is_lang(en, "en"):
            f.append("en_lang")
        if leak(it):
            f.append("query_leak")
        if not all(fact_ok(x, text_stems, text_nums) for x in it["facts"]):
            f.append("fact_unmatched")
        if not (title & (stems(it["question"]) | set(words(it["question"])))):
            f.append("no_anchor")
        if padded:
            f.append("padded")
        out.append(f)
    return out


def main(limit=None):
    con = sqlite3.connect(DB, timeout=120)
    con.execute(
        "create table if not exists checks (id text primary key, items text, created_at text default current_timestamp)"
    )
    done = {r[0] for r in con.execute("select id from checks")}
    n = 0
    with open(PAGES) as fh:
        for line in fh:
            page = json.loads(line)
            if page["id"] in done:
                continue
            row = con.execute(
                "select items from questions where id = ? and error is null",
                (page["id"],),
            ).fetchone()
            if not row:
                continue
            con.execute(
                "insert into checks (id, items, model) values (?, ?, ?)",
                (page["id"], json.dumps(flags(page, json.loads(row[0]))), "rules"),
            )
            n += 1
            if n % 2000 == 0:
                con.commit()
                print(n, flush=True)
            if limit and n >= limit:
                break
    con.commit()
    print("finished", n, flush=True)


def test():
    page = {
        "id": "x",
        "title": "Burg Clam",
        "in_language": "de",
        "char_count": 3000,
        "text": "Die Burg Clam liegt in Oberösterreich. Sie wurde um 1100 errichtet.",
    }
    good = {
        "question": "Wann wurde die Burg Clam in Oberösterreich errichtet?",
        "question_en": "When was Clam Castle in Upper Austria built?",
        "facts": ["Die Burg wurde um 1100 errichtet."],
        "query": "Clam Castle Upper Austria construction date",
    }
    bad = {
        "question": "Wann wurde die Burg Clam in Oberösterreich errichtet?",
        "question_en": "Quando è stato costruito il castello?",
        "facts": ["Die Burg wurde 1350 von Mönchen gebaut."],
        "query": "Clam Burg Errichtung 1350",
    }
    assert flags(page, [good]) == [[]], flags(page, [good])
    got = flags(page, [good, bad])[1]
    assert {"dup", "en_lang", "query_leak", "fact_unmatched"} <= set(got), got
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test()
    else:
        main(int(sys.argv[2]) if sys.argv[1:2] == ["--limit"] else None)
