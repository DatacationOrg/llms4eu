"""Temporary: extra test-set layers on top of the RAG items (gen_rag.py), generated and checked by Ling 3.1 Flash (a
second Ling call checks each generation, so no output is kept on the generator's word). What each layer tests and how
to use it: wiki_qa_dataset/README.md. Steps (each resumes from its table; one row per page or spec, model id stored):
  variants  balanced pages: each ok item as a keyword query, a typo'd question, a verbose chatty question and a
            conversational follow-up (history turn + pronoun question), plus tags time_sensitive and reasoning
  unans     balanced pages: 2 unanswerable questions (false_premise, not_covered)
  compare   balanced pages: one question comparing the page with a similar one (same language, country, category)
  meta      list (category x country) and geo (category within r km of an anchor place) questions; gold sets come
            from the metadata, Ling only phrases them
  qrels     per question (ok challenge items, corpus items of balanced pages, checked unanswerable items): BM25 top 20
            + the place's other-language pages + the gold page, each judged from an excerpt (match yes|partly|no,
            answers) -> multi-page relevance and hard negatives
Usage: CLEAN_MODEL=ling gen_extra.py STEP [--workers 100] [--limit N]"""

import argparse
import collections
import json
import math
import re
import sqlite3
import threading
import zlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

import bm25
import bunny
from gen_rag import spans
from qspec import RAG


def crc(s):
    return zlib.crc32(s.encode())


class Listed(BaseModel):
    items: list

    @model_validator(mode="before")
    @classmethod
    def wrap_bare_list(cls, v):  # free models often return the bare list
        return {"items": v} if isinstance(v, list) else v


class Variant(BaseModel):
    j: int
    keyword: str
    typo: str
    verbose: str
    history: str
    followup: str
    time_sensitive: bool
    reasoning: Literal["single_fact", "multi_fact", "inference"]


class VCheck(BaseModel):
    j: int
    keyword_ok: bool
    typo_ok: bool
    verbose_ok: bool
    followup_ok: bool


class Unans(BaseModel):
    type: Literal["false_premise", "not_covered"]
    question: str
    question_en: str
    why: str


class UCheck(BaseModel):
    j: int
    unanswerable: bool
    type_ok: bool
    natural: bool


class Comp(BaseModel):
    question: str
    question_en: str
    answer: str
    evidence_a: list[str] = Field(min_length=1, max_length=2)
    evidence_b: list[str] = Field(min_length=1, max_length=2)
    qtype: Literal["number", "date", "attribute", "common"]


class CCheck(BaseModel):
    one_comparison: bool
    needs_both: bool
    answer_supported: bool
    identifiable: bool
    clean_text: bool


class Phrased(BaseModel):
    j: int
    question: str
    question_en: str


class PCheck(BaseModel):
    j: int
    matches_spec: bool
    fluent: bool


class Judge(BaseModel):
    c: int
    match: Literal["yes", "partly", "no"]
    answers: bool


class Variants(Listed):
    items: list[Variant]


class VChecks(Listed):
    items: list[VCheck]


class Unanss(Listed):
    items: list[Unans] = Field(min_length=2, max_length=2)


class UChecks(Listed):
    items: list[UCheck]


class Phraseds(Listed):
    items: list[Phrased]


class PChecks(Listed):
    items: list[PCheck]


class Judges(Listed):
    items: list[Judge]


VARIANT_PROMPT = """You extend a RAG test set over ~130k Wikipedia pages about European places. For each item (a
question about the page below and its answer), write in the question's language:
- keyword: a 2-6 word search-box query with the same information need (the place name only if the question has it).
- typo: the question with 1-3 realistic typing errors (missing diacritics, swapped or doubled letters); nothing else.
- verbose: a chatty 2-3 sentence version with personal context (why the asker wants to know), the same information
  need and no extra facts about the place.
- history: a first user turn that names the place (e.g. asking for an overview of it); followup: the question as the
  next turn, referring to the place only with a pronoun or "there", so it is unclear without history.
- time_sensitive: true if the answer may change over time (population, owner, opening, current state).
- reasoning: single_fact (one stated fact), multi_fact (combine 2+ facts), inference (needs a light conclusion).
Return {"items": [...]}, one object per item with j."""

VCHECK_PROMPT = """You check rewrites of test questions. For each item judge (true/false):
- keyword_ok: keyword is a 2-6 word query with the same information need as question, without its answer.
- typo_ok: typo is question with 1-3 realistic typing errors and otherwise unchanged.
- verbose_ok: verbose has the same information need as question, is natural and adds no facts about the place.
- followup_ok: followup asks the same as question, is ambiguous alone, and is clear after history.
Return {"items": [...]}, one object per item with j."""

UNANS_PROMPT = """You extend a RAG test set over ~130k Wikipedia pages about European places with UNANSWERABLE
questions, testing whether a system abstains instead of inventing. Write 2 questions about the place below, in the
article's language, each standing alone (name the place, the asker has no context):
- false_premise: assumes a plausible but false fact that the article contradicts (a wrong century, river, owner,
  height...), e.g. "Why was <castle> rebuilt after the fire of 1850?" when the article has no such fire.
- not_covered: a natural question about this place that the article does not answer at all (opening hours,
  ticket price, a detail never mentioned), which no Wikipedia page is likely to answer.
Each: type, question, question_en (faithful English translation), why (what is false or missing, in English).
Return {"items": [two objects]}."""

UCHECK_PROMPT = """You check unanswerable test questions against the article. For each item judge (true/false):
- unanswerable: the article does not state the information asked for, as asked.
- type_ok: for false_premise, the article contradicts the assumed fact; for not_covered, the article never
  mentions the topic asked about.
- natural: the question is natural, fluent and stands alone (names the place).
Return {"items": [...]}, one object per item with j."""

COMPARE_PROMPT = """You extend a RAG test set over ~130k Wikipedia pages about European places with questions that
need TWO pages. Write one question in the pages' language that compares places A and B (which is older, higher,
larger, ... or what they have in common), naming both, answerable only by combining a fact from A with a fact from
B. ONE question that compares like with like (never two questions joined, never two unrelated asks); if a name
is common, add its locator (municipality or region) so each place is identifiable; no register ids; no typos.
Give: question, question_en (English translation), answer (1-2 sentences), evidence_a and evidence_b (1-2
sentences each, copied EXACTLY from A and from B), qtype (number, date, attribute, common). Never invent facts."""

CCHECK_PROMPT = """You strictly check a two-page comparison test question; when in doubt, false. Judge from the
evidence only:
- one_comparison: a single question comparing comparable properties of the two places (two questions joined, two
  separate asks, or comparing unlike things such as a chamber length with a cave system length = false).
- needs_both: answering needs a fact from evidence_a and one from evidence_b.
- answer_supported: every claim in the answer is stated in the evidence.
- identifiable: both places are named and each name plus its context points to one place (a common name with no
  locator = false); no register ids.
- clean_text: question and answer are free of typos, garbled or invented words, and grammatical."""

PHRASE_PROMPT = """You phrase structured search tasks as natural questions a person would ask a search system over
Wikipedia pages about European places. For each spec write a complete question (interrogative form: "Which ...?",
"What ... are there ...?", never a bare noun phrase like "All lakes near X?") in its language (lang), keeping every
condition exactly (category, country, place name; a distance means "within / at most N km", never "N km away");
add no condition (no country unless the spec has one); inflect names correctly; a parenthesis after a name
disambiguates it: render it naturally ("Mondsee (Speyer)" -> "the Mondsee near Speyer"); use the country's name,
not its code. Also question_en: an English translation of your question (not the spec).
Return {"items": [...]}, one object per spec with j."""

PCHECK_PROMPT = """You check questions phrased from structured specs. For each item judge (true/false):
- matches_spec: the question asks for exactly what the spec says, every condition kept ("within N km", not "N km
  away"), nothing added.
- fluent: a complete, natural, grammatical question in its language (a bare noun phrase is false; check plurals
  and cases).
Return {"items": [...]}, one object per item with j."""

QRELS_PROMPT = """You judge search results for a test set over ~130k Wikipedia pages about European places in many
languages. A user asked the question below without context. For EVERY candidate (title and excerpts), judge only
from what is shown:
- match: "yes" if the page is about a place that fits everything the question says about the place it asks about
  (its kind, location, properties, its name if given); "partly" if nothing shown contradicts the question but some
  conditions are not shown; "no" if it is another kind of place or something shown contradicts the question.
- answers: true if the shown text states the information the question asks for, as asked (correcting a wrong
  assumption of the question does not count).
Pages in other languages are judged like any other page.
Return {"items": [{"c": n, "match": ..., "answers": ...}, ...]}, one object per candidate."""

CATS = {
    "castle": "castles",
    "castle_ruin": "castle ruins",
    "fortification": "fortifications",
    "national_park": "national parks",
    "nature_reserve": "nature reserves",
    "natural_monument": "natural monuments",
    "forest": "forests",
    "cave": "caves",
    "waterfall": "waterfalls",
    "lake": "lakes",
    "mountain": "mountains",
    "garden": "gardens",
}

PAGES = {}


def load_pages():
    for line in open(bunny.PAGES):
        p = json.loads(line)
        PAGES[p["id"]] = p


def ok_items(con):
    """page id -> its answers items that pass every RAG criterion of the Ling judge (with their position n)."""
    labs = {
        i: json.loads(x)
        for i, x in con.execute(
            "select id, items from answer_labels_ling where error is null"
        )
    }
    out = {}
    for i, x in con.execute("select id, items from answers where error is null"):
        its, lb = json.loads(x), labs.get(i)
        if lb and len(lb) == len(its):
            out[i] = [
                it | {"n": n}
                for n, (it, b) in enumerate(zip(its, lb))
                if all(b[k] for k in RAG)
            ]
    return out


def balanced(ids):
    """The first 400 pages per language by id hash (as build_clean.py)."""
    by = collections.defaultdict(list)
    for i in ids:
        by[PAGES[i]["in_language"]].append(i)
    return {i for v in by.values() for i in sorted(v, key=crc)[:400]}


def checked(chain, prompt, msg, n):
    """One structured call (3 tries) whose items must be j = 0..n-1; returns them sorted, or None."""
    for _ in range(3):
        out = bunny.call(chain, [("system", prompt), ("user", msg)])
        if out and sorted(x.j for x in out.items) == list(range(n)):
            return [
                x.model_dump(exclude={"j"})
                for x in sorted(out.items, key=lambda x: x.j)
            ]
    return None


def lines(rows):
    return "\n".join(
        json.dumps({"j": j} | r, ensure_ascii=False) for j, r in enumerate(rows)
    )


def step_variants(con, a):
    items = ok_items(con)
    gen, chk = bunny.llm(Variants, 0.5), bunny.llm(VChecks, 0.0)

    def run(pid):
        p, its = PAGES[pid], items[pid]
        qa = [{"question": it["question"], "answer": it["answer"]} for it in its]
        v = checked(
            gen,
            VARIANT_PROMPT,
            f"Page: {p['title']} ({p['in_language']})\n\n{p['text'][:6000]}\n\nItems:\n{lines(qa)}",
            len(its),
        )
        if v is None:
            return None
        rows = [q | x for q, x in zip(qa, v)]
        c = checked(chk, VCHECK_PROMPT, lines(rows), len(rows))
        if c is None:
            return None
        return [r | {"n": it["n"], "check": k} for r, it, k in zip(rows, its, c)]

    return [i for i in balanced(items) if items[i]], run


def step_unans(con, a):
    items = ok_items(con)
    gen, chk = bunny.llm(Unanss, 0.7), bunny.llm(UChecks, 0.0)

    def run(pid):
        p = PAGES[pid]
        art = f"Article ({p['in_language']}): {p['title']}\n\n{p['text'][:30000]}"
        for _ in range(3):
            out = bunny.call(gen, [("system", UNANS_PROMPT), ("user", art)])
            if out:
                break
        else:
            return None
        rows = [x.model_dump() for x in out.items]
        c = checked(chk, UCHECK_PROMPT, f"{art}\n\nItems:\n{lines(rows)}", len(rows))
        return c and [r | {"check": k} for r, k in zip(rows, c)]

    return [i for i in balanced(items) if items[i]], run


def partner(pid, index):
    p = PAGES[pid]
    cands = {
        c
        for cat in p["categories"]
        for c in index[(p["in_language"], p["country"], cat)]
        if PAGES[c]["wikidata_id"] != p["wikidata_id"]
    }
    return min(cands, key=lambda c: crc(pid + c), default=None)


def step_compare(con, a):
    items = ok_items(con)
    index = collections.defaultdict(list)
    for p in PAGES.values():
        if p["char_count"] >= 1500:
            for cat in p["categories"]:
                index[(p["in_language"], p["country"], cat)].append(p["id"])
    gen, chk = bunny.llm(Comp, 0.5), bunny.llm(CCheck, 0.0)

    def run(pid):
        b = partner(pid, index)
        if b is None:
            return []  # no similar page: done, nothing to ask
        A, B = PAGES[pid], PAGES[b]
        msg = f"A: {A['title']}\n{A['text'][:12000]}\n\nB: {B['title']}\n{B['text'][:12000]}"
        for _ in range(3):
            out = bunny.call(gen, [("system", COMPARE_PROMPT), ("user", msg)])
            if out:
                break
        else:
            return None
        r = out.model_dump()
        c = bunny.call(
            chk,
            [("system", CCHECK_PROMPT), ("user", json.dumps(r, ensure_ascii=False))],
        )
        if c is None:
            return None
        sa, oka = spans(A["text"], r["evidence_a"])
        sb, okb = spans(B["text"], r["evidence_b"])
        return [
            r
            | {
                "pages": [pid, b],
                "spans_a": sa,
                "spans_b": sb,
                "verbatim": oka and okb,
                "check": c.model_dump(),
            }
        ]

    return [i for i in balanced(items) if PAGES[i]["char_count"] >= 1500], run


def km(lat, lon, lats, lons):
    p = math.pi / 180
    h = (
        np.sin((lats - lat) * p / 2) ** 2
        + math.cos(lat * p) * np.cos(lats * p) * np.sin((lons - lon) * p / 2) ** 2
    )
    return 12742 * np.arcsin(np.sqrt(h))


def meta_specs():
    """spec id -> spec: list (every place of a category in a country, 2-20 places) and geo (places of a category
    within r km of an anchor page, 1-15 places, anchor excluded). gold = Wikidata ids; every page of them is
    relevant. Language: a hash pick among the languages the places (list) or the anchor (geo) have pages in."""
    qid = {}
    for p in PAGES.values():
        q = qid.setdefault(
            p["wikidata_id"],
            {
                "cats": set(p["categories"]),
                "country": p["country"],
                "lat": p["latitude"],
                "lon": p["longitude"],
                "pages": [],
            },
        )
        q["pages"].append(p["id"])
    specs = {}
    groups = collections.defaultdict(list)
    for q, v in qid.items():
        for c in v["cats"]:
            groups[(c, v["country"])].append(q)
    for (c, country), qs in groups.items():
        if 2 <= len(qs) <= 20:
            langs = sorted(
                {PAGES[i]["in_language"] for q in qs for i in qid[q]["pages"]}
            )
            lang = langs[crc(c + country) % len(langs)]
            specs[f"list|{c}|{country}"] = {
                "kind": "list",
                "lang": lang,
                "spec": f"all {CATS[c]} in country {country}",
                "gold": sorted(qs),
            }
    qs = [q for q in qid if qid[q]["lat"] is not None]
    lats, lons = (
        np.array([qid[q]["lat"] for q in qs]),
        np.array([qid[q]["lon"] for q in qs]),
    )
    anchors = sorted(
        balanced([i for i in PAGES if PAGES[i]["latitude"] is not None]), key=crc
    )[:4000]
    for pid in anchors:
        p = PAGES[pid]
        d = km(p["latitude"], p["longitude"], lats, lons)
        opts = []
        for r in (5, 10, 25):
            near = [qs[k] for k in np.where(d <= r)[0] if qs[k] != p["wikidata_id"]]
            for c in CATS:
                gold = [q for q in near if c in qid[q]["cats"]]
                if 1 <= len(gold) <= 15:
                    opts.append((r, c, gold))
        if opts:
            r, c, gold = opts[crc(pid) % len(opts)]
            specs[f"geo|{pid}"] = {
                "kind": "geo",
                "lang": p["in_language"],
                "spec": f"all {CATS[c]} within {r} km of {p['title']}",
                "anchor": pid,
                "radius_km": r,
                "category": c,
                "gold": sorted(gold),
            }
    for s in specs.values():
        s["gold_pages"] = sorted(i for q in s["gold"] for i in qid[q]["pages"])
    return specs


def step_meta(con, a):
    specs = meta_specs()
    keys = sorted(specs, key=crc)
    batches = {f"meta|{n}": keys[n : n + 20] for n in range(0, len(keys), 20)}
    gen, chk = bunny.llm(Phraseds, 0.5), bunny.llm(PChecks, 0.0)

    def run(bid):
        ss = [specs[k] for k in batches[bid]]
        v = checked(
            gen,
            PHRASE_PROMPT,
            lines([{"lang": s["lang"], "spec": s["spec"]} for s in ss]),
            len(ss),
        )
        if v is None:
            return None
        c = checked(
            chk,
            PCHECK_PROMPT,
            lines([{"spec": s["spec"]} | x for s, x in zip(ss, v)]),
            len(ss),
        )
        return c and [
            {"key": k} | s | x | {"check": y}
            for k, s, x, y in zip(batches[bid], ss, v, c)
        ]

    return list(batches), run


SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def excerpt(p, q, n=6):
    """Lead + the n sentences sharing the most idf weight with the question: where a matching fact would be. Words
    match on their first 5 letters (a crude stemmer: inflected languages)."""
    N = len(bm25._ids)
    qt = {w[:5]: math.log(N / bm25._df.get(w, 1)) for w in bm25.tok(q)}
    sents = [s for s in SENT.split(p["text"][700:]) if len(s) > 20]
    score = [
        sum(qt[w] for w in qt.keys() & {w[:5] for w in bm25.tok(s)}) for s in sents
    ]
    top = sorted(sorted(range(len(sents)), key=lambda k: -score[k])[:n])
    return (
        " ".join(p["text"][:700].split())
        + " ... "
        + " ... ".join(sents[k][:400] for k in top)
    )


def step_qrels(con, a):
    """Questions: ok challenge items, corpus items of balanced pages, checked unanswerable items; balanced first.
    Row id = page|crc of the question."""
    items = ok_items(con)
    bal = balanced(items)
    qs = []
    for pid, its in items.items():
        for it in its:
            if it["kind"] == "challenge" or pid in bal:
                qs.append((pid, it["kind"], it["question"]))
    for pid, x in con.execute(
        "select id, items from unanswerable where items is not null"
    ):
        for it in json.loads(x):
            if all(it["check"].values()):
                qs.append((pid, it["type"], it["question"]))
    by_q = collections.defaultdict(list)
    for p in PAGES.values():
        by_q[p["wikidata_id"]].append(p["id"])
    jobs = {f"{pid}|{crc(q):08x}": (pid, kind, q) for pid, kind, q in qs}
    order = sorted(jobs, key=lambda k: (jobs[k][0] not in bal, crc(k)))
    chain = bunny.llm(Judges, 0.0)
    r = bm25.load()
    lock = threading.Lock()

    def pool_of(
        k,
    ):  # BM25 top 20 (bm25s under a lock: one query is ms, Ling calls take a minute)
        pid = jobs[k][0]
        with lock:
            docs, _ = r.retrieve(
                [bm25.tok(jobs[k][2]) or ["_"]], k=20, show_progress=False
            )
        ranked = [bm25._ids[d] for d in docs[0]]
        pool = dict.fromkeys(ranked + by_q[PAGES[pid]["wikidata_id"]][:6] + [pid])
        return sorted(pool, key=lambda c: crc(k + c)), {
            c: n + 1 for n, c in enumerate(ranked)
        }

    def run(k):
        pid, kind, q = jobs[k]
        pool, rank = pool_of(k)
        msg = f"Question: {q}\n\nCandidates:\n" + "\n\n".join(
            f"[{n}] {PAGES[c]['title']} ({PAGES[c]['in_language']}): {excerpt(PAGES[c], q)}"
            for n, c in enumerate(pool)
        )
        for _ in range(3):
            out = bunny.call(chain, [("system", QRELS_PROMPT), ("user", msg)])
            if out and sorted(x.c for x in out.items) == list(range(len(pool))):
                js = sorted(out.items, key=lambda x: x.c)
                return {
                    "question": q,
                    "kind": kind,
                    "gold": pid,
                    "cands": [
                        {
                            "id": c,
                            "bm25_rank": rank.get(c),
                            "match": x.match,
                            "answers": x.answers,
                        }
                        for c, x in zip(pool, js)
                    ],
                }
        return None

    return order, run


class TSpec(BaseModel):
    axis: Literal["col", "row"]
    index: int
    exclude: list[int] = []
    op: Literal["max", "min", "mean", "sum", "count", "argmax", "argmin"]
    unit: str = ""
    question: str
    question_en: str


class TOut(BaseModel):
    usable: bool
    items: list[TSpec] = Field(default=[], max_length=3)


TABLE_PROMPT = """You write test questions over a DATA TABLE of a Wikipedia article about a European place; a program
computes the answers, so you only choose what to compute. Rows are numbered r0.., columns c0.. (c0 = the row label).
usable = false if it is not a data table (an infobox of key/value facts, a mixed list) or no numeric line makes
sense to aggregate. Else up to 3 specs, each different:
- axis "col" (a column over the rows) or "row" (a row over the columns), index of that column or row;
- exclude: the row (axis col) or column (axis row) indices to leave out: totals, sums, annual values, sub-headers;
- op: max, min, mean, sum, count (how many numeric entries), argmax / argmin (which row label, or column header,
  has the largest / smallest value). Only meaningful ones: no mean of years or ids, no sum of heights;
- unit of the values (e.g. "m", "°C", "ha", "" if none);
- question: in the article's language, standing alone (name the place and what the table lists, never "the
  table"), asking exactly for that computation; question_en: its English translation.
Return {"usable": ..., "items": [...]}."""


def compute(t, s):
    """The answer of spec s on table t (tables.tables row): value, label (argmax/argmin), and the values used."""
    from tables import number

    rows, head = t["rows"], t["header"]
    if s["axis"] == "col":
        if not 0 < s["index"] < len(head):
            return None
        vals = [
            (r[0], number(r[s["index"]]))
            for i, r in enumerate(rows)
            if i not in s["exclude"]
        ]
    else:
        if not 0 <= s["index"] < len(rows):
            return None
        r = rows[s["index"]]
        vals = [
            (head[c], number(r[c]))
            for c in range(1, len(head))
            if c not in s["exclude"]
        ]
    vals = [(lab, v) for lab, v in vals if v is not None]
    if len(vals) < 2:
        return None
    xs = [v for _, v in vals]
    op = s["op"]
    if op in ("argmax", "argmin"):
        best = (max if op == "argmax" else min)(vals, key=lambda x: x[1])
        return {"value": best[1], "label": best[0], "values": vals}
    value = {
        "max": max,
        "min": min,
        "sum": sum,
        "count": len,
        "mean": lambda x: sum(x) / len(x),
    }[op](xs)
    return {"value": round(value, 2), "label": None, "values": vals}


def step_tables(con, a):
    """Computed table questions: every data table (tables.py) -> up to 3 specs by Ling, answers by compute()."""
    import tables

    found = {
        f"{pid}|{n}": t
        for pid, p in PAGES.items()
        for n, t in enumerate(tables.tables(p["text"]))
    }
    gen = bunny.llm(TOut, 0.3)

    def run(k):
        pid, t = k.split("|")[0], found[k]
        p = PAGES[pid]
        lines = ["      " + " | ".join(f"c{c}: {h}" for c, h in enumerate(t["header"]))]
        lines += [f"r{i}: " + " | ".join(r) for i, r in enumerate(t["rows"])]
        msg = f"Article ({p['in_language']}): {p['title']}\n\n" + "\n".join(lines)
        for _ in range(3):
            out = bunny.call(gen, [("system", TABLE_PROMPT), ("user", msg)])
            if out:
                break
        else:
            return None
        items = []
        for x in out.items if out.usable else []:
            spec = x.model_dump()
            items.append(
                spec
                | {
                    "answer": compute(t, spec),
                    "table": t["header"],
                    "n_rows": len(t["rows"]),
                }
            )
        return items

    return sorted(found, key=crc), run


STEPS = {
    "tables": ("table_q", step_tables),
    "variants": ("variants", step_variants),
    "unans": ("unanswerable", step_unans),
    "compare": ("compare", step_compare),
    "meta": ("meta_q", step_meta),
    "qrels": ("qrels", step_qrels),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rpm", type=int)
    a = ap.parse_args()
    table, fn = STEPS[a.step]
    con = sqlite3.connect(bunny.DB, timeout=120, check_same_thread=False)
    con.execute(
        f"create table if not exists {table} (id text primary key, items text, error text, model text, created_at text default current_timestamp)"
    )
    con.execute(
        "create table if not exists unanswerable (id text primary key, items text, error text, model text, created_at text default current_timestamp)"
    )
    load_pages()
    bunny.RPM = a.rpm or bunny.RPM
    keys, run = fn(con, a)
    done = {i for (i,) in con.execute(f"select id from {table} where error is null")}
    keys = [k for k in keys if k not in done][: a.limit]
    print(len(keys), a.step, "to do", flush=True)
    n = 0
    for k, res in bunny.run_all(lambda k: (k, run(k)), keys, a.workers):
        if res is None and bunny.STOP.is_set():
            continue
        con.execute(
            f"insert or replace into {table} (id, items, error, model) values (?, ?, ?, ?)",
            (
                k,
                None if res is None else json.dumps(res, ensure_ascii=False),
                "failed" if res is None else None,
                bunny.MODEL_ID,
            ),
        )
        con.commit()
        n += 1
        if n % 200 == 0:
            print(n, "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
