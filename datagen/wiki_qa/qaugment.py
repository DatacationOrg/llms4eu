"""Temporary: synthetic bad items for the quality classifier (training only; val keeps the natural distribution).
Each corruption starts from an item labelled keep and sets the labels it breaks. Usage: augment(rows, rng) -> extra rows."""

import copy
import json
import re


def _swap(page, j, item, label):
    p = copy.deepcopy(page)
    p["items"][j] = item
    return {"page": p, "j": j, "label": label, "split": "train", "synthetic": True}


def _bad(label, verdict, **flags):
    return {**label, **flags, "verdict": verdict}


def corruptions(r, other_page, rng):
    """All applicable corruptions of keep-item r; other_page supplies foreign facts/translations."""
    p, j, lab = r["page"], r["j"], r["label"]
    it, others = p["items"][j], [x for n, x in enumerate(p["items"]) if n != j]
    out = []
    if others:
        o = rng.choice(others)
        out.append(
            (
                "duplicate",
                {
                    **it,
                    "question": o["question"],
                    "question_en": o["question_en"],
                    "query": o["query"],
                },
                _bad(lab, "drop", duplicate=True),
            )
        )
        out.append(
            (
                "facts_swap",
                {**it, "facts": o["facts"]},
                _bad(lab, "fix", facts_answer=False),
            )
        )
        out.append(
            (
                "translation",
                {**it, "question_en": o["question_en"]},
                _bad(lab, "fix", translation_ok=False),
            )
        )
    fact = rng.choice(it["facts"])
    out.append(
        (
            "self_answering",
            {**it, "question": f"{it['question'].rstrip('?').rstrip()} — {fact}?"},
            _bad(lab, "drop", self_answering=True),
        )
    )
    nums = re.findall(r"\d[\d.,]*|\w{6,}", fact)
    if nums:
        out.append(
            (
                "query_leak",
                {**it, "query": f"{it['query']} {' '.join(nums[:3])}"},
                _bad(lab, "fix", query_ok=False),
            )
        )
    if p["lang"] != "en":
        out.append(
            (
                "query_native",
                {**it, "query": it["question"].rstrip("?")},
                _bad(lab, "fix", query_ok=False),
            )
        )
        out.append(
            (
                "english_question",
                {**it, "question": it["question_en"]},
                _bad(lab, "fix", language_ok=False),
            )
        )
    words = it["question"].rstrip("?").split()
    if len(words) > 5:
        w = words[:]
        k = rng.randrange(1, len(w) - 2)
        w[k], w[k + 1] = w[k + 1], w[k]  # swapped neighbours + a mangled word
        i = rng.randrange(len(w))
        if len(w[i]) > 4:
            w[i] = w[i][:-2] + w[i][-1] + w[i][-2]
        out.append(
            (
                "garbled",
                {**it, "question": " ".join(w) + "?"},
                _bad(lab, "fix", fluent=False),
            )
        )
    if other_page["items"]:
        foreign = rng.choice(other_page["items"])["facts"]
        out.append(
            (
                "foreign_facts",
                {**it, "facts": foreign},
                _bad(lab, "drop", facts_supported="no", facts_answer=False),
            )
        )
    return out


def augment(rows, rng, per_item=1):
    """per_item random corruptions for every keep row (labels consistent with qspec.consistency)."""
    pages = [r["page"] for r in rows]
    extra = []
    for r in rows:
        if r["label"]["verdict"] != "keep":
            continue
        cands = corruptions(r, rng.choice(pages), rng)
        for _, item, label in rng.sample(cands, min(per_item, len(cands))):
            extra.append(_swap(r["page"], r["j"], item, label))
    return extra


if __name__ == "__main__":  # self-check on a toy page
    import random

    page = {
        "lang": "de",
        "items": [
            {
                "question": "Wie hoch ist der Berg X bei Y im Land Z?",
                "question_en": "How high is X?",
                "facts": ["Er ist 1200 m hoch."],
                "query": "X height",
            },
            {
                "question": "Wer baute die Burg X bei Y?",
                "question_en": "Who built X castle?",
                "facts": ["Graf Otto baute sie."],
                "query": "X castle builder",
            },
        ],
    }
    keep = dict(
        duplicate=False,
        language_ok=True,
        fluent=True,
        realistic=True,
        self_answering=False,
        unique_place=True,
        facts_supported="yes",
        facts_answer=True,
        translation_ok=True,
        query_ok=True,
        answer_difficulty="easy",
        retrieval_difficulty="easy",
        verdict="keep",
    )
    r = {"page": page, "j": 0, "label": keep}
    names = [c[0] for c in corruptions(r, page, random.Random(0))]
    assert {
        "duplicate",
        "self_answering",
        "query_leak",
        "english_question",
        "garbled",
        "foreign_facts",
    } <= set(names), names
    ex = augment([r], random.Random(0), per_item=3)
    assert len(ex) == 3 and all(e["label"]["verdict"] != "keep" for e in ex)
    assert page["items"][0]["query"] == "X height"  # original untouched
    print("ok", names)


def load_synth(rows, path):
    """LLM-written bad variants (gen_bad.py) as train rows; rows supply the original pages."""
    pages = {r["page"]["id"]: r["page"] for r in rows}
    out = []
    for s in map(json.loads, open(path)):
        if s["id"] in pages:
            out.append(_swap(pages[s["id"]], s["j"], s["item"], s["label"]))
    return out


def load_balanced(pattern):
    """Agent-written balanced set (synth2/a_*/variants.jsonl): one target class per variant, all 13 labels given."""
    import glob

    out = []
    for d in sorted(glob.glob(pattern)):
        page = json.load(open(f"{d}/page.json"))
        for v in map(json.loads, open(f"{d}/variants.jsonl")):
            out.append(_swap(page, v["j"], v["item"], v["labels"]))
    return out
