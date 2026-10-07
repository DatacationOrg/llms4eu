"""Temporary: quality labels for one generated question item (question, question_en, facts, query) given its
article and the page's other questions. One source for the agent rules, the Laya questions and the checker.
SPEC[name] = (kind, question, {option: (short Laya text, gloss for annotators)}); kind "bool" has no options.
Usage: qspec.py check qlabel/<dir>   (validates an agent's labels.jsonl against its batch.jsonl)"""

import json
import sys

SPEC = {
    "duplicate": (
        "bool",
        "Does `item` ask the same thing as, or share its answer with, one of `other_questions`?",
        None,
    ),
    "language_ok": (
        "bool",
        "Is the question in `item` written in the language of `article`?",
        None,
    ),
    "fluent": (
        "bool",
        "Is the question in `item` fluent and grammatical, with names correctly inflected?",
        None,
    ),
    "realistic": (
        "bool",
        "Would a real visitor or curious person ask the question in `item`, rather than a database or trivia question?",
        None,
    ),
    "self_answering": (
        "bool",
        "Does the question in `item` already contain or give away its own answer?",
        None,
    ),
    "unique_place": (
        "bool",
        "Does the question in `item` single out this one place, so it fits no other of ~100k European place pages?",
        None,
    ),
    "facts_supported": (
        "choice",
        "Are the facts in `item` stated in `article`?",
        {
            "yes": (
                "all stated in the article",
                "every fact is stated in or directly follows from the article",
            ),
            "partly": (
                "some are not in the article",
                "at least one fact is unsupported, wrong, or garbled, but some are right",
            ),
            "no": (
                "not in the article or wrong",
                "the facts are invented, wrong, or unreadable",
            ),
        },
    ),
    "facts_answer": (
        "bool",
        "Do the facts in `item` actually answer its question?",
        None,
    ),
    "translation_ok": (
        "bool",
        "Is the English question in `item` a faithful translation of the question?",
        None,
    ),
    "query_ok": (
        "bool",
        "Is the query in `item` English keywords built only from the question, without the answer?",
        None,
    ),
    "answer_difficulty": (
        "choice",
        "How hard is it to answer the question in `item` from `article`?",
        {
            "easy": (
                "one stated fact",
                "a single number, name or sentence stated directly",
            ),
            "medium": (
                "a few facts or a passage",
                "needs a passage read or two or three facts combined",
            ),
            "hard": (
                "combining or inferring",
                "needs facts from several parts of the article combined, or light inference",
            ),
        },
    ),
    "retrieval_difficulty": (
        "choice",
        "How hard is it to find `article` among ~100k pages from the question in `item` alone?",
        {
            "easy": (
                "the page title is in the question",
                "the exact page name is in the question; keyword search finds it",
            ),
            "medium": (
                "named but paraphrased or common name",
                "named but with a common or inflected name, or the rest is paraphrased",
            ),
            "hard": (
                "described, not named",
                "the place is described rather than named; needs semantic matching",
            ),
        },
    ),
    "verdict": (
        "choice",
        "Should `item` be kept as a RAG test question?",
        {
            "keep": ("good as it is", "correct and useful as it is"),
            "fix": (
                "useful but needs an edit",
                "worth keeping after a small edit (grammar, translation, query, a fact)",
            ),
            "drop": (
                "remove it",
                "wrong, unanswerable, trivial, duplicate or unfixable",
            ),
        },
    ),
}
IDS = list(SPEC)

# Criteria for a RAG triple (context: a chunk of an article, possibly the wrong one; question; answer), all bool so a
# classifier is one multi-label head. A good triple is all True.
RAG = {
    "context_relevant": "Does `context` contain information about what `question` asks?",
    "answerable": "Can `question` be fully answered from `context` alone?",
    "answer_relevant": "Does `answer` respond to what `question` asks (even if it is wrong)?",
    "answer_supported": "Is every claim in `answer` stated in or directly implied by `context`?",
    "answer_complete": "Does `answer` answer every part of `question`?",
    "language_match": "Is `answer` written in the same language as `question`?",
    "answer_fluent": "Is `answer` fluent and grammatical in its language?",
}


def options(k):
    return [False, True] if SPEC[k][0] == "bool" else list(SPEC[k][2])


def laya_questions():
    return {
        k: {"type": "noul", "instructions": text}
        if kind == "bool"
        else {
            "type": "choice",
            "instructions": text,
            "criteria": {o: v[0] for o, v in opts.items()},
        }
        for k, (kind, text, opts) in SPEC.items()
    }


def target(lab):  # label dict -> class index per tag
    return [options(k).index(lab[k]) for k in IDS]


def state(page, j, max_chars=6000):  # Laya input for item j of a page
    # item first, article last: Laya truncates at max_len, and the item must always survive
    its = page["items"]
    return {
        "item": json.dumps(its[j], ensure_ascii=False),
        "other_questions": "\n".join(
            it["question"] for n, it in enumerate(its) if n != j
        )
        or "(none)",
        "article": page["text"][:max_chars],
    }


def problems(d):
    batch = {r["id"]: r for r in map(json.loads, open(f"{d}/batch.jsonl"))}
    try:
        got = [json.loads(line) for line in open(f"{d}/labels.jsonl")]
    except (OSError, json.JSONDecodeError) as e:
        return [f"labels.jsonl: {e}"]
    errs = [f"missing page {i}" for i in set(batch) - {r.get("id") for r in got}]
    for r in got:
        if r.get("id") not in batch:
            errs.append(f"unexpected page {r.get('id')}")
            continue
        labs, n = r.get("labels"), len(batch[r["id"]]["items"])
        if not isinstance(labs, list) or len(labs) != n:
            errs.append(f"{r['id']}: need {n} label dicts, one per item in order")
            continue
        bad = [
            f"{r['id']} item {j}: {k} must be one of {options(k)}, got {lab.get(k)!r}"
            for j, lab in enumerate(labs)
            for k in IDS
            if k not in lab or lab[k] not in options(k)
        ]
        errs += bad or consistency(r["id"], batch[r["id"]]["items"], labs)
    return errs


def consistency(pid, items, labs):
    """Label combinations that contradict each other or the data itself."""
    errs, seen = [], set()
    for j, (it, lab) in enumerate(zip(items, labs)):
        q = it["question"].strip().lower()
        if q in seen and not lab["duplicate"]:
            errs.append(
                f"{pid} item {j}: same question text as an earlier item, so duplicate must be true"
            )
        seen.add(q)
        flags = [k for k in ("duplicate", "self_answering") if lab[k]]
        flags += [
            k
            for k in (
                "language_ok",
                "fluent",
                "realistic",
                "unique_place",
                "facts_answer",
                "translation_ok",
                "query_ok",
            )
            if not lab[k]
        ]
        flags += ["facts_supported"] if lab["facts_supported"] != "yes" else []
        if lab["verdict"] == "keep" and flags:
            errs.append(
                f"{pid} item {j}: verdict keep contradicts {flags}; use fix or drop, or correct the flag"
            )
        if lab["verdict"] != "drop" and (
            lab["duplicate"] or lab["facts_supported"] == "no"
        ):
            errs.append(
                f"{pid} item {j}: a duplicate or unsupported item must be verdict drop"
            )
    return errs


if __name__ == "__main__" and sys.argv[1] == "check":
    errs = problems(sys.argv[2])
    print("\n".join(errs[:40]) or "OK")
    sys.exit(bool(errs))
