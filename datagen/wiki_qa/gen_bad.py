"""Temporary: LLM-written bad variants of good question items (stealth/space-bunny-alpha), one target problem each,
balanced across problems, for training the quality classifier. Labels follow by construction: the original keep
labels with the target problem's flags and verdict. Resumable: appends to qlabel_synth.jsonl
rows {id, j, problem, item, label}. Needs OPENROUTER_API_KEY. Usage: gen_bad.py [per_problem=150] [workers=12]"""

import glob
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

S = os.path.dirname(os.path.abspath(__file__))
OUT = f"{S}/qlabel_synth.jsonl"

# problem -> (instruction for the rewriter, label overrides incl. verdict)
PROBLEMS = {
    "not_unique": (
        "Remove every distinguishing detail (municipality, region, range, nearby town, country) from the question so a plain, common name remains that could match many places. Keep it natural.",
        {"unique_place": False, "verdict": "fix"},
    ),
    "unrealistic": (
        "Replace the question with a register/database trivia question nobody would really ask about this place (e.g. its catchment/basin code, cadastral number, coordinates, map sheet, protection decree number), with matching facts from the article if any.",
        {"realistic": False, "verdict": "drop"},
    ),
    "not_fluent": (
        "Rewrite the question with 1-2 realistic grammar errors a weak model would make in this language (wrong case or inflection of the name, wrong word order, a wrong preposition), keeping its meaning.",
        {"fluent": False, "verdict": "fix"},
    ),
    "self_answering": (
        "Rewrite the question so it already contains or gives away its own answer (e.g. a yes/no question stating the fact, or the answer embedded in the question).",
        {"self_answering": True, "verdict": "drop"},
    ),
    "unsupported_fact": (
        "Change one fact so it states something plausible but NOT in the article (a wrong number, date, name or claim). Keep the other facts.",
        {"facts_supported": "partly", "verdict": "fix"},
    ),
    "invented_facts": (
        "Replace all facts with plausible-sounding invented facts that are not in the article.",
        {"facts_supported": "no", "facts_answer": False, "verdict": "drop"},
    ),
    "facts_dont_answer": (
        "Keep the facts true to the article but make them not answer the question (facts about a different aspect of the place).",
        {"facts_answer": False, "verdict": "fix"},
    ),
    "bad_translation": (
        "Change question_en so it is a plausible but unfaithful translation (a wrong detail, a mistranslated name or word, a changed meaning). Keep the question itself unchanged.",
        {"translation_ok": False, "verdict": "fix"},
    ),
    "query_leak": (
        "Rewrite the query so it contains words or numbers from the answer (facts) that are not in the question.",
        {"query_ok": False, "verdict": "fix"},
    ),
    "query_not_english": (
        "Rewrite the query as keywords in the article's language instead of English.",
        {"query_ok": False, "verdict": "fix"},
    ),
    "wrong_language": (
        "Rewrite the question in a different European language than the article's (not English unless the article is English), keeping its meaning.",
        {"language_ok": False, "verdict": "fix"},
    ),
}


class Item(BaseModel):
    question: str
    question_en: str
    facts: list[str] = Field(min_length=1, max_length=3)
    query: str


PROMPT = """You create negative training examples for a quality classifier of RAG test questions.
Below is a Wikipedia article about a European place and one good generated item (question in the article's language,
question_en = English translation, facts = short facts from the article answering it, query = English search keywords).
Rewrite the item so that it has exactly this problem, and nothing else changes:

PROBLEM: {instruction}

Make the problem realistic and varied, the kind of mistake a generator model really makes, not an obvious joke.
Return the full rewritten item."""


def pages_and_keeps():
    rows = []
    for d in sorted(glob.glob(f"{S}/qlabel/b_*")):
        pages = {r["id"]: r for r in map(json.loads, open(f"{d}/batch.jsonl"))}
        for lab in map(json.loads, open(f"{d}/labels.jsonl")):
            p = pages[lab["id"]]
            if p["split"] != "train":
                continue
            rows += [
                (p, j, lb)
                for j, lb in enumerate(lab["labels"])
                if lb["verdict"] == "keep"
            ]
    return rows


def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    done = (
        {(r["id"], r["j"], r["problem"]) for r in map(json.loads, open(OUT))}
        if os.path.exists(OUT)
        else set()
    )
    keeps = pages_and_keeps()
    rng = random.Random(7)
    tasks = []
    for prob in PROBLEMS:  # a different random sample of good items per problem
        for p, j, lb in rng.sample(keeps, min(per, len(keeps))):
            if prob == "wrong_language" and p["lang"] == "en":
                continue
            if (p["id"], j, prob) not in done:
                tasks.append((p, j, lb, prob))
    rng.shuffle(tasks)
    print(len(tasks), "to generate", flush=True)
    llm = ChatOpenAI(
        model="stealth/space-bunny-alpha",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.7,
        max_retries=0,
        timeout=120,
    ).with_structured_output(Item, method="json_schema")
    lock, nxt = threading.Lock(), [0.0]

    def run(t):
        p, j, lb, prob = t
        with lock:  # <= 50 requests started per minute
            time.sleep(max(0.0, nxt[0] - time.monotonic()))
            nxt[0] = max(nxt[0], time.monotonic()) + 1.2
        msg = (
            f"Article ({p['lang']}): {p['title']}\n\n{p['text'][:6000]}\n\nOther questions on this page:\n"
            + "\n".join(x["question"] for n, x in enumerate(p["items"]) if n != j)
            + f"\n\nGood item:\n{json.dumps(p['items'][j], ensure_ascii=False)}"
        )
        for _ in range(2):
            try:
                it = llm.invoke(
                    [
                        ("system", PROMPT.format(instruction=PROBLEMS[prob][0])),
                        ("user", msg),
                    ]
                )
                return {
                    "id": p["id"],
                    "j": j,
                    "problem": prob,
                    "item": it.model_dump(),
                    "label": {**lb, **PROBLEMS[prob][1]},
                }
            except Exception as e:
                print(
                    "retry", p["id"], prob, type(e).__name__, str(e)[:120], flush=True
                )
                if "429" in str(e):
                    return None
        return None

    with ThreadPoolExecutor(workers) as ex, open(OUT, "a") as f:
        for n, res in enumerate(ex.map(run, tasks), 1):
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
            if n % 100 == 0:
                print(n, "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
