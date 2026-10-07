"""Temporary: the tag set. One source for the teacher prompt, the Laya questions and the label schema.
SPEC[name] = (kind, question, {option: (short text for Laya, gloss for the teacher)}) ; kind "bool" has no options."""

SPEC = {
    "article_type": (
        "choice",
        "What kind of Wikipedia article is `article`?",
        {
            "descriptive": (
                "prose describing the place",
                "prose describing the place: its features, surroundings, history or use",
            ),
            "registry_stub": (
                "infobox and a template sentence",
                "mostly an infobox or register numbers with only one or two template sentences",
            ),
            "list_or_disambiguation": (
                "list or disambiguation page",
                "a list of places or a page pointing to other pages",
            ),
            "other": ("something else", "none of the above"),
        },
    ),
    "information_richness": (
        "choice",
        "How much does `article` tell about the place?",
        {
            "low": (
                "only name, location, numbers",
                "almost nothing beyond name, location and numbers",
            ),
            "medium": ("a few real facts", "a few real facts or a short description"),
            "high": (
                "a rich description",
                "a rich, detailed description with many distinct facts",
            ),
        },
    ),
    "tourist_appeal": (
        "choice",
        "How interesting does the place in `article` look for a visitor?",
        {
            "low": (
                "nothing to draw a visitor",
                "nothing that would draw a visitor: an ordinary pond, a registry entry, a site with no visible features",
            ),
            "medium": (
                "pleasant or notable",
                "pleasant or locally notable, but ordinary",
            ),
            "high": (
                "distinctive, worth a trip",
                "distinctive scenery, a landmark, or attractions and activities that would make someone plan a trip",
            ),
        },
    ),
    "primary_focus": (
        "choice",
        "What is most of `article` about?",
        {
            "description": (
                "what the place is like",
                "what the place looks like and consists of: landscape, buildings, nature",
            ),
            "history": (
                "its history",
                "events, dates, owners and people connected to the place",
            ),
            "visiting": (
                "visiting and activities",
                "how to get there, trails, activities, facilities",
            ),
            "data": (
                "measurements and register data",
                "measurements, codes, coordinates and administrative facts",
            ),
            "protection": (
                "legal protection",
                "protected-area status, regulations and conservation",
            ),
            "other": ("something else", "none of the above"),
        },
    ),
    "visitor_info": (
        "bool",
        "Does `article` tell how to reach or visit the place, or what to do there?",
        None,
    ),
    "history": ("bool", "Does `article` tell the history of the place?", None),
    "nature": (
        "bool",
        "Does `article` describe landscape, plants, animals or ecology?",
        None,
    ),
    "culture": (
        "bool",
        "Does `article` mention cultural or architectural significance, legends or notable people?",
        None,
    ),
}
IDS = list(SPEC)


def label_model():
    from typing import Literal
    from pydantic import create_model

    return create_model(
        "Label",
        **{
            k: (bool if v[0] == "bool" else Literal[tuple(v[2])], ...)
            for k, v in SPEC.items()
        },
    )


def cls_prompt():  # system prompt for the E4B classifier (teacher prompt without the article slot)
    return teacher_prompt().replace(
        "\n\nARTICLE:\n", "\n\nReply with a JSON object with exactly these keys."
    )


def laya_questions():
    q = {}
    for k, (kind, text, opts) in SPEC.items():
        q[k] = (
            {"type": "noul", "instructions": text}
            if kind == "bool"
            else {
                "type": "choice",
                "instructions": text,
                "criteria": {o: v[0] for o, v in opts.items()},
            }
        )
    return q


def target(lab):  # label dict -> class index per question
    return [
        int(lab[k]) if SPEC[k][0] == "bool" else list(SPEC[k][2]).index(lab[k])
        for k in IDS
    ]


def teacher_prompt():
    lines = []
    for k, (kind, text, opts) in SPEC.items():
        lines.append(
            f"- {k}: {text.replace('`article`', 'the article')}"
            + (" Answer true or false." if kind == "bool" else "")
        )
        for o, v in (opts or {}).items():
            lines.append(f"    {o} = {v[1]}")
    return (
        "Judge this Wikipedia article about a place (any language).\n"
        + "\n".join(lines)
        + "\n\nARTICLE:\n"
    )


def labelled():  # teacher-labelled rows from both pools (label_pool.json, label_pool2.json): {i, split, lang, text, label}; test = heldout + dev
    import json
    import os

    S = os.path.dirname(os.path.abspath(__file__))
    pool = {
        r["i"]: r
        for f in ("label_pool.json", "label_pool2.json")
        for r in json.load(open(f"{S}/{f}"))
    }
    return [
        {**pool[r["i"]], "label": r["label"]}
        for r in map(json.loads, open(f"{S}/labels.jsonl"))
    ]
