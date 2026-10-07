"""Step 1: how BM25 success relates to a question's anchors, the tokens it shares with its translation.

    uv run python -m experiments.language_anchors.shared_tokens

Rows: dev, answer_ok, x_ok, with stored BM25 ranks for both versions (`rank_o[3]`, `rank_x[3]`). An anchor is a
token (casefolded `\\w+`) in both `question` and `question_x` with 3+ characters or a digit. Kind: `number` if it
has a digit, `name` if it is capitalised inside the original question (not as its first word), else `other`
(cognates, shared words). Prints markdown tables; read-only.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from src.db.dataset import ROOT

WORD = re.compile(r"\w+")
SCRIPT = {
    "bg": "cyrillic",
    "el": "greek",
}  # every other language in the set is written in Latin script


def anchors(question: str, question_x: str) -> dict[str, list[str]]:
    """Tokens shared by both versions, by kind."""
    shared = {t.casefold() for t in WORD.findall(question)} & {
        t.casefold() for t in WORD.findall(question_x)
    }
    shared = {t for t in shared if len(t) >= 3 or any(c.isdigit() for c in t)}
    capitalised = {t.casefold() for t in WORD.findall(question)[1:] if t[:1].isupper()}
    kinds: dict[str, list[str]] = {"number": [], "name": [], "other": []}
    for t in sorted(shared):
        kind = (
            "number"
            if any(c.isdigit() for c in t)
            else "name"
            if t in capitalised
            else "other"
        )
        kinds[kind].append(t)
    return kinds


def load() -> pd.DataFrame:
    df = duckdb.sql(
        f"""select id, kind, lang, x_lang, machine_generated, question, question_x,
        rank_o[3] <= 10 as hit_o, rank_x[3] <= 10 as hit_x
        from '{ROOT / "qa" / "wiki_qa_rag.parquet"}'
        where answer_ok and x_ok and split = 'dev' and rank_o[3] is not null and rank_x[3] is not null"""  # nosec B608 - constant path
    ).df()
    a = [anchors(q, x) for q, x in zip(df.question, df.question_x, strict=True)]
    for kind in ("number", "name", "other"):
        df[f"n_{kind}"] = [len(x[kind]) for x in a]
    df["n_anchors"] = df.n_number + df.n_name + df.n_other
    df["anchor_types"] = [
        "+".join(k for k in ("number", "name") if x[k])
        or ("other only" if x["other"] else "none")
        for x in a
    ]
    df["script_pair"] = [
        "same script"
        if SCRIPT.get(o, "latin") == SCRIPT.get(x, "latin")
        else "different script"
        for o, x in zip(df.lang, df.x_lang, strict=True)
    ]
    df["example"] = [
        "; ".join(f"{k}: {', '.join(v[:4])}" for k, v in x.items() if v) for x in a
    ]
    return df


def rates(g: pd.DataFrame) -> pd.Series:
    """BM25 hit@10 for the original and the translated question, and how often a found page survives translation."""
    return pd.Series(
        {
            "n": len(g),
            "hit10_original": g.hit_o.mean(),
            "hit10_translated": g.hit_x.mean(),
            "survives": g[g.hit_o].hit_x.mean() if g.hit_o.any() else float("nan"),
        }
    )


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n### {title}\n")
    print(table.round(3).to_markdown())


def main() -> None:
    df = load()
    show(
        "Anchors per question",
        df.groupby("kind").agg(
            n=("id", "size"),
            mean_anchors=("n_anchors", "mean"),
            any_anchor=("n_anchors", lambda s: (s > 0).mean()),
            any_number=("n_number", lambda s: (s > 0).mean()),
            any_name=("n_name", lambda s: (s > 0).mean()),
        ),
    )
    df["anchor_bucket"] = df.n_anchors.clip(upper=4).map(
        lambda n: "4+" if n == 4 else str(n)
    )
    for scope, part in [("all", df), ("excl. bot pages", df[~df.machine_generated])]:
        show(
            f"BM25 by number of anchors ({scope})",
            part.groupby(["kind", "anchor_bucket"]).apply(rates, include_groups=False),
        )
    show(
        "BM25 by anchor type",
        df.groupby(["kind", "anchor_types"]).apply(rates, include_groups=False),
    )
    show(
        "BM25 by script pair",
        df.groupby(["kind", "script_pair"]).apply(rates, include_groups=False),
    )
    sample = df[(df.kind == "challenge") & df.hit_o].sample(8, random_state=7)
    print("\n### Examples (hard, found in the original language)\n")
    for r in sample.itertuples():
        print(
            f"- [{r.lang}>{r.x_lang}] translated found: {r.hit_x} | anchors: {r.example or 'none'}\n  {r.question}\n  {r.question_x}"
        )


if __name__ == "__main__":
    main()
