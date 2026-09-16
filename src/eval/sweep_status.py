"""Report how far chunk-variant preparation has got.

Read-only, deliberately: it never migrates the schema, so it is safe to point at
the durable database. A table that does not exist yet reads as zero rather than
being created as a side effect of asking about it.

Answers the two questions that decide whether a sweep can run yet: does every
variant have chunks, and do they all have labels for the same questions? Variants
labelled to different depths are not comparable, so the sweep refuses to rank
them and this is where that shows up first.
"""

from __future__ import annotations

import argparse

from src.db.pages import connect_pages as connect
from src.shared.env import chroma_path


def format_status() -> str:
    with connect() as conn:
        approved = int(
            conn.execute(
                "select count(*) from eval_questions where approved = 1"
            ).fetchone()[0]
        )
        anchored = _count(conn, "eval_answer_anchors")
        rows = conn.execute(
            """
            select c.variant,
                   count(distinct c.id) as chunks,
                   count(distinct r.question_id) as questions,
                   count(r.chunk_id) as labels,
                   sum(c.char_count) as store_chars
            from page_chunks c
            left join eval_relevant_chunks r on r.chunk_id = c.id
            group by c.variant
            order by c.variant
            """
        ).fetchall()

    lines = [
        "=== answer anchors ===",
        f"{anchored}/{approved} answers placed in their page "
        f"({anchored / approved if approved else 0:.1%})",
        "One model pass serves every variant; labelling is arithmetic from here.",
        "",
        "=== chunk variants ===",
        f"{'variant':10s} {'chunks':>7s} {'labelled':>9s} {'links':>7s} "
        f"{'coverage':>9s} {'q/10k ch':>9s}",
    ]
    for row in rows:
        coverage = row["questions"] / approved if approved else 0.0
        store_chars = row["store_chars"] or 0
        # Questions per 10,000 indexed characters: the number the density design
        # claims is the same for every variant. Printed rather than asserted
        # because it is also how a half-finished generation run shows up — a cut
        # still being generated reads as thinly probed, not as differently cut.
        density = 10_000 * row["questions"] / store_chars if store_chars else 0.0
        lines.append(
            f"{row['variant']:10s} {row['chunks']:>7d} {row['questions']:>9d} "
            f"{row['labels']:>7d} {coverage:>8.1%} {density:>9.2f}"
        )
    lines.append(f"\napproved questions: {approved}")
    lines.append(
        "q/10k ch is questions per 10,000 indexed characters. Under the "
        "density-normalised design (`generate_dataset.py --density`) every "
        "variant should read about the same; a column four times another's is "
        "the legacy one-question-per-type design, where the smallest cut is "
        "probed most densely."
    )

    ready = [row["variant"] for row in rows if row["questions"] == approved]
    waiting = [row["variant"] for row in rows if row["questions"] < approved]
    if waiting:
        lines.append(f"still labelling: {', '.join(waiting)}")
    lines.append(
        f"comparable now: {', '.join(ready) if ready else 'none'}"
        + ("" if len(ready) > 1 else "  (need at least two to compare)")
    )

    lines.extend(["", "=== chroma collections ===", *_collections()])
    return "\n".join(lines)


def _count(conn, table: str) -> int:
    """Row count, or 0 when the table has not been created yet."""
    exists = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?", (table,)
    ).fetchone()
    if not exists:
        return 0
    return int(conn.execute(f"select count(*) from {table}").fetchone()[0])


def _collections() -> list[str]:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path()))
        collections = client.list_collections()
    except Exception as error:
        return [f"  unavailable: {error}"]
    if not collections:
        return ["  none built yet"]
    return [
        f"  {collection.name:44s} {collection.count():>6d}"
        for collection in sorted(collections, key=lambda item: item.name)
    ]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(format_status())


if __name__ == "__main__":
    main()
