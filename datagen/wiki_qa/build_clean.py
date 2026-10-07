"""Temporary: the cleaned question dataset as a jsonl export, built from the tables without changing them. Per item:
the verified repair if its relabel is `keep`, else the original; each row carries its provenance and labels, so a
consumer can filter (e.g. verdict == keep). Nothing is dropped here: unlabelled pages keep their originals with
labels null. Also writes wiki_qa_challenge.jsonl (challenging items with labels, BM25/dense ranks, challenge_ok,
likely_ambiguous) and wiki_qa_rag.jsonl (rag_rows: answers, evidence, cross-lingual, criteria, ranks).
Official output: zstd Parquet (wiki_qa_*.parquet; nested dicts as JSON strings); the JSONL it is converted from is
moved to legacy_jsonl/ (human-readable copy, not for use). All in wiki_qa_dataset/ (OUT).
Usage: uv run --with bm25s --with pyarrow python build_clean.py [out=wiki_qa_clean.jsonl]"""

import collections
import glob
import json
import math
import os
import re
import sqlite3
import sys
import zlib

con = sqlite3.connect("wiki_qa.db")
OUT = "wiki_qa_dataset"  # the release folder: Parquet, README.md (guide), manifest, SHA256SUMS, legacy_jsonl/


def table(t):
    if not con.execute(
        "select count(*) from sqlite_master where name = ?", (t,)
    ).fetchone()[0]:
        return {}
    return {
        i: json.loads(v)
        for i, v in con.execute(f"select id, items from {t} where items is not null")
    }


def extra(t):
    """A gen_extra.py table: row id -> items (rows marked for redo carry an error and are skipped)."""
    return {
        i: json.loads(v)
        for i, v in con.execute(
            f"select id, items from {t} where items is not null and error is null"
        )
    }


_qrels = {}
QREL_KEYS = (
    "gold_match",
    "gold_answers",
    "relevant",
    "partial",
    "answering",
    "hard_negatives",
    "n_relevant",
    "hits",
    "pool_saturated",
)


def qrel_fields(pid, q):
    """Pooled relevance of question q (gen_extra.py qrels: BM25 top 20 + gold, each judged by Ling from an excerpt).
    relevant/partial = other pages that match the question fully/partly (beyond the gold page: they make it
    ambiguous); answering = every page whose excerpt answers it; hard_negatives = BM25-ranked non-matching pages
    (best first, max 10); gold_match = the judge on the gold page (a control: should be yes)."""
    if not _qrels:
        _qrels.update(extra("qrels") or {"": None})
    r = _qrels.get(f"{pid}|{zlib.crc32(q.encode()):08x}")
    if not r:  # every field present (null) so the schema is the same on every row (Parquet, pandas)
        return {"qrels_judged": False} | dict.fromkeys(QREL_KEYS)
    other = [c for c in r["cands"] if c["id"] != pid]
    gold = next(c for c in r["cands"] if c["id"] == pid)
    return {
        "qrels_judged": True,
        "gold_match": gold["match"],
        "gold_answers": gold["answers"],
        "relevant": [c["id"] for c in other if c["match"] == "yes"],
        "partial": [c["id"] for c in other if c["match"] == "partly"],
        "answering": [c["id"] for c in r["cands"] if c["answers"]],
        "hard_negatives": [
            c["id"]
            for c in sorted(other, key=lambda c: c["bm25_rank"] or 99)
            if c["match"] == "no" and c["bm25_rank"]
        ][:10],
    } | hits(r, pid)


def hits(r, pid):
    """How many pages the question fits: n_relevant = gold + other full matches; hits unique (1), few (2-3), many
    (4+); pool_saturated = >= 10 of the ~20 pooled pages match fully or partly, so the true count is likely higher
    than judged (the pool is only BM25 top 20). A question with many hits cannot test single-page retrieval."""
    n = 1 + sum(c["match"] == "yes" for c in r["cands"] if c["id"] != pid)
    loose = sum(c["match"] != "no" for c in r["cands"] if c["id"] != pid)
    return {
        "n_relevant": n,
        "hits": "unique" if n == 1 else "few" if n <= 3 else "many",
        "pool_saturated": loose >= 10,
    }


def split(qid):
    """dev (20%) / test (80%) by a hash of the Wikidata id: every question about a place falls in one split."""
    return "dev" if zlib.crc32(qid.encode()) % 5 == 0 else "test"


def template_sim():
    """page id -> max cosine of its lead embedding (page_emb.npz, title + first 1000 chars) to any other page: high
    = a templated page (bot stubs differ only in names and numbers). CPU, cached in page_sim.npz."""
    import numpy as np

    if not os.path.exists("page_sim.npz"):
        d = np.load("page_emb.npz")
        e = d["emb"].astype(np.float32)
        best = np.zeros(len(e), np.float32)
        for s in range(0, len(e), 4096):
            sim = e[s : s + 4096] @ e.T
            sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -1
            best[s : s + 4096] = sim.max(1)
        np.savez("page_sim.npz", ids=d["ids"], best=best)
    d = np.load("page_sim.npz")
    return dict(zip(d["ids"].tolist(), d["best"].round(3).tolist()))


def lex_overlap(q, text, words):
    """Share of the question's idf weight whose words (first 5 letters: a crude stemmer) occur in the page: ~1 =
    keyword search has everything it needs, ~0 = only semantics can find the page. words: the page's prefixes."""
    import bm25

    N = len(bm25._ids)
    w = {t[:5]: math.log(N / bm25._df.get(t, 1)) for t in bm25.tok(q)}
    total = sum(w.values())
    return (
        round(sum(v for t, v in w.items() if t in words) / total, 3) if total else None
    )


def norm(q):
    return re.sub(r"\W+", " ", q.lower()).strip()


def challenge_ok(lb):
    """Quality filter for a challenging item: the qspec labels minus `duplicate` and the verdict (they count the
    five questions of a page as duplicates: all point to the same place on purpose) and minus `realistic` (the
    labeller calls the wanted "which lake has the property of ..." questions trivia; Sonnet found 87% natural)."""
    return (
        bool(lb)
        and lb["facts_supported"] == "yes"
        and not lb["self_answering"]
        and all(
            lb[k]
            for k in (
                "language_ok",
                "fluent",
                "unique_place",
                "facts_answer",
                "translation_ok",
                "query_ok",
            )
        )
    )


def likely_ambiguous(it, dense):
    """Neither BM25 nor dense retrieval finds the page (both ranks > 10): 63% of those were ambiguous to Sonnet
    (46/73), against 16-38% otherwise. dense = dense_rank.json entry [rank, margin] or None (not ranked yet)."""
    return it["bm25_rank"] > 10 and dense is not None and dense[0] > 10


def near_dups(threshold=0.98):
    """Keys of corpus questions whose nearest question on another page has cosine >= threshold (near_dups.py);
    0.98 = near-identical text (lower bands are the same template about differently named places)."""
    import os

    import numpy as np

    if not os.path.exists("near_dups.npz"):
        return set()
    d = np.load("near_dups.npz")
    return {
        k
        for k, v in zip(d["keys"], d["best_sim"])
        if v >= threshold and k.startswith("q|")
    }


def rows():
    """Yield one row per generated item, with the verified repair substituted where it passed. Flag
    `ambiguous_across_pages`: the same (or near-identical, embedding cosine >= 0.98) question is asked on another page (e.g. 19 sv lakes named Långtjärnen
    in one parish), so it cannot single out its page; flagged, not removed."""
    name = dict(con.execute("select model, name from models"))
    labels, repairs, relabels = (
        table("bunny_labels"),
        table("repairs"),
        table("repair_labels"),
    )
    rows_ = []
    for pid, model, items in con.execute(
        "select id, model, items from questions where error is null"
    ):
        items, labs = json.loads(items), labels.get(pid)
        if labs and len(labs) != len(items):
            labs = None
        rep = {r["j"]: r for r in repairs.get(pid, [])}
        relab = relabels.get(pid)
        if relab and len(relab) != len(
            items
        ):  # a relabel with the wrong item count verifies nothing
            relab = None
        for j, it in enumerate(items):
            row = {
                "id": pid,
                "j": j,
                "source": name.get(model, model),
                "item": it,
                "labels": labs[j] if labs else None,
            }
            if j in rep and relab and relab[j]["verdict"] == "keep":
                row |= {
                    "item": rep[j]["item"],
                    "labels": relab[j],
                    "repaired": rep[j]["action"],
                    "original": it,
                }
            rows_.append(row)
    near = near_dups()
    seen = collections.defaultdict(set)
    for row in rows_:
        seen[norm(row["item"]["question"])].add(row["id"])
    for row in rows_:
        row["ambiguous_across_pages"] = (
            len(seen[norm(row["item"]["question"])]) > 1
            or f"q|{row['id']}|{row['j']}" in near
        )
        yield row


def challenge_rows():
    """One row per challenging item (Bunny): labels, BM25 rank, dense rank, and the two quality flags."""
    dense = (
        json.load(open("dense_rank.json")) if os.path.exists("dense_rank.json") else {}
    )
    for pid, its in con.execute(
        "select id, items from challenge_items where items is not null"
    ):
        for j, it in enumerate(json.loads(its)):
            d = dense.get(f"{pid}|{j}")
            yield {
                "id": pid,
                "j": j,
                "item": {
                    k: it[k]
                    for k in ("question", "question_en", "answer", "facts", "query")
                },
                "labels": it.get("labels"),
                "prompt": it.get("prompt", "v1"),
                "bm25_rank": it["bm25_rank"],
                "dense_rank": d[0] if d else None,
                "challenge_ok": challenge_ok(it.get("labels")),
                "question_hard": it.get(
                    "question_hard"
                ),  # rejected BM25-hardening rewrite (log 14:40)
                "likely_ambiguous": likely_ambiguous(it, d),
            }


def rag_rows(src_labels):
    """src_labels: (page id, kind, src_j) -> (current question, qspec labels). One row per current item of table answers (gen_rag.py): page metadata, question, answer, evidence spans (recomputed,
    markup-tolerant), qtype, the item's qspec labels, the RAG criteria (true only if every judge that judged it says
    so; `judges` names them), answer_ok (all criteria but x_ok), the cross-lingual version with x_ok, and per
    question (o = original, x = cross-lingual) [dense rank, dense margin, BM25 rank] of its own page."""
    from gen_rag import spans

    judges = [
        (t, table(t)) for t in ("answer_labels", "answer_labels_ling") if table(t)
    ]
    import bm25

    bm25.load()
    rank = json.load(open("rag_rank.json")) if os.path.exists("rag_rank.json") else {}
    answers = table("answers")
    tsim = template_sim()
    ptags = table(
        "labels"
    )  # 8 page tags (tag_spec.py), Gemma E4B LoRA classifier, all pages
    var = {(pid, v["question"]): v for pid, vs in extra("variants").items() for v in vs}
    for line in open("/data/llms4eu/wiki/pages.jsonl"):
        p = json.loads(line)
        if p["id"] in answers:
            words = {t[:5] for t in bm25.tok(p["title"] + " " + p["text"])}
            title = norm(re.sub(r"\s*\(.*\)$", "", p["title"]))
        for n, it in enumerate(answers.get(p["id"], [])):
            q, src = src_labels.get((p["id"], it["kind"], it["src_j"]), (None, None))
            if q != it["question"]:
                continue  # stale: the source item was repaired or hardened since; gen_rag.py redoes the page
            labs = [
                lab[p["id"]][n]
                for _, lab in judges
                if len(lab.get(p["id"], [])) == len(answers[p["id"]])
            ]
            crit = {k: all(lb[k] for lb in labs) for k in labs[0]} if labs else None
            sp, ok = spans(p["text"], it["evidence"])
            v = var.get((p["id"], it["question"]), {})
            yield {
                "id": p["id"],
                "n": n,
                "kind": it["kind"],
                "lang": p["in_language"],
                **{
                    k: p[k]
                    for k in (
                        "wikidata_id",
                        "url",
                        "title",
                        "country",
                        "categories",
                        "machine_generated",
                        "sitelinks",
                        "latitude",
                        "longitude",
                        "char_count",
                    )
                },
                "question": it["question"],
                "answer": it["answer"],
                "qtype": it["qtype"],
                "evidence": it["evidence"],
                "spans": sp,
                "verbatim": ok,
                "evidence_pos": round(sp[0][0] / len(p["text"]), 3) if sp else None,
                # some evidence in a Markdown table row (infobox): chunkers often split or drop these
                "evidence_in_table": any(
                    p["text"][p["text"].rfind("\n", 0, a) + 1 :].startswith("|")
                    for a, _ in sp
                ),
                "labels": src,
                "criteria": crit,
                "judges": len(labs),
                "answer_ok": crit and all(v for k, v in crit.items() if k != "x_ok"),
                "x_lang": it["x_lang"],
                "question_x": it["question_x"],
                "answer_x": it["answer_x"],
                # translations need both judges: Ling alone passes 95% vs 79% for both (blind Sonnet sided with neither)
                "x_ok": crit["x_ok"] if len(labs) == len(judges) == 2 else None,
                "rank_o": rank.get(f"{p['id']}|{n}|o"),
                "rank_x": rank.get(f"{p['id']}|{n}|x"),
                "split": split(p["wikidata_id"]),
                "stub": p["char_count"] < 2000,
                "page_template_sim": tsim.get(p["id"]),
                "page_tags": ptags.get(p["id"]),
                "title_in_question": title in norm(it["question"]),
                "lex_overlap": lex_overlap(it["question"], p["text"], words),
                # Ling tags and query variants (gen_extra.py variants, balanced pages); *_ok = Ling's check
                "time_sensitive": v.get("time_sensitive"),
                "reasoning": v.get("reasoning"),
                "variants": {
                    k: v[k]
                    for k in ("keyword", "typo", "verbose", "history", "followup")
                }
                | {
                    "check": v["check"],
                    # a question asking which place it is cannot be a follow-up: the history names the answer
                    "followup_applicable": it["qtype"] != "identify",
                }
                if v
                else None,
                **qrel_fields(p["id"], it["question"]),
            }


def write_extra(rag):
    """The gen_extra.py layers as their own files (wiki_qa_dataset/README.md), plus manifest.json: counts per file and slice."""
    pages = {}
    un, cmp, meta, tq = (
        extra("unanswerable"),
        extra("compare"),
        extra("meta_q"),
        extra("table_q"),
    )  # read once: Ling keeps writing
    need = (
        set(un)
        | set(cmp)
        | {t.split("|")[0] for t in tq}
        | {s["anchor"] for v in meta.values() for s in v if s.get("anchor")}
    )
    for line in open("/data/llms4eu/wiki/pages.jsonl"):
        p = json.loads(line)
        if p["id"] in need:
            pages[p["id"]] = {
                k: p[k] for k in ("wikidata_id", "title", "in_language", "country")
            }
    files = collections.defaultdict(list)
    for pid, its in un.items():
        for it in its:
            q = qrel_fields(pid, it["question"])
            files["unanswerable"].append(
                {
                    "id": pid,
                    "lang": pages[pid]["in_language"],
                    "title": pages[pid]["title"],
                }
                | it
                | q
                | {
                    # the article check passed and no pooled page answers it (unjudged pools: not ok yet)
                    "ok": all(it["check"].values())
                    and q["qrels_judged"]
                    and not q["answering"],
                    "split": split(pages[pid]["wikidata_id"]),
                }
            )
    for pid, its in cmp.items():
        for it in its:
            files["compare"].append(
                {"id": pid, "lang": pages[pid]["in_language"]}
                | it
                | {
                    "ok": all(it["check"].values()) and it["verbatim"],
                    "split": split(pages[pid]["wikidata_id"]),
                }
            )
    for its in meta.values():
        for it in its:
            files["meta"].append(
                it
                | {
                    "ok": all(it["check"].values()),
                    "n_gold": len(
                        it["gold"]
                    ),  # places to find: use recall@k with k >= n_gold
                    "split": split(pages[it["anchor"]]["wikidata_id"])
                    if it.get("anchor")
                    else split(it["key"]),
                }
            )
    # table questions: answers computed by code (gen_extra.compute), 200 verified by Sonnet (xcheck/extra_tables_*)
    verdict = {
        v["key"]: v
        for f in glob.glob("xcheck/extra_tables_*_sonnet.json")
        for v in json.load(open(f))
    }
    for tid, its in tq.items():
        pid = tid.split("|")[0]
        for n, it in enumerate(its):
            v = verdict.get(f"{tid}|{n}")
            files["tables"].append(
                {"id": pid, "table": int(tid.split("|")[1]), "n": n}
                | {k: pages[pid][k] for k in ("title", "in_language")}
                | it
                | {
                    # Sonnet checked 200 (max, user): answer_final = computed if correct, else Sonnet's own answer; unchecked
                    # rows are computed only (~14% wrong: shifted columns, decimal commas, joined values) -> not ok
                    "sonnet_checked": v is not None,
                    "sonnet": v
                    and {
                        k: v[k]
                        for k in (
                            "answer_correct",
                            "question_ok",
                            "your_answer",
                            "note",
                        )
                    },
                    "answer_final": None
                    if v is None
                    else it["answer"]
                    if v["answer_correct"]
                    else v["your_answer"],
                    "answer_source": "unverified"
                    if v is None
                    else "computed"
                    if v["answer_correct"]
                    else "sonnet",
                    "ok": bool(v and v["question_ok"]),
                    "split": split(pages[pid]["wikidata_id"]),
                }
            )
    man = {"built": __import__("datetime").datetime.now().isoformat(timespec="minutes")}
    for name, rs in files.items():
        with open(f"wiki_qa_{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man[f"wiki_qa_{name}.parquet"] = count(
            rs, ("type", "kind", "qtype", "op", "answer_source")
        )
    man["wiki_qa_rag.parquet"] = count(rag, ("kind",), ok="answer_ok")
    os.makedirs(OUT, exist_ok=True)
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(json.dumps(man)[:3000])


def write_parquet(path):
    """path (.jsonl) -> the same rows as zstd Parquet next to it. One schema for all rows: the union of the keys
    (absent = null); a column holding dicts or mixed types (labels, variants, criteria, ...) is stored as JSON
    strings (json.loads to read). Needs pyarrow (uv run --with pyarrow)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [json.loads(line) for line in open(path)]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for k in keys:
        types = {type(r.get(k)) for r in rows} - {type(None)}
        if dict in types or len(types - {int, float}) + bool(types & {int, float}) > 1:
            for r in rows:
                if r.get(k) is not None:
                    r[k] = json.dumps(r[k], ensure_ascii=False)
    out = f"{OUT}/" + path.replace(".jsonl", ".parquet")
    # to a temp name, then rename: an interrupted build never leaves a truncated official file
    pq.write_table(
        pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows]),
        out + ".tmp",
        compression="zstd",
    )
    os.replace(out + ".tmp", out)


def count(rs, keys, ok="ok"):
    """rows, ok rows, and ok rows per split, language and each key present."""
    out = {"rows": len(rs), "ok": sum(bool(r.get(ok)) for r in rs)}
    for k in ("split", "lang") + keys:
        c = collections.Counter(r[k] for r in rs if r.get(ok) and k in r)
        if c:
            out[f"ok_by_{k}"] = dict(c.most_common())
    return out


def sums():
    """OUT/SHA256SUMS of the release files (check with: cd wiki_qa_dataset && sha256sum -c SHA256SUMS)."""
    import hashlib

    lines = []
    for f in sorted(glob.glob(f"{OUT}/*.parquet") + [f"{OUT}/manifest.json"]):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {os.path.basename(f)}")
    open(f"{OUT}/SHA256SUMS", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    out = open(sys.argv[1] if len(sys.argv) > 1 else "wiki_qa_clean.jsonl", "w")
    stats = collections.Counter()
    src_labels = {}
    for row in rows():
        src_labels[(row["id"], "corpus", row["j"])] = (
            row["item"]["question"],
            row["labels"],
        )
        verdict = row["labels"]["verdict"] if row["labels"] else "unlabelled"
        stats[(row["source"], row.get("repaired", "orig"), verdict)] += 1
        stats[("ambiguous_across_pages", row["ambiguous_across_pages"])] += 1
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for k, v in sorted(stats.items()):
        print(*k, v)
    cs = collections.Counter()
    with open("wiki_qa_challenge.jsonl", "w") as f:
        for r in challenge_rows():
            src_labels[(r["id"], "challenge", r["j"])] = (
                r["item"]["question"],
                r["labels"],
            )
            cs[(r["prompt"], r["challenge_ok"], r["likely_ambiguous"])] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        "challenge (prompt, challenge_ok, likely_ambiguous):", dict(sorted(cs.items()))
    )
    rs, rag = collections.Counter(), list(rag_rows(src_labels))
    # balanced: a language-balanced page subset (the first 400 pages per language by a hash of the id, stable as the
    # data grows), so sv lake stubs (a third of the items) do not dominate a test
    by_lang = collections.defaultdict(set)
    for r in rag:
        by_lang[r["lang"]].add(r["id"])
    balanced = {
        i
        for ids in by_lang.values()
        for i in sorted(ids, key=lambda i: zlib.crc32(i.encode()))[:400]
    }
    with open("wiki_qa_rag.jsonl", "w") as f:
        for r in rag:
            r["balanced"] = r["id"] in balanced
            rs[(r["kind"], r["judges"], r["answer_ok"], r["x_ok"])] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rag (kind, judges, answer_ok, x_ok):", dict(sorted(rs.items())))
    write_extra(rag)
    out.close()
    del rag
    # official format: zstd Parquet (6.5x smaller, column reads ~100x faster); the JSONL goes to legacy_jsonl/
    os.makedirs(f"{OUT}/legacy_jsonl", exist_ok=True)
    for path in sorted(glob.glob("wiki_qa_*.jsonl")):
        write_parquet(path)
        os.replace(path, f"{OUT}/legacy_jsonl/{path}")
        print("parquet", path.replace(".jsonl", ".parquet"))
    sums()
