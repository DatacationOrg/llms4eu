"""Temporary: LoRA training sets for Qwen3.5-4B from the Bunny tables, as jsonl rows {system, user, target, split, lang}
(split: 5% of pages held out by a hash of the id, so it is stable as tables grow).
  sft_qg.jsonl         gen_questions prompt -> the page's final items, pages whose items are all keep after repair
                       (at most 300 train pages per language)
  sft_fix.jsonl        bunny_fix prompt -> repairs whose relabel is keep
  sft_cls.jsonl        bunny_label prompt (article cut at 6000 chars) -> Bunny's 13 labels per item; train pages
                       balanced: all with a fix/drop item + as many all-keep pages
  sft_challenge.jsonl  gen_challenge prompt -> the page's v2 challenging items that pass challenge_ok (pages with at
                       least one passing item with bm25_rank > 1)
fix and challenge only from pages <= 8000 chars (their prompts carry the whole article). Prints per-language counts. Usage: build_sft.py"""

import collections
import json
import os
import zlib

import bunny
import bunny_label
import build_clean
from build_clean import challenge_ok, likely_ambiguous
import bunny_fix
import gen_challenge
import gen_questions


def split(pid):
    return "val" if zlib.crc32(pid.encode()) % 20 == 0 else "train"


def write(name, rows, cap=None):
    """cap: at most this many train rows per language (stable choice by id hash), so sv lakes don't dominate."""
    if cap:
        rows.sort(key=lambda r: zlib.crc32(r["user"].encode()))
        seen = collections.Counter()
        rows = [
            r
            for r in rows
            if r["split"] == "val"
            or (seen.update([r["lang"]]) or seen[r["lang"]] <= cap)
        ]
    cnt = collections.Counter()
    with open(name, "w") as f:
        for r in rows:
            cnt[(r["lang"], r["split"])] += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr = sum(v for (_, s), v in cnt.items() if s == "train")
    print(name, "train", tr, "val", sum(cnt.values()) - tr)
    print(
        "  ", dict(sorted(collections.Counter(lg for lg, _ in cnt.elements()).items()))
    )


def main():
    by_page = collections.defaultdict(list)
    for row in build_clean.rows():
        by_page[row["id"]].append(row)
    good = {
        pid: rs
        for pid, rs in by_page.items()
        if all(r["labels"] and r["labels"]["verdict"] == "keep" for r in rs)
    }
    labels = build_clean.table("bunny_labels")
    repairs = build_clean.table("repairs")
    relabels = build_clean.table("repair_labels")
    challenge = build_clean.table("challenge_items")
    want = set(good) | set(repairs) | set(challenge) | set(labels)
    dense = (
        json.load(open("dense_rank.json")) if os.path.exists("dense_rank.json") else {}
    )

    def challenge_keep(pid, j, it):
        return challenge_ok(it.get("labels")) and not likely_ambiguous(
            it, dense.get(f"{pid}|{j}")
        )

    qg, fix, ch, cls = [], [], [], []
    for p, items in bunny.pages(want):
        pid, lang = p["id"], p["in_language"]
        base = {"split": split(pid), "lang": lang}
        if pid in good:
            target = {"items": [r["item"] for r in good[pid]]}
            qg.append(
                base
                | {
                    "system": gen_questions.PROMPT,
                    "user": gen_questions.user_msg(p),
                    "target": json.dumps(target, ensure_ascii=False),
                }
            )
        labs = labels.get(pid)
        if labs and len(labs) == len(
            items
        ):  # classifier: article cut at 6000 chars, like the QG LoRA
            cls.append(
                base
                | {
                    "system": bunny_label.PROMPT,
                    "user": bunny_label.message(p | {"text": p["text"][:6000]}, items),
                    "target": json.dumps({"labels": labs}),
                    "bad": any(lb["verdict"] != "keep" for lb in labs),
                }
            )
        short = (
            len(p["text"]) <= 8000
        )  # fix/challenge prompts carry the full article: keep training steps small
        if (
            short
            and pid in repairs
            and len(relabels.get(pid) or []) == len(items)
            and len(labels.get(pid) or []) == len(items)
        ):
            ok = [r for r in repairs[pid] if relabels[pid][r["j"]]["verdict"] == "keep"]
            if ok and len(labels[pid]) == len(items):
                fix.append(
                    base
                    | {
                        "system": bunny_fix.PROMPT,
                        "user": bunny_fix.message(p, items, labels[pid]),
                        "target": json.dumps(
                            {"repairs": [{"j": r["j"], "item": r["item"]} for r in ok]},
                            ensure_ascii=False,
                        ),
                    }
                )
        its = challenge.get(pid, [])
        if (
            short
            and its
            and its[0].get("prompt") in ("v2", "v3")
            and any(
                it["bm25_rank"] > 1 and challenge_keep(pid, j, it)
                for j, it in enumerate(its)
            )
        ):
            its = [it for j, it in enumerate(its) if challenge_keep(pid, j, it)]
            keys = ("question", "question_en", "answer", "facts", "query")
            ch.append(
                base
                | {
                    "system": gen_challenge.PROMPT,
                    "user": gen_challenge.message(p),
                    "target": json.dumps(
                        {"items": [{k: it[k] for k in keys} for it in its]},
                        ensure_ascii=False,
                    ),
                }
            )
    write("sft_qg.jsonl", qg, cap=300)
    write("sft_fix.jsonl", fix)
    write("sft_challenge.jsonl", ch)
    # balanced: every train page with a fix/drop item, and as many all-keep train pages (stable choice)
    bad = [r for r in cls if r["bad"] or r["split"] == "val"]
    keep = sorted(
        (r for r in cls if not r["bad"] and r["split"] == "train"),
        key=lambda r: zlib.crc32(r["user"].encode()),
    )
    write("sft_cls.jsonl", bad + keep[: sum(r["split"] == "train" for r in bad)])


if __name__ == "__main__":
    main()
