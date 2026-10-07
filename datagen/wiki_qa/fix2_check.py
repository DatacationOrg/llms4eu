"""Temporary: validate an agent's fixed.jsonl against its batch.jsonl (same urls, schema, limits).
Usage: fix2_check.py fix2/<dir>   (prints problems, exit 1 if any; `all` adds every valid dir to qg_clean.jsonl, replacing rows with the same url)"""

import glob
import json
import os
import sys

S = os.path.dirname(os.path.abspath(__file__))
KEYS = ["question", "question_en", "facts", "query"]


def problems(d):
    batch = [json.loads(line)["url"] for line in open(f"{d}/batch.jsonl")]
    if not os.path.exists(f"{d}/fixed.jsonl"):
        return ["no fixed.jsonl"]
    out, errs = {}, []
    for n, line in enumerate(open(f"{d}/fixed.jsonl"), 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"line {n}: bad json {e}")
            continue
        out[r.get("url")] = r
        its = r.get("items")
        if not isinstance(its, list) or len(its) > 5:
            errs.append(f"line {n}: items must be a list of 0-5")
            continue
        for j, it in enumerate(its):
            if list(it) != KEYS:
                errs.append(
                    f"line {n} item {j}: keys must be {KEYS} in order, got {list(it)}"
                )
            elif not all(
                isinstance(it[k], str) and it[k].strip()
                for k in ("question", "question_en", "query")
            ):
                errs.append(f"line {n} item {j}: empty or non-string field")
            elif not (
                isinstance(it["facts"], list)
                and 1 <= len(it["facts"]) <= 3
                and all(isinstance(f, str) and f.strip() for f in it["facts"])
            ):
                errs.append(f"line {n} item {j}: facts must be 1-3 non-empty strings")
    missing = [u for u in batch if u not in out]
    extra = [u for u in out if u not in batch]
    errs += [f"missing url {u}" for u in missing] + [
        f"unexpected url {u}" for u in extra
    ]
    return errs


if __name__ == "__main__":
    if sys.argv[1] == "all":
        path = f"{S}/qg_clean.jsonl"
        rows = (
            {r["url"]: r for r in map(json.loads, open(path))}
            if os.path.exists(path)
            else {}
        )
        for d in sorted(glob.glob(f"{S}/fix2/*/")):
            if problems(d):
                print("skip", d, problems(d)[:3])
                continue
            for r in map(json.loads, open(f"{d}/fixed.jsonl")):
                rows[r["url"]] = {"url": r["url"], "items": r["items"]}
        with open(path, "w") as f:
            f.writelines(
                json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()
            )
        print(sum(1 for _ in open(f"{S}/qg_clean.jsonl")), "articles in qg_clean.jsonl")
    else:
        errs = problems(sys.argv[1])
        print("\n".join(errs[:50]) or "OK")
        sys.exit(bool(errs))
