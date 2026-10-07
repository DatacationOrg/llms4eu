"""Temporary: validate a balanced-set agent folder synth2/<dir>: variants.jsonl must hold one row per target in
targets.json (same order): {target_label, target_class, j, item, labels}; item has the 4 fields, labels has all 13
qspec labels, the target label has the target class, and the labels are consistent (qspec.consistency).
Usage: synth_check.py synth2/<dir>"""

import json
import sys

from qspec import IDS, consistency, options

KEYS = ["question", "question_en", "facts", "query"]


def norm(v):
    return str(v).lower() if isinstance(v, bool) else v


def problems(d):
    page = json.load(open(f"{d}/page.json"))
    targets = json.load(open(f"{d}/targets.json"))
    try:
        rows = [json.loads(line) for line in open(f"{d}/variants.jsonl")]
    except (OSError, json.JSONDecodeError) as e:
        return [f"variants.jsonl: {e}"]
    if len(rows) != len(targets):
        return [f"need {len(targets)} rows, one per target in order, got {len(rows)}"]
    errs = []
    for n, (r, t) in enumerate(zip(rows, targets)):
        if (r.get("target_label"), norm(r.get("target_class"))) != (
            t["label"],
            t["class"],
        ):
            errs.append(f"row {n}: target must be {t}")
        if not isinstance(r.get("j"), int) or not 0 <= r["j"] < len(page["items"]):
            errs.append(
                f"row {n}: j must be the index of the page item you started from"
            )
        it, lab = r.get("item"), r.get("labels")
        if (
            not isinstance(it, dict)
            or list(it) != KEYS
            or not (isinstance(it["facts"], list) and 1 <= len(it["facts"]) <= 3)
        ):
            errs.append(f"row {n}: item must have keys {KEYS} and 1-3 facts")
            continue
        bad = [
            k for k in IDS if not isinstance(lab, dict) or lab.get(k) not in options(k)
        ]
        if bad:
            errs.append(f"row {n}: invalid or missing labels {bad}")
            continue
        if norm(lab[t["label"]]) != t["class"]:
            errs.append(
                f"row {n}: labels[{t['label']}] must be {t['class']} (the target)"
            )
        errs += [f"row {n}: {e}" for e in consistency(page["id"], [it], [lab])]
    return errs


if __name__ == "__main__":
    errs = problems(sys.argv[1])
    print("\n".join(errs[:30]) or "OK")
    sys.exit(bool(errs))
