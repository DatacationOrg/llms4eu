"""Temporary: label every qlabel/ item with Jev (typesafe/jev-1.13, OpenRouter decisions API) on the qspec questions.
State = qspec.state (article first 6000 chars, item JSON, the page's other questions). Resumable: appends to
qlabel_jev.jsonl rows {id, j, label, probs, usage}. Needs OPENROUTER_API_KEY. Usage: jev_qlabel.py [workers=10]"""

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from qspec import IDS, SPEC, laya_questions, state  # noqa: E402

OUT = f"{S}/qlabel_jev.jsonl"
KEY = os.environ["OPENROUTER_API_KEY"]
QS = laya_questions()
done = (
    {(r["id"], r["j"]) for r in map(json.loads, open(OUT))}
    if os.path.exists(OUT)
    else set()
)
todo = [
    (p, j)
    for d in sorted(glob.glob(f"{S}/qlabel/b_*"))
    for p in map(json.loads, open(f"{d}/batch.jsonl"))
    for j in range(len(p["items"]))
    if (p["id"], j) not in done
]


def run(task):
    p, j = task
    body = {"model": "typesafe/jev-1.13", "state": state(p, j), "questions": QS}
    for attempt in range(3):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/alpha/decisions",
                json=body,
                timeout=120,
                headers={"Authorization": f"Bearer {KEY}"},
            )
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 10)))
                continue
            r.raise_for_status()
            a = r.json()["answers"]
            label = {
                k: a[k]["noul"] >= 0.5 if SPEC[k][0] == "bool" else a[k]["choice"]
                for k in IDS
            }
            return {
                "id": p["id"],
                "j": j,
                "label": label,
                "probs": a,
                "usage": r.json().get("usage"),
            }
        except Exception as e:
            print("retry", p["id"], j, type(e).__name__, str(e)[:200], flush=True)
            time.sleep(3)
    return None


print(len(todo), "items to label", flush=True)
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
with ThreadPoolExecutor(workers) as ex, open(OUT, "a") as f:
    for n, res in enumerate(ex.map(run, todo), 1):
        if res:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
        if n % 100 == 0:
            print(n, "done", flush=True)
print("finished", flush=True)
