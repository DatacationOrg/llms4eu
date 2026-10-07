"""Temporary: rank-8 LoRA on laya-multilingual for the qspec question-quality labels.
State per item: article (first 6000 chars), the item JSON, the page's other questions.
Train: bulk Sonnet labels on the 500 train pages (qlabel/), one-article labels (qlabel_one/) where present; or, with
QL_BUNNY=1, the Space Bunny teacher labels (bunny_labels). QL_LABELS=a,b trains only those labels.
Val gold: one-article Sonnet labeller A (qlabel2/val_a_*). Reports per-label accuracy and macro-F1 vs gold for:
majority class, bulk Sonnet labels, a second one-article labeller B (its 89 pages), Jev, Laya zero-shot, Laya + LoRA.
Writes laya_quality_r8.pt and laya_quality_eval.json. Usage: train_laya_quality.py [epochs=5] [max_len=2048] [lr=1e-4]"""

import glob
import json
import os
import random
import sys
import time
from collections import Counter

import laya
import numpy as np
import torch
from laya.common import collate_items
from peft import LoraConfig, inject_adapter_in_model

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from qaugment import augment, load_balanced, load_synth  # noqa: E402
from qspec import IDS, laya_questions, options, state, target  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAXLEN = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
ACC = 8  # effective batch size 8 (one item per forward)
QPS = int(os.environ.get("QL_QPS", 4))  # labels trained per item per step (of 13)


def load(pattern):
    rows = []
    for d in sorted(glob.glob(f"{S}/{pattern}")):
        if not os.path.exists(f"{d}/labels.jsonl"):
            continue
        pages = {r["id"]: r for r in map(json.loads, open(f"{d}/batch.jsonl"))}
        for lab in map(json.loads, open(f"{d}/labels.jsonl")):
            p = pages[lab["id"]]
            rows += [
                {"page": p, "j": j, "label": lb, "split": p["split"]}
                for j, lb in enumerate(lab["labels"])
            ]
    return rows


def load_bunny():
    """Space Bunny teacher labels (table bunny_labels), every labelled page except the gold val pages."""
    import sqlite3

    import bunny

    con = sqlite3.connect(f"{S}/wiki_qa.db")
    labs = {
        i: json.loads(v)
        for i, v in con.execute(
            "select id, items from bunny_labels where items is not null"
        )
    }
    val_ids = {r["page"]["id"] for r in load("qlabel2/val_a_*")}
    rows = []
    for p, items in bunny.pages(set(labs) - val_ids):
        if len(items) == len(labs[p["id"]]):
            page = p | {"items": items}
            rows += [
                {"page": page, "j": j, "label": lb, "split": "train"}
                for j, lb in enumerate(labs[p["id"]])
            ]
    return rows


one = {(r["page"]["id"], r["j"]): r for r in load("qlabel_one/p_*")}
if os.environ.get("QL_BUNNY") == "1":
    train = load_bunny()
else:
    train = [
        one.get((r["page"]["id"], r["j"]), r)
        for r in load("qlabel/b_*")
        if r["split"] == "train"
    ]
if os.environ.get(
    "QL_PER_CLASS"
):  # class-balanced subsample by verdict, so keep doesn't swamp fix/drop
    _rng, _per = random.Random(3), int(os.environ["QL_PER_CLASS"])
    _by = {}
    for r in train:
        _by.setdefault(r["label"]["verdict"], []).append(r)
    train = [r for v in _by.values() for r in _rng.sample(v, min(_per, len(v)))]
    print(
        "balanced by verdict:",
        {k: min(_per, len(v)) for k, v in _by.items()},
        flush=True,
    )
# labels to train (all 13 by default); e.g. QL_LABELS=verdict,answer_difficulty for the not-overly-biased ones
TRAIN_IDS = os.environ.get("QL_LABELS", ",".join(IDS)).split(",")
val = load("qlabel2/val_a_*")
n_real = len(train)
if (
    os.environ.get("QL_BALANCED") == "1"
):  # agent-written balanced set (>= 20 per label class)
    train += load_balanced(f"{S}/synth2/a_*")
synth = f"{S}/qlabel_synth.jsonl"
if os.environ.get("QL_SYNTH") == "1" and os.path.exists(
    synth
):  # LLM-written bad variants, balanced per problem
    train += load_synth(train, synth)
if os.environ.get("QL_AUG", "1") != "0":
    train += augment(
        train, random.Random(1), per_item=int(os.environ.get("QL_AUG", 1))
    )  # synthetic bad items
# inverse-frequency class weights per label (on the augmented train set), capped so rare classes don't explode
_cnt = {k: Counter(target(r["label"])[j] for r in train) for j, k in enumerate(IDS)}
CW = {
    k: torch.tensor(
        [
            min(10.0, len(train) / (len(options(k)) * max(_cnt[k][c], 1)))
            for c in range(len(options(k)))
        ]
    )
    for k in IDS
}
by_key = lambda rows: {(r["page"]["id"], r["j"]): r["label"] for r in rows}  # noqa: E731
others = {
    "bulk_sonnet": by_key(r for r in load("qlabel/b_*") if r["split"] == "val"),
    "sonnet_one_article_B": by_key(load("qlabel2/val_b_*")),
    "jev": {
        (r["id"], r["j"]): r["label"]
        for r in map(json.loads, open(f"{S}/qlabel_jev.jsonl"))
    },
}
print(len(train), f"train items ({n_real} real),", len(val), "val items", flush=True)

agent = laya.load("convaiinnovations/laya-multilingual")
model = agent.model
QS = laya_questions()
internal = {q: agent._to_internal(QS[q]) for q in IDS}


def batch(r, ids=IDS):
    items = agent._encode_state(state(r["page"], r["j"]), ids, internal, max_len=MAXLEN)
    b = collate_items([items], agent.tok.pad_token_id)
    return {
        k: (v.to(agent.device) if torch.is_tensor(v) else v) for k, v in b.items()
    }, items


def logits_of(r, ids=IDS):
    b, items = batch(r, ids)
    logits, _ = model(
        b["input_ids"],
        b["attention_mask"],
        b["marker_pos"],
        b["marker_mask"],
        b["qtype"],
    )
    return logits, items


def predict(rs):
    model.eval()
    out, t0 = [], time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in rs:
            logits, items = logits_of(r)
            out.append(
                [
                    int(logits[j, : len(items[j]["markers"])].float().argmax())
                    for j in range(len(IDS))
                ]
            )
    return np.array(out), (time.perf_counter() - t0) / len(rs)


def scores(pred, rs):
    gold = np.array([target(r["label"]) for r in rs])
    res = {}
    for j, k in enumerate(IDS):
        f1s = []
        for c in range(
            len(options(k))
        ):  # macro-F1 over classes present in gold or pred
            tp = np.sum((pred[:, j] == c) & (gold[:, j] == c))
            fp, fn = (
                np.sum((pred[:, j] == c) & (gold[:, j] != c)),
                np.sum((pred[:, j] != c) & (gold[:, j] == c)),
            )
            if tp + fp + fn:
                f1s.append(2 * tp / (2 * tp + fp + fn))
        res[k] = {
            "acc": round(float(np.mean(pred[:, j] == gold[:, j])), 3),
            "macro_f1": round(float(np.mean(f1s)), 3),
        }
    return res


def show(tag, res):
    print(
        f"{tag:10}",
        " ".join(
            f"{k} {v['acc']:.2f}/{v['macro_f1']:.2f}"
            for k, v in res.items()
            if isinstance(v, dict)
        ),
        flush=True,
    )


gold_train = np.array([target(r["label"]) for r in train])
majority = np.tile(
    [Counter(gold_train[:, j]).most_common(1)[0][0] for j in range(len(IDS))],
    (len(val), 1),
)
res = {
    "label_dist_val": {k: dict(Counter(str(r["label"][k]) for r in val)) for k in IDS}
}
res["majority"] = scores(majority, val)
show("majority", res["majority"])
pred, _ = predict(val)
res["laya_zero_shot"] = scores(pred, val)
show("zero-shot", res["laya_zero_shot"])

for p in model.parameters():
    p.requires_grad_(False)
inject_adapter_in_model(
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=r".*encoder.*\.(Wqkv|Wo|Wi)",
    ),
    model,
)
params = [p for n, p in model.named_parameters() if "lora_" in n]
for p in params:
    p.data = p.data.float()
    p.requires_grad_(True)
if os.environ.get("QL_CKPT") == "1":  # off by default: ~20 GB without it, and faster
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
steps = EPOCHS * ((len(train) + ACC - 1) // ACC)
opt = torch.optim.AdamW(params, lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
rng = random.Random(0)
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    rng.shuffle(train)
    tot = 0.0
    for s, r in enumerate(train, 1):
        # Laya encodes the whole state once per label: train QPS random labels per item per step (speed)
        ids = rng.sample(TRAIN_IDS, min(QPS, len(TRAIN_IDS)))
        y_all = dict(zip(IDS, target(r["label"])))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, items = logits_of(r, ids)
        loss = sum(
            torch.nn.functional.cross_entropy(
                logits[j, : len(items[j]["markers"])].float()[None],
                torch.tensor([y_all[k]], device=logits.device),
                weight=CW[k].to(logits.device),
            )
            for j, k in enumerate(ids)
        ) / len(ids)
        (loss / ACC).backward()
        tot += loss.item()
        if s % ACC == 0 or s == len(train):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        if s % 200 == 0:
            print(
                f"ep {ep + 1} {s}/{len(train)} loss {tot / s:.3f} {time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"epoch {ep + 1}/{EPOCHS} loss {tot / len(train):.3f}", flush=True)
    # eval + save after every epoch, so the run can be stopped once the curve flattens
    pred, sec = predict(val)
    res[f"laya_lora_ep{ep + 1}"] = scores(pred, val)
    res[f"laya_lora_ep{ep + 1}"]["steps"] = (ep + 1) * ((len(train) + ACC - 1) // ACC)
    show(f"lora ep{ep + 1}", res[f"laya_lora_ep{ep + 1}"])
    res["speed_ms_per_item_unmerged"] = round(1000 * sec, 1)
    torch.save(
        {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n},
        f"{S}/laya_quality_r8_ep{ep + 1}.pt",
    )
    json.dump(res, open(f"{S}/laya_quality_eval.json", "w"), indent=1)
print("finished", flush=True)
