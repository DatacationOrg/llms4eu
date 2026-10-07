"""Temporary: rank-8 LoRA (Qwen3.5-4B by default, QG_MID to change), distilled from the teacher. Loss on the answer tokens only.
--task qg: question generation (gen_questions PROMPT, targets from qg_clean.jsonl, agent-cleaned teacher items).
--task cls: the 8 tag_spec tags (cls_prompt, targets from labels.jsonl).
--task file --data sft_*.jsonl: rows {system, user, target, split} from build_sft.py (adapter name from the file).
Saves peft adapters <QG_NAME>_<task>_s<step>/ for vLLM. Usage: train_lora.py --task qg [--steps 1000] [--ckpt]"""

import argparse
import json
import os
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from gen_questions import PROMPT, user_msg  # noqa: E402
from tag_spec import IDS, cls_prompt, labelled  # noqa: E402

MID = os.environ.get(
    "QG_MID", "Qwen/Qwen3.5-4B"
)  # the tag classifier used google/gemma-4-E4B-it
NAME = os.environ.get("QG_NAME", "qwen35")  # adapter dir prefix
NO_THINK = {"enable_thinking": False}  # Qwen3.5 template switch; Gemma ignores it


def pools():
    return {
        r["url"]: r
        for f in ("label_pool.json", "label_pool2.json", "label_pool3.json")
        for r in json.load(open(f"{S}/{f}"))
    }


def examples(task, data=None):
    """(system, user, target json, split) per labelled article."""
    if task == "file":
        return [
            (r["system"], r["user"], r["target"], r["split"])
            for r in map(json.loads, open(data))
        ]
    if task == "cls":
        return [
            (
                cls_prompt(),
                user_msg(r),
                json.dumps({k: r["label"][k] for k in IDS}),
                r["split"],
            )
            for r in labelled()
        ]
    pool = pools()
    out = []
    for fx in map(json.loads, open(f"{S}/qg_clean.jsonl")):
        r = pool[fx["url"]]
        target = json.dumps({"items": fx["items"]}, ensure_ascii=False)
        out.append((PROMPT, user_msg(r), target, r["split"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qg", "cls", "file"], required=True)
    ap.add_argument("--data", help="--task file: jsonl from build_sft.py")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--all", action="store_true", help="train on heldout too (final model)"
    )
    a = ap.parse_args()
    random.seed(0)

    proc = AutoProcessor.from_pretrained(MID)
    tok = proc.tokenizer
    data = []
    if a.task == "file":
        a.task = os.path.basename(a.data).removesuffix(
            ".jsonl"
        )  # names the adapter, e.g. qwen35_sft_fix_s0500
    for system, user, target, split in examples("file" if a.data else a.task, a.data):
        if split != "train" and not a.all:
            continue
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = proc.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **NO_THINK
        )
        full = proc.apply_chat_template(
            chat + [{"role": "assistant", "content": target}],
            tokenize=False,
            **NO_THINK,
        )
        assert full.startswith(prompt)
        data.append(
            (
                tok(prompt, add_special_tokens=False).input_ids,
                tok(full[len(prompt) :], add_special_tokens=False).input_ids,
            )
        )
    print(len(data), "examples", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MID, dtype=torch.bfloat16, device_map="cuda"
    )
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    if a.ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05
    )
    model.train()
    step, losses, ep = 0, [], 0
    while step < a.steps:  # batch 1: no padding needed
        ep += 1
        random.shuffle(data)
        for p, t in data:
            if step == a.steps:
                break
            ids = torch.tensor([p + t], device="cuda")
            k = len(t) + 1  # logits only for the answer window
            logits = model(input_ids=ids, logits_to_keep=k).logits
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1].float(), ids[0, -len(t) :]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            losses.append(loss.item())
            if step % a.save_every == 0 or step == a.steps:
                model.save_pretrained(f"{S}/{NAME}_{a.task}_s{step:04d}")
            if step % 20 == 0:
                print(
                    f"ep {ep} step {step}/{a.steps} loss {sum(losses[-20:]) / 20:.3f} "
                    f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                    flush=True,
                )
    json.dump(losses, open(f"{S}/{NAME}_{a.task}_losses.json", "w"))
    print("finished", flush=True)


if __name__ == "__main__":
    main()
