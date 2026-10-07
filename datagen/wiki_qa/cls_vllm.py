"""Temporary: the Qwen3.5-4B LoRA quality classifier distilled from Bunny (sft_cls), offline with vLLM, same prompt and
article cut (6000 chars) as in training, json_schema decoding (bunny_label.Page).
  --val     the 100 gold val pages (qlabel2/val_a_*) -> qlabel_qwen_val.jsonl (same format as qlabel_bunny_val.jsonl)
  --corpus  every page with generated items and no Bunny label yet -> table pred_labels (id, model, items, error);
            used only to order Bunny's queue (problem pages first), never as final labels.
Usage: cls_vllm.py <adapter dir> --val | --corpus [--chunk 2000]"""

import argparse
import json
import os
import sqlite3

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

import bunny
from bunny_label import PROMPT, Page, message, val_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    con = sqlite3.connect(bunny.DB, timeout=120)
    if a.val:
        todo = list(val_pages())
    else:
        con.execute(
            "create table if not exists pred_labels (id text primary key, model text, items text, error text, created_at text default current_timestamp)"
        )
        done = {
            r[0]
            for r in con.execute("select id from bunny_labels where items is not null")
        }
        done |= {
            r[0] for r in con.execute("select id from pred_labels where error is null")
        }
        todo = (t for t in bunny.pages() if t[0]["id"] not in done)
    name = os.path.basename(a.adapter.rstrip("/"))
    llm = LLM(
        "Qwen/Qwen3.5-4B",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=8,
        max_num_seqs=256,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    lora = LoRARequest(name, 1, os.path.abspath(a.adapter))
    sp = SamplingParams(
        temperature=0,
        max_tokens=1500,
        structured_outputs=StructuredOutputsParams(json=Page.model_json_schema()),
    )
    out = open("qlabel_qwen_val.jsonl", "w") if a.val else None
    buf, n = [], 0

    def flush(buf):
        convs = [
            [
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": message(p | {"text": p["text"][:6000]}, its),
                },
            ]
            for p, its in buf
        ]
        outs = llm.chat(
            convs,
            sp,
            lora_request=lora,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        for (p, its), o in zip(buf, outs):
            labs, err = None, None
            try:
                labs = [
                    lb.model_dump()
                    for lb in Page.model_validate_json(o.outputs[0].text).labels
                ]
                if len(labs) != len(its):
                    labs, err = None, f"need {len(its)} labels, got {len(labs)}"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
            if out:
                out.write(
                    json.dumps({"id": p["id"], "labels": labs, "error": err}) + "\n"
                )
            else:
                con.execute(
                    "insert or replace into pred_labels (id, model, items, error) values (?, ?, ?, ?)",
                    (p["id"], name, json.dumps(labs) if labs else None, err),
                )
        con.commit()

    for t in todo:
        buf.append(t)
        if len(buf) == a.chunk:
            flush(buf)
            n += len(buf)
            buf = []
            print(n, "done", flush=True)
    if buf:
        flush(buf)
        n += len(buf)
    print("finished", n, flush=True)


if __name__ == "__main__":
    main()
