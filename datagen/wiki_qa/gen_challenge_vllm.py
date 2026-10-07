"""Temporary: challenging items from the Qwen3.5-4B LoRA distilled from Bunny (sft_challenge), offline with vLLM.
Same prompt and message as gen_challenge.py (forbidden words from bm25), pages of 1,500-8,000 chars (the trained
range) not yet done by Bunny or here, languages round-robin. Each item gets its BM25 rank and prompt "v2-qwen".
Output table challenge_qwen (id, model, items, error); never replaces anything.
Usage: gen_challenge_vllm.py <adapter dir> [--limit N] [--chunk 512]"""

import argparse
import json
import os
import sqlite3

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

import bm25
import bunny
from gen_challenge import PROMPT, Items, message, queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    con = sqlite3.connect(bunny.DB, timeout=120)
    con.execute(
        "create table if not exists challenge_qwen (id text primary key, model text, items text, error text, created_at text default current_timestamp)"
    )
    done = {r[0] for r in con.execute("select id from challenge_items")} | {
        r[0] for r in con.execute("select id from challenge_qwen where error is null")
    }
    ids = queue(done)
    size = {}
    for line in open(bunny.PAGES):
        p = json.loads(line)
        size[p["id"]] = p["char_count"]
    ids = [i for i in ids if size[i] <= 8000][: a.limit]
    order = {i: n for n, i in enumerate(ids)}
    pages = sorted(
        (p for p in map(json.loads, open(bunny.PAGES)) if p["id"] in order),
        key=lambda p: order[p["id"]],
    )
    print(len(pages), "pages", flush=True)
    bm25.load()
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
        temperature=0.6,
        max_tokens=2048,
        structured_outputs=StructuredOutputsParams(json=Items.model_json_schema()),
    )
    for s in range(0, len(pages), a.chunk):
        chunk = pages[s : s + a.chunk]
        convs = [
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": message(p)},
            ]
            for p in chunk
        ]
        outs = llm.chat(
            convs,
            sp,
            lora_request=lora,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        for p, o in zip(chunk, outs):
            items, err = None, None
            try:
                its = [
                    it.model_dump()
                    for it in Items.model_validate_json(o.outputs[0].text).items
                ]
                ranks = bm25.rank([it["question"] for it in its], [p["id"]] * len(its))
                for it, r in zip(its, ranks):
                    it |= {"challenging": True, "bm25_rank": r, "prompt": "v2-qwen"}
                items = json.dumps(its, ensure_ascii=False)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
            con.execute(
                "insert or replace into challenge_qwen (id, model, items, error) values (?, ?, ?, ?)",
                (p["id"], name, items, err),
            )
        con.commit()
        print(s + len(chunk), "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
