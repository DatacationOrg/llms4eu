"""Temporary: offline vLLM + LoRA with json_schema decoding: --qg (Qwen3.5-4B questions) and/or --cls (E4B tags).
--pool file.json: only those rows (eval; ids are urls). Without --pool: the whole corpus in gen_questions.queue
order (short, non-SE-lake first), in chunks, resumable. Results go to --db, tables questions / labels.
One base model per run (QG_MID): questions with Qwen/Qwen3.5-4B (default), tags with QG_MID=google/gemma-4-E4B-it.
Usage: run_vllm.py --qg qwen35_qg_s1000 [--pool eval.json] --db out.db [--deadline 2026-10-01T06:00]"""

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from gen_questions import PROMPT, Questions, queue, read_row, user_msg  # noqa: E402
from tag_spec import cls_prompt, label_model  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

MID = os.environ.get("QG_MID", "Qwen/Qwen3.5-4B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qg", help="question LoRA dir ('base' = no LoRA)")
    ap.add_argument("--cls", help="classifier LoRA dir ('base' = no LoRA)")
    ap.add_argument("--pool")
    ap.add_argument("--db", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument(
        "--deadline", help="UTC ISO time, e.g. 2026-09-30T05:45: no new chunks after it"
    )
    a = ap.parse_args()

    Label = label_model()
    tasks = {}  # table -> (system prompt, schema, max tokens, lora dir)
    if a.qg:
        tasks["questions"] = (PROMPT, Questions, 2048, a.qg)
    if a.cls:
        tasks["labels"] = (cls_prompt(), Label, 200, a.cls)
    db = sqlite3.connect(a.db, timeout=120)  # the API teacher may write to the same DB
    for t in tasks:
        db.execute(
            f"create table if not exists {t} (id text primary key, model text, items text,"
            " error text, created_at text default current_timestamp)"
        )
    done = set.intersection(
        *(
            {i for (i,) in db.execute(f"select id from {t} where error is null")}
            for t in tasks
        )
    )
    if a.pool:
        rows = [
            {**r, "id": r["url"]}
            for r in json.load(open(a.pool))
            if r["url"] not in done
        ]
    else:
        offs = queue(done)
        print(len(offs), "rows to do", flush=True)
        rows = (read_row(o) for o in offs)

    llm = LLM(
        MID,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        **(
            {"enable_lora": True, "max_lora_rank": 8, "max_loras": 2}
            if any(t[3] != "base" for t in tasks.values())
            else {}
        ),
        max_num_seqs=256,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    lora = {
        t: None if d == "base" else LoRARequest(t, n + 1, os.path.abspath(d))
        for n, (t, (_, _, _, d)) in enumerate(tasks.items())
    }
    sp = {
        t: SamplingParams(
            temperature=0,
            max_tokens=mt,
            structured_outputs=StructuredOutputsParams(json=schema.model_json_schema()),
        )
        for t, (_, schema, mt, _) in tasks.items()
    }
    name = {t: os.path.basename(d.rstrip("/")) for t, (_, _, _, d) in tasks.items()}

    def chunks():
        buf = []
        for r in rows:
            buf.append(r)
            if len(buf) == a.chunk:
                yield buf
                buf = []
        if buf:
            yield buf

    total = 0
    for chunk in chunks():
        if a.deadline and dt.datetime.now(dt.UTC) >= dt.datetime.fromisoformat(
            a.deadline
        ).replace(tzinfo=dt.UTC):
            print("deadline reached, stopping", flush=True)
            break
        t0 = time.time()
        convs, keys = [], []
        for t, (system, _, _, _) in tasks.items():
            for r in chunk:
                convs.append(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg(r)},
                    ]
                )
                keys.append((t, r["id"]))
        outs = llm.chat(
            convs,
            [sp[t] for t, _ in keys],
            lora_request=[lora[t] for t, _ in keys],
            use_tqdm=False,
            chat_template_kwargs={
                "enable_thinking": False
            },  # Qwen3.5; Gemma ignores it
        )
        for (t, pid), o in zip(keys, outs):
            text, items, err = o.outputs[0].text, None, None
            try:
                obj = tasks[t][1].model_validate_json(text)
                items = json.dumps(
                    [q.model_dump() for q in obj.items]
                    if t == "questions"
                    else obj.model_dump(),
                    ensure_ascii=False,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]} | {text[-200:]}"
            db.execute(
                # never overwrite a good row (e.g. written by the teacher meanwhile); replace errors
                f"insert into {t} (id, model, items, error) values (?, ?, ?, ?) on conflict(id) do update"
                " set model=excluded.model, items=excluded.items, error=excluded.error"
                f" where {t}.error is not null",
                (pid, name[t], items, err),
            )
        db.commit()
        total += len(chunk)
        print(
            f"{dt.datetime.now(dt.UTC):%H:%M} +{len(chunk)} (total {total}) "
            f"{time.time() - t0:.0f}s = {(time.time() - t0) / len(chunk):.2f}s/article",
            flush=True,
        )
    print("finished", flush=True)


if __name__ == "__main__":
    main()
