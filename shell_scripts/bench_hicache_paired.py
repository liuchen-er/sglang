import asyncio
import os
import time
from pathlib import Path

import aiohttp
import requests
from transformers import AutoConfig, AutoTokenizer

from hicache_bench_common import (
    fit_text_ids, flush, force_eviction, metrics, random_ids,
    refresh_metrics, runtime_warmup, stream_generate,
    sync_generate, write_jsonl,
)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
PREFIX_LENGTHS = [int(x) for x in os.getenv("PREFIX_LENGTHS", "512,1024,2048,4096,8192,16384").split(",")]
RUNNING_BS_LIST = [int(x) for x in os.getenv("RUNNING_BS_LIST", "0,4,16,32").split(",")]
TARGETS_PER_STATE = int(os.getenv("TARGETS_PER_STATE", "3"))
TRIALS = int(os.getenv("TRIALS", "3"))
TAIL_LEN = int(os.getenv("TAIL_LEN", "64"))
BG_INPUT_LEN = int(os.getenv("BG_INPUT_LEN", "256"))
BG_OUTPUT_LEN = int(os.getenv("BG_OUTPUT_LEN", "2048"))
BG_START_DELAY = float(os.getenv("BG_START_DELAY", "1.0"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "22"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "29000"))

RESULT_FILE = Path(os.getenv(
    "RESULT_FILE",
    f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/paired_{POLICY}.jsonl",
))

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size

TARGET_BODY = (
    "This is persistent context for a multi-turn agent. It contains system instructions, "
    "tool definitions, previous observations and reusable task state. "
)
TAIL_BODY = (
    "A new user observation arrives. Continue from the previous state and return the next action. "
)

def build_target(prefix_len, idx, trial, running_bs):
    prefix = fit_text_ids(
        tokenizer,
        f"Paired profile target {trial}-{running_bs}-{idx}. ",
        TARGET_BODY,
        prefix_len,
    )
    unique_id = 1000 + (trial * 257 + running_bs * 17 + idx) % max(VOCAB_SIZE - 2001, 1)
    prefix[0] = min(unique_id, VOCAB_SIZE - 1)
    tail = fit_text_ids(
        tokenizer,
        f"Next turn {trial}-{running_bs}-{idx}. ",
        TAIL_BODY,
        TAIL_LEN,
    )
    return prefix, prefix + tail

async def run_state(prefix_len, requested_bs, trial):
    flush(BASE_URL)

    targets = []
    for i in range(TARGETS_PER_STATE):
        prefix, revisit = build_target(prefix_len, i, trial, requested_bs)
        targets.append((i, prefix, revisit))

    print(f"\n[State] policy={POLICY} prefix={prefix_len} requested_bs={requested_bs} trial={trial}")
    for i, prefix, _ in targets:
        sync_generate(
            BASE_URL,
            prefix,
            f"paired_warm_{POLICY}_bs{requested_bs}_t{trial}_i{i}_p{prefix_len}",
            1,
        )

    time.sleep(0.8)
    refresh_metrics(BASE_URL, VOCAB_SIZE, 12_000_000 + prefix_len + trial + requested_bs)

    prep_evicted = force_eviction(
        BASE_URL,
        VOCAB_SIZE,
        EVICTOR_LEN,
        NUM_EVICTORS,
        13_000_000 + prefix_len * 100 + requested_bs * 10000 + trial * NUM_EVICTORS,
    )

    before = metrics(BASE_URL)
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        bg_tasks = []
        for i in range(requested_bs):
            ids = random_ids(
                BG_INPUT_LEN,
                14_000_000 + prefix_len * 100 + requested_bs * 10000 + trial * 100 + i,
                VOCAB_SIZE,
            )
            bg_tasks.append(asyncio.create_task(
                stream_generate(
                    session,
                    BASE_URL,
                    ids,
                    f"paired_bg_{POLICY}_bs{requested_bs}_t{trial}_i{i}_p{prefix_len}",
                    BG_OUTPUT_LEN,
                    {
                        "record_type": "background",
                        "policy": POLICY,
                        "requested_running_bs": requested_bs,
                        "trial": trial,
                        "prefix_len": prefix_len,
                    },
                )
            ))

        if bg_tasks:
            await asyncio.sleep(BG_START_DELAY)

        rows = []
        for i, _, revisit in targets:
            rid = f"paired_target_{POLICY}_bs{requested_bs}_t{trial}_i{i}_p{prefix_len}"
            row = await stream_generate(
                session,
                BASE_URL,
                revisit,
                rid,
                1,
                {
                    "record_type": "request",
                    "workload": "paired_profile",
                    "policy": POLICY,
                    "requested_running_bs": requested_bs,
                    "trial": trial,
                    "target_id": i,
                    "prefix_len": prefix_len,
                    "tail_len": TAIL_LEN,
                },
            )
            rows.append(row)

        if bg_tasks:
            await asyncio.gather(*bg_tasks)

    time.sleep(0.3)
    after = metrics(BASE_URL)
    load_back_delta = after["load_back"] - before["load_back"]

    if POLICY == "always_restore" and load_back_delta <= 0:
        raise RuntimeError(
            f"No load-back observed: prefix={prefix_len}, bs={requested_bs}, trial={trial}"
        )
    if POLICY == "always_recompute" and load_back_delta != 0:
        raise RuntimeError(
            f"Unexpected load-back under recompute: {load_back_delta}"
        )

    summary = {
        "record_type": "summary",
        "workload": "paired_profile",
        "policy": POLICY,
        "requested_running_bs": requested_bs,
        "trial": trial,
        "prefix_len": prefix_len,
        "requests": len(rows),
        "prep_evicted_delta": prep_evicted,
        "measurement_evicted_delta": after["evicted"] - before["evicted"],
        "load_back_delta": load_back_delta,
    }

    write_jsonl(RESULT_FILE, rows + [summary])
    vals = sorted(x["ttft_ms"] for x in rows)
    print(
        f"[PASS] prefix={prefix_len} bs={requested_bs} trial={trial} "
        f"median_TTFT={vals[len(vals)//2]:.3f}ms load_back={load_back_delta:.0f}"
    )

async def main():
    if POLICY not in ("always_restore", "always_recompute"):
        raise ValueError("Paired profiling only supports always_restore / always_recompute")

    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    runtime_warmup(BASE_URL, VOCAB_SIZE)

    print(
        f"POLICY={POLICY}, prefix={PREFIX_LENGTHS}, running_bs={RUNNING_BS_LIST}, "
        f"targets/state={TARGETS_PER_STATE}, trials={TRIALS}"
    )

    for trial in range(TRIALS):
        for running_bs in RUNNING_BS_LIST:
            for prefix_len in PREFIX_LENGTHS:
                await run_state(prefix_len, running_bs, trial)

if __name__ == "__main__":
    asyncio.run(main())
