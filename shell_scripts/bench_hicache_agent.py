import asyncio
import math
import os
import time
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

from hicache_bench_common import (
    fit_text_ids, flush, force_eviction, metrics, refresh_metrics,
    run_requests, runtime_warmup, sync_generate, write_jsonl,
)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
PREFIX_LENGTHS = [int(x) for x in os.getenv("PREFIX_LENGTHS", "512,1024,2048,4096,8192,16384").split(",")]
SESSIONS_PER_LEN = int(os.getenv("SESSIONS_PER_LEN", "8"))
TAIL_LEN = int(os.getenv("TAIL_LEN", "64"))
OUTPUT_LEN = int(os.getenv("OUTPUT_LEN", "128"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
REQUEST_RATE_RAW = os.getenv("REQUEST_RATE", "inf")
REQUEST_RATE = float("inf") if REQUEST_RATE_RAW == "inf" else float(REQUEST_RATE_RAW)
TRIALS = int(os.getenv("TRIALS", "3"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))
RESULT_FILE = Path(os.getenv(
    "RESULT_FILE",
    f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/agent_{POLICY}.jsonl",
))

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size

CONTEXT_BODY = (
    "You are an autonomous agent solving a multi-step task. "
    "The context contains system instructions, tool descriptions, observations, "
    "intermediate state and persistent task memory. The stable context is reused "
    "across multiple turns while each new user step contributes only a short suffix. "
)
TAIL_BODY = (
    "User provides the next observation. Continue the task using the existing context "
    "and produce the next action and explanation. "
)

def build_session(prefix_len, sid, trial):
    prefix = fit_text_ids(
        tokenizer,
        f"Agent session {trial}-{sid}. Persistent context begins here. ",
        CONTEXT_BODY,
        prefix_len,
    )
    tail = fit_text_ids(
        tokenizer,
        f"Round two request for session {trial}-{sid}. ",
        TAIL_BODY,
        TAIL_LEN,
    )
    return prefix, prefix + tail

async def run_trial(trial):
    flush(BASE_URL)
    targets = []
    sid = 0

    for prefix_len in PREFIX_LENGTHS:
        for _ in range(SESSIONS_PER_LEN):
            prefix, revisit = build_session(prefix_len, sid, trial)
            targets.append((sid, prefix_len, prefix, revisit))
            sid += 1

    print(f"\n[Trial {trial}] warming {len(targets)} Agent prefixes...")
    for sid, prefix_len, prefix, _ in targets:
        sync_generate(BASE_URL, prefix, f"agent_warm_t{trial}_s{sid}_p{prefix_len}", 1)

    time.sleep(1.0)
    host = refresh_metrics(BASE_URL, VOCAB_SIZE, 600000 + trial)
    if host["host_used"] <= 0:
        raise RuntimeError("Host KV cache is still empty after warmup.")

    print(f"[Trial {trial}] forcing GPU KV eviction...")
    prep_evicted = force_eviction(
        BASE_URL, VOCAB_SIZE, EVICTOR_LEN, NUM_EVICTORS,
        7_000_000 + trial * NUM_EVICTORS,
    )

    before = metrics(BASE_URL)
    specs = []

    for sid, prefix_len, _, revisit in targets:
        specs.append({
            "ids": revisit,
            "rid": f"agent_target_{POLICY}_t{trial}_s{sid}_p{prefix_len}",
            "max_new_tokens": OUTPUT_LEN,
            "extra": {
                "record_type": "request",
                "workload": "agent_l2_hit",
                "policy": POLICY,
                "trial": trial,
                "session_id": sid,
                "prefix_len": prefix_len,
                "tail_len": TAIL_LEN,
                "group": "target",
            },
        })

    start = time.perf_counter()
    rows = await run_requests(
        specs, BASE_URL, MAX_CONCURRENCY, REQUEST_RATE,
        8_000_000 + trial,
    )
    duration = time.perf_counter() - start
    after = metrics(BASE_URL)

    total_output = sum(x["output_tokens"] for x in rows)
    summary = {
        "record_type": "summary",
        "workload": "agent_l2_hit",
        "policy": POLICY,
        "trial": trial,
        "requests": len(rows),
        "duration_s": duration,
        "request_throughput": len(rows) / duration,
        "output_throughput": total_output / duration,
        "prep_evicted_delta": prep_evicted,
        "measurement_evicted_delta": after["evicted"] - before["evicted"],
        "load_back_delta": after["load_back"] - before["load_back"],
    }

    write_jsonl(RESULT_FILE, rows + [summary])
    print(
        f"[PASS] trial={trial} duration={duration:.3f}s "
        f"req/s={summary['request_throughput']:.3f} "
        f"out_tok/s={summary['output_throughput']:.1f} "
        f"load_back={summary['load_back_delta']:.0f}"
    )

async def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    runtime_warmup(BASE_URL, VOCAB_SIZE)
    print(
        f"POLICY={POLICY}, prefix={PREFIX_LENGTHS}, sessions/len={SESSIONS_PER_LEN}, "
        f"concurrency={MAX_CONCURRENCY}, rate={REQUEST_RATE_RAW}, trials={TRIALS}"
    )
    for trial in range(TRIALS):
        await run_trial(trial)

if __name__ == "__main__":
    asyncio.run(main())
