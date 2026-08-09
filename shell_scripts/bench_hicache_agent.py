import asyncio
import math
import os
import time
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

from hicache_bench_common import (
    fit_text_ids, flush, force_eviction, metric, metrics, refresh_metrics,
    run_requests, runtime_warmup, sync_generate, write_jsonl,clear_hicache_storage
)


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
PREFIX_LENGTHS = [int(x) for x in os.getenv("PREFIX_LENGTHS", "512,1024,2048,4096,8192,16384").split(",")]
SESSIONS_PER_LEN = int(os.getenv("SESSIONS_PER_LEN", "32"))
TAIL_LEN = int(os.getenv("TAIL_LEN", "64"))
OUTPUT_LEN = int(os.getenv("OUTPUT_LEN", "128"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
REQUEST_RATE_RAW = os.getenv("REQUEST_RATE", "inf")
REQUEST_RATE = float("inf") if REQUEST_RATE_RAW == "inf" else float(REQUEST_RATE_RAW)
TRIALS = int(os.getenv("TRIALS", "8"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "29000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "22"))
HOST_SAFETY_RATIO = float(os.getenv("HOST_SAFETY_RATIO", "0.94"))
RESULT_FILE = Path(os.getenv(
    "RESULT_FILE",
    f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/agent_{POLICY}_c{MAX_CONCURRENCY}.jsonl",
))
CLEAR_L3 = os.getenv("CLEAR_L3", "0") == "1"

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size

CONTEXT_BODY = (
    "You are an autonomous agent solving a multi-step task. "
    "The context contains system instructions, tool descriptions, observations, intermediate state "
    "and persistent task memory. The stable context is reused across multiple turns while each new "
    "user step contributes only a short suffix. "
)
TAIL_BODY = (
    "User provides the next observation. Continue the task using the existing context and produce "
    "the next action and explanation. "
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

def check_host_capacity(prefix_len):
    host_total = metric(BASE_URL, "sglang:hicache_host_total_tokens")
    estimated = prefix_len * SESSIONS_PER_LEN + EVICTOR_LEN * NUM_EVICTORS
    limit = host_total * HOST_SAFETY_RATIO
    print(f"[Capacity] prefix={prefix_len}, estimated_host_pressure={estimated}, host_total={host_total:.0f}, safety_limit={limit:.0f}")
    if estimated >= limit:
        raise RuntimeError(
            f"Estimated Host KV pressure is too high: {estimated} >= {limit:.0f}. "
            "Reduce SESSIONS_PER_LEN / NUM_EVICTORS / EVICTOR_LEN."
        )

async def run_case(prefix_len, trial):
    flush(BASE_URL)
    if CLEAR_L3:
        clear_hicache_storage(BASE_URL)
    check_host_capacity(prefix_len)

    targets = []
    for sid in range(SESSIONS_PER_LEN):
        prefix, revisit = build_session(prefix_len, sid, trial)
        targets.append((sid, prefix, revisit))

    print(f"\n[Case] policy={POLICY} c={MAX_CONCURRENCY} prefix={prefix_len} trial={trial}")
    print(f"[Warm] {SESSIONS_PER_LEN} target prefixes...")

    for sid, prefix, _ in targets:
        sync_generate(
            BASE_URL,
            prefix,
            f"agent_warm_{POLICY}_c{MAX_CONCURRENCY}_t{trial}_s{sid}_p{prefix_len}",
            1,
        )

    time.sleep(1.0)
    host = refresh_metrics(BASE_URL, VOCAB_SIZE, 6_000_000 + prefix_len * 10 + trial)
    print(f"[After warm] host_used={host['host_used']:.0f}")

    if host["host_used"] <= 0:
        raise RuntimeError(f"Host KV is empty after warmup: prefix={prefix_len}, trial={trial}")

    print(f"[Evict] {NUM_EVICTORS} x {EVICTOR_LEN} tokens...")
    prep_evicted = force_eviction(
        BASE_URL,
        VOCAB_SIZE,
        EVICTOR_LEN,
        NUM_EVICTORS,
        7_000_000 + prefix_len * 100 + trial * NUM_EVICTORS,
    )

    before = metrics(BASE_URL)

    specs = []
    for sid, _, revisit in targets:
        specs.append({
            "ids": revisit,
            "rid": f"agent_target_{POLICY}_c{MAX_CONCURRENCY}_t{trial}_s{sid}_p{prefix_len}",
            "max_new_tokens": OUTPUT_LEN,
            "extra": {
                "record_type": "request",
                "workload": "agent_l2_hit",
                "policy": POLICY,
                "trial": trial,
                "session_id": sid,
                "prefix_len": prefix_len,
                "tail_len": TAIL_LEN,
                "max_concurrency": MAX_CONCURRENCY,
                "group": "target",
            },
        })

    start = time.perf_counter()
    rows = await run_requests(
        specs,
        BASE_URL,
        MAX_CONCURRENCY,
        REQUEST_RATE,
        8_000_000 + prefix_len * 100 + trial,
    )
    duration = time.perf_counter() - start
    time.sleep(0.3)
    after = metrics(BASE_URL)

    load_back_delta = after["load_back"] - before["load_back"]
    measurement_evicted_delta = after["evicted"] - before["evicted"]

    if POLICY == "always_restore" and load_back_delta <= 0:
        raise RuntimeError(
            f"always_restore did not trigger L2->L1 load-back: prefix={prefix_len}, "
            f"trial={trial}, load_back_delta={load_back_delta}"
        )

    total_output = sum(x["output_tokens"] for x in rows)
    summary = {
        "record_type": "summary",
        "workload": "agent_l2_hit",
        "policy": POLICY,
        "trial": trial,
        "prefix_len": prefix_len,
        "max_concurrency": MAX_CONCURRENCY,
        "requests": len(rows),
        "duration_s": duration,
        "request_throughput": len(rows) / duration,
        "output_throughput": total_output / duration,
        "prep_evicted_delta": prep_evicted,
        "measurement_evicted_delta": measurement_evicted_delta,
        "load_back_delta": load_back_delta,
        "host_used_before_revisit": before["host_used"],
    }

    write_jsonl(RESULT_FILE, rows + [summary])

    ttfts = sorted(x["ttft_ms"] for x in rows)
    p50 = ttfts[len(ttfts) // 2]
    print(
        f"[PASS] prefix={prefix_len} trial={trial} req={len(rows)} "
        f"P50_TTFT={p50:.3f}ms load_back={load_back_delta:.0f} "
        f"evicted={measurement_evicted_delta:.0f} out_tok/s={summary['output_throughput']:.1f}"
    )

async def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    runtime_warmup(BASE_URL, VOCAB_SIZE)

    print(
        f"POLICY={POLICY}, prefix={PREFIX_LENGTHS}, sessions/len={SESSIONS_PER_LEN}, "
        f"concurrency={MAX_CONCURRENCY}, rate={REQUEST_RATE_RAW}, trials={TRIALS}, "
        f"evictors={NUM_EVICTORS}x{EVICTOR_LEN}, result={RESULT_FILE}"
    )

    for trial in range(TRIALS):
        for prefix_len in PREFIX_LENGTHS:
            await run_case(prefix_len, trial)

if __name__ == "__main__":
    asyncio.run(main())
