import asyncio
import os
import time
from pathlib import Path

import aiohttp
import requests
from transformers import AutoConfig, AutoTokenizer

from hicache_bench_common import (
    fit_text_ids, flush, force_eviction, metrics, refresh_metrics,
    runtime_warmup, stream_generate, sync_generate, write_jsonl,
)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
INJECT = int(os.getenv("INJECT", "1"))
TRIALS = int(os.getenv("TRIALS", "3"))

DECODE_REQUESTS = int(os.getenv("DECODE_REQUESTS", "32"))
DECODE_INPUT_LEN = int(os.getenv("DECODE_INPUT_LEN", "512"))
DECODE_OUTPUT_LEN = int(os.getenv("DECODE_OUTPUT_LEN", "1024"))

TARGET_PREFIX_LENGTHS = [int(x) for x in os.getenv(
    "TARGET_PREFIX_LENGTHS", "512,1024,2048,4096"
).split(",")]
TARGETS_PER_LEN = int(os.getenv("TARGETS_PER_LEN", "4"))
TARGET_TAIL_LEN = int(os.getenv("TARGET_TAIL_LEN", "64"))
TARGET_OUTPUT_LEN = int(os.getenv("TARGET_OUTPUT_LEN", "128"))
INJECT_DELAY = float(os.getenv("INJECT_DELAY", "1.0"))

EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))

DEFAULT_NAME = "mixed_control.jsonl" if not INJECT else f"mixed_{POLICY}.jsonl"
RESULT_FILE = Path(os.getenv(
    "RESULT_FILE",
    f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/{DEFAULT_NAME}",
))

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size

TARGET_BODY = (
    "This is persistent context for a long-running agent. It contains tool definitions, "
    "task state, previous observations, constraints and reusable knowledge. "
)
TAIL_BODY = (
    "A new observation has arrived. Continue from the cached state and return the next action. "
)
DECODE_BODY = (
    "Generate a detailed technical explanation of large language model inference, GPU execution, "
    "attention, KV cache management and continuous batching. Continue discussing the topic in detail. "
)

def target(prefix_len, sid, trial):
    prefix = fit_text_ids(
        tokenizer,
        f"Target Agent {trial}-{sid}. Persistent state. ",
        TARGET_BODY,
        prefix_len,
    )
    tail = fit_text_ids(
        tokenizer,
        f"New turn {trial}-{sid}. ",
        TAIL_BODY,
        TARGET_TAIL_LEN,
    )
    return prefix, prefix + tail

def decode_prompt(idx, trial):
    return fit_text_ids(
        tokenizer,
        f"Decode workload request {trial}-{idx}. ",
        DECODE_BODY,
        DECODE_INPUT_LEN,
    )

async def run_trial(trial):
    flush(BASE_URL)

    targets = []
    sid = 0
    for prefix_len in TARGET_PREFIX_LENGTHS:
        for _ in range(TARGETS_PER_LEN):
            prefix, revisit = target(prefix_len, sid, trial)
            targets.append((sid, prefix_len, prefix, revisit))
            sid += 1

    print(f"\n[Trial {trial}] preparing {len(targets)} L2 target prefixes...")
    for sid, prefix_len, prefix, _ in targets:
        sync_generate(BASE_URL, prefix, f"mixed_warm_t{trial}_s{sid}_p{prefix_len}", 1)

    time.sleep(1.0)
    host = refresh_metrics(BASE_URL, VOCAB_SIZE, 9_000_000 + trial)
    if host["host_used"] <= 0:
        raise RuntimeError("Host KV cache is empty after target warmup.")

    prep_evicted = force_eviction(
        BASE_URL, VOCAB_SIZE, EVICTOR_LEN, NUM_EVICTORS,
        10_000_000 + trial * NUM_EVICTORS,
    )

    before = metrics(BASE_URL)

    decode_specs = []
    for i in range(DECODE_REQUESTS):
        decode_specs.append({
            "ids": decode_prompt(i, trial),
            "rid": f"mixed_decode_{POLICY}_t{trial}_{i}",
            "max_new_tokens": DECODE_OUTPUT_LEN,
            "extra": {
                "record_type": "request",
                "workload": "mixed",
                "policy": POLICY,
                "trial": trial,
                "group": "decode",
            },
        })

    target_specs = []
    for sid, prefix_len, _, revisit in targets:
        target_specs.append({
            "ids": revisit,
            "rid": f"mixed_target_{POLICY}_t{trial}_s{sid}_p{prefix_len}",
            "max_new_tokens": TARGET_OUTPUT_LEN,
            "extra": {
                "record_type": "request",
                "workload": "mixed",
                "policy": POLICY,
                "trial": trial,
                "group": "target",
                "session_id": sid,
                "prefix_len": prefix_len,
                "tail_len": TARGET_TAIL_LEN,
            },
        })

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()

        decode_tasks = [
            asyncio.create_task(
                stream_generate(
                    session, BASE_URL, x["ids"], x["rid"],
                    x["max_new_tokens"], x["extra"],
                )
            )
            for x in decode_specs
        ]

        await asyncio.sleep(INJECT_DELAY)

        target_tasks = []
        if INJECT:
            target_tasks = [
                asyncio.create_task(
                    stream_generate(
                        session, BASE_URL, x["ids"], x["rid"],
                        x["max_new_tokens"], x["extra"],
                    )
                )
                for x in target_specs
            ]

        decode_rows = await asyncio.gather(*decode_tasks)
        target_rows = await asyncio.gather(*target_tasks) if target_tasks else []
        duration = time.perf_counter() - start

    after = metrics(BASE_URL)
    rows = decode_rows + target_rows
    total_output = sum(x["output_tokens"] for x in rows)

    summary = {
        "record_type": "summary",
        "workload": "mixed",
        "policy": POLICY,
        "trial": trial,
        "inject": INJECT,
        "decode_requests": len(decode_rows),
        "target_requests": len(target_rows),
        "duration_s": duration,
        "output_throughput": total_output / duration,
        "prep_evicted_delta": prep_evicted,
        "measurement_evicted_delta": after["evicted"] - before["evicted"],
        "load_back_delta": after["load_back"] - before["load_back"],
    }

    write_jsonl(RESULT_FILE, rows + [summary])
    print(
        f"[PASS] trial={trial} inject={INJECT} duration={duration:.3f}s "
        f"out_tok/s={summary['output_throughput']:.1f} "
        f"measure_evict={summary['measurement_evicted_delta']:.0f} "
        f"load_back={summary['load_back_delta']:.0f}"
    )

async def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    runtime_warmup(BASE_URL, VOCAB_SIZE)
    print(
        f"POLICY={POLICY}, inject={INJECT}, decode={DECODE_REQUESTS}x{DECODE_OUTPUT_LEN}, "
        f"targets={len(TARGET_PREFIX_LENGTHS) * TARGETS_PER_LEN}, delay={INJECT_DELAY}s"
    )
    for trial in range(TRIALS):
        await run_trial(trial)

if __name__ == "__main__":
    asyncio.run(main())
