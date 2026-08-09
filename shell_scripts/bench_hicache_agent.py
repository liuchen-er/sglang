import asyncio
import os
import time
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

from hicache_bench_common import (
    clear_hicache_storage,
    fit_text_ids,
    flush,
    force_eviction,
    metric,
    metrics,
    refresh_metrics,
    run_requests,
    runtime_warmup,
    sync_generate,
    wait_metric_delta,
    write_jsonl,
)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")

PREFIX_LENGTHS = [
    int(x)
    for x in os.getenv(
        "PREFIX_LENGTHS",
        "256,512,1024,4096,16384",
    ).split(",")
]

SESSIONS_PER_LEN = int(os.getenv("SESSIONS_PER_LEN", "32"))
TAIL_LEN = int(os.getenv("TAIL_LEN", "64"))
OUTPUT_LEN = int(os.getenv("OUTPUT_LEN", "128"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))

REQUEST_RATE_RAW = os.getenv("REQUEST_RATE", "inf")
REQUEST_RATE = (
    float("inf")
    if REQUEST_RATE_RAW == "inf"
    else float(REQUEST_RATE_RAW)
)

TRIALS = int(os.getenv("TRIALS", "1"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "29000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "22"))

TARGET_CACHE_TIER = os.getenv("TARGET_CACHE_TIER", "L2").upper()
HOST_SAFETY_RATIO = float(os.getenv("HOST_SAFETY_RATIO", "0.94"))
L3_OVERFLOW_RATIO = float(os.getenv("L3_OVERFLOW_RATIO", "1.03"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "64"))
L3_BACKUP_TIMEOUT_S = float(os.getenv("L3_BACKUP_TIMEOUT_S", "180"))
L3_PREP_MODE = os.getenv("L3_PREP_MODE", "flush").lower()

CLEAR_L3 = (
    os.getenv(
        "CLEAR_L3",
        "1" if TARGET_CACHE_TIER == "L3" else "0",
    )
    == "1"
)

VALIDATE_CACHE_TIER = os.getenv("VALIDATE_CACHE_TIER", "1") == "1"
RESET_RESULT = os.getenv("RESET_RESULT", "1") == "1"
METRIC_PROBE_TOKENS = 128

RESULT_FILE = Path(
    os.getenv(
        "RESULT_FILE",
        f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/"
        f"agent_{POLICY}_c{MAX_CONCURRENCY}.jsonl",
    )
)

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
    "User provides the next observation. Continue the task using the existing "
    "context and produce the next action and explanation. "
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
    m = metrics(BASE_URL)
    host_total = int(m["host_total"])
    target_tokens = prefix_len * SESSIONS_PER_LEN
    evictor_tokens = EVICTOR_LEN * NUM_EVICTORS

    if TARGET_CACHE_TIER == "L2":
        estimated = target_tokens + METRIC_PROBE_TOKENS + evictor_tokens
        safety_limit = int(host_total * HOST_SAFETY_RATIO)

        print(
            f"[Capacity] tier=L2 prefix={prefix_len} target={target_tokens} "
            f"probe={METRIC_PROBE_TOKENS} evictor={evictor_tokens} "
            f"estimated={estimated} host_total={host_total} "
            f"safety_limit={safety_limit}"
        )

        if estimated >= safety_limit:
            raise RuntimeError(
                "Estimated Host KV pressure is too high for L2-hit workload: "
                f"{estimated} >= {safety_limit}. "
                "Reduce SESSIONS_PER_LEN / NUM_EVICTORS / EVICTOR_LEN."
            )
        return

    if TARGET_CACHE_TIER == "L3":
        if L3_PREP_MODE == "flush":
            print(
                f"[Capacity] tier=L3 prep=flush prefix={prefix_len} "
                f"target={target_tokens} host_total={host_total}"
            )
            return
        # Targets are older than the metric probe and evictors.
        # Newer cache pressure itself must exceed Host L2 capacity
        # so all target prefixes can be pushed out of L2.
        newer_pressure = METRIC_PROBE_TOKENS + evictor_tokens
        required = int(host_total * L3_OVERFLOW_RATIO)

        print(
            f"[Capacity] tier=L3 prefix={prefix_len} target={target_tokens} "
            f"probe={METRIC_PROBE_TOKENS} evictor={evictor_tokens} "
            f"newer_pressure={newer_pressure} host_total={host_total} "
            f"required_newer_pressure={required}"
        )

        if newer_pressure <= required:
            raise RuntimeError(
                "Post-warm pressure is too low to reliably evict targets from L2: "
                f"{newer_pressure} <= {required}. "
                "Increase NUM_EVICTORS / EVICTOR_LEN."
            )
        return

    raise ValueError(
        f"Unknown TARGET_CACHE_TIER={TARGET_CACHE_TIER}; expected L2 or L3."
    )


def _cache_details(row):
    meta = row.get("cache_meta") or {}
    details = meta.get("cached_tokens_details") or {}
    return {
        "device": int(details.get("device", 0) or 0),
        "host": int(details.get("host", 0) or 0),
        "storage": int(details.get("storage", 0) or 0),
        "storage_backend": details.get("storage_backend"),
    }


def validate_cache_tier(rows, prefix_len):
    if not VALIDATE_CACHE_TIER:
        return

    # Strict tier validation is meaningful for the restore path.
    # always_recompute intentionally rejects the available cache,
    # so its final cache report cannot be interpreted as a restore source.
    if POLICY == "always_recompute":
        return

    for row in rows:
        d = _cache_details(row)

        if TARGET_CACHE_TIER == "L2":
            if d["device"] != 0 or d["host"] < prefix_len:
                raise RuntimeError(
                    f"L2-hit validation failed: rid={row['rid']} "
                    f"device={d['device']} host={d['host']} "
                    f"storage={d['storage']} expected_host>={prefix_len}"
                )

        elif TARGET_CACHE_TIER == "L3":
            if (
                d["device"] != 0
                or d["host"] != 0
                or d["storage"] < prefix_len
            ):
                raise RuntimeError(
                    f"L3-hit validation failed: rid={row['rid']} "
                    f"device={d['device']} host={d['host']} "
                    f"storage={d['storage']} expected_storage>={prefix_len}"
                )

            if d["storage_backend"] != "HiCacheFile":
                raise RuntimeError(
                    f"Unexpected L3 backend: rid={row['rid']} "
                    f"backend={d['storage_backend']}"
                )

    print(
        f"[CacheTier PASS] tier={TARGET_CACHE_TIER} "
        f"prefix={prefix_len} requests={len(rows)}"
    )


async def run_case(prefix_len, trial):
    flush(BASE_URL)

    if CLEAR_L3:
        clear_hicache_storage(BASE_URL)

    check_host_capacity(prefix_len)

    storage_backup_before = metric(
        BASE_URL,
        "sglang:backuped_tokens_total",
        default=0.0,
    )

    targets = []
    for sid in range(SESSIONS_PER_LEN):
        prefix, revisit = build_session(prefix_len, sid, trial)
        targets.append((sid, prefix, revisit))

    print(
        f"\n[Case] tier={TARGET_CACHE_TIER} policy={POLICY} "
        f"c={MAX_CONCURRENCY} prefix={prefix_len} trial={trial}"
    )
    print(f"[Warm] {SESSIONS_PER_LEN} target prefixes...")

    for sid, prefix, _ in targets:
        sync_generate(
            BASE_URL,
            prefix,
            f"agent_warm_{POLICY}_c{MAX_CONCURRENCY}_"
            f"t{trial}_s{sid}_p{prefix_len}",
            1,
        )

    if TARGET_CACHE_TIER == "L3":
        aligned_prefix_len = (prefix_len // PAGE_SIZE) * PAGE_SIZE
        expected_backup_tokens = aligned_prefix_len * SESSIONS_PER_LEN

        print(
            f"[L3] waiting for storage backup completion: "
            f"expected={expected_backup_tokens} tokens..."
        )

        wait_metric_delta(
            BASE_URL,
            "sglang:backuped_tokens_total",
            storage_backup_before,
            expected_backup_tokens,
            timeout_s=L3_BACKUP_TIMEOUT_S,
            poll_s=0.5,
        )

    if TARGET_CACHE_TIER == "L3" and L3_PREP_MODE == "flush":
        host = metrics(BASE_URL)
    else:
        host = refresh_metrics(
            BASE_URL,
            VOCAB_SIZE,
            6_000_000 + prefix_len * 10 + trial,
        )

    print(
        f"[After warm] host_used={host['host_used']:.0f} "
        f"host_total={host['host_total']:.0f} "
        f"storage_backuped={host['storage_backuped']:.0f}"
    )

    if host["host_used"] <= 0:
        raise RuntimeError(
            f"Host KV is empty after warmup: prefix={prefix_len}, trial={trial}"
        )

    if TARGET_CACHE_TIER == "L3" and L3_PREP_MODE == "flush":
        print("[L3 Prep] flushing L1/L2 while preserving L3...")
        flush(BASE_URL)
        time.sleep(0.5)
        prep_evicted = 0.0

    elif TARGET_CACHE_TIER == "L3" and L3_PREP_MODE == "pressure":
        print(f"[Evict] {NUM_EVICTORS} x {EVICTOR_LEN} tokens...")
        prep_evicted = force_eviction(
            BASE_URL,
            VOCAB_SIZE,
            EVICTOR_LEN,
            NUM_EVICTORS,
            7_000_000 + prefix_len * 100 + trial * NUM_EVICTORS,
        )
        time.sleep(0.5)

    elif TARGET_CACHE_TIER == "L2":
        print(f"[Evict] {NUM_EVICTORS} x {EVICTOR_LEN} tokens...")
        prep_evicted = force_eviction(
            BASE_URL,
            VOCAB_SIZE,
            EVICTOR_LEN,
            NUM_EVICTORS,
            7_000_000 + prefix_len * 100 + trial * NUM_EVICTORS,
        )

    else:
        raise ValueError(
            f"Unknown L3_PREP_MODE={L3_PREP_MODE}; expected flush or pressure"
        )

    before = metrics(BASE_URL)
    workload = f"agent_{TARGET_CACHE_TIER.lower()}_hit"

    specs = []
    for sid, _, revisit in targets:
        specs.append(
            {
                "ids": revisit,
                "rid": (
                    f"agent_target_{POLICY}_c{MAX_CONCURRENCY}_"
                    f"t{trial}_s{sid}_p{prefix_len}"
                ),
                "max_new_tokens": OUTPUT_LEN,
                "extra": {
                    "record_type": "request",
                    "workload": workload,
                    "target_cache_tier": TARGET_CACHE_TIER,
                    "policy": POLICY,
                    "trial": trial,
                    "session_id": sid,
                    "prefix_len": prefix_len,
                    "tail_len": TAIL_LEN,
                    "max_concurrency": MAX_CONCURRENCY,
                    "group": "target",
                },
            }
        )

    start = time.perf_counter()

    rows = await run_requests(
        specs,
        BASE_URL,
        MAX_CONCURRENCY,
        REQUEST_RATE,
        8_000_000 + prefix_len * 100 + trial,
    )

    duration = time.perf_counter() - start

    # Validate actual cache source before interpreting latency.
    validate_cache_tier(rows, prefix_len)

    time.sleep(0.3)
    after = metrics(BASE_URL)

    load_back_delta = after["load_back"] - before["load_back"]
    measurement_evicted_delta = after["evicted"] - before["evicted"]

    if POLICY == "always_restore" and load_back_delta <= 0:
        raise RuntimeError(
            "always_restore did not trigger restore/load-back: "
            f"tier={TARGET_CACHE_TIER}, prefix={prefix_len}, "
            f"trial={trial}, load_back_delta={load_back_delta}"
        )

    if POLICY == "always_recompute" and load_back_delta != 0:
        raise RuntimeError(
            "always_recompute unexpectedly triggered load-back: "
            f"tier={TARGET_CACHE_TIER}, prefix={prefix_len}, "
            f"trial={trial}, load_back_delta={load_back_delta}"
        )

    total_output = sum(x["output_tokens"] for x in rows)

    summary = {
        "record_type": "summary",
        "workload": workload,
        "target_cache_tier": TARGET_CACHE_TIER,
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
        "host_total": before["host_total"],
    }

    write_jsonl(RESULT_FILE, rows + [summary])

    ttfts = sorted(x["ttft_ms"] for x in rows)
    p50 = ttfts[len(ttfts) // 2]

    print(
        f"[PASS] tier={TARGET_CACHE_TIER} prefix={prefix_len} "
        f"trial={trial} req={len(rows)} "
        f"P50_TTFT={p50:.3f}ms "
        f"load_back={load_back_delta:.0f} "
        f"evicted={measurement_evicted_delta:.0f} "
        f"out_tok/s={summary['output_throughput']:.1f}"
    )


async def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()

    if TARGET_CACHE_TIER not in ("L2", "L3"):
        raise ValueError(
            f"TARGET_CACHE_TIER must be L2 or L3, got {TARGET_CACHE_TIER}"
        )

    if RESET_RESULT:
        RESULT_FILE.unlink(missing_ok=True)

    runtime_warmup(BASE_URL, VOCAB_SIZE)

    print(
        f"POLICY={POLICY}, tier={TARGET_CACHE_TIER}, "
        f"prefix={PREFIX_LENGTHS}, sessions/len={SESSIONS_PER_LEN}, "
        f"concurrency={MAX_CONCURRENCY}, rate={REQUEST_RATE_RAW}, "
        f"trials={TRIALS}, evictors={NUM_EVICTORS}x{EVICTOR_LEN}, "
        f"clear_l3={CLEAR_L3}, validate_tier={VALIDATE_CACHE_TIER}, "
        f"result={RESULT_FILE}"
    )

    for trial in range(TRIALS):
        for prefix_len in PREFIX_LENGTHS:
            await run_case(prefix_len, trial)


if __name__ == "__main__":
    asyncio.run(main())