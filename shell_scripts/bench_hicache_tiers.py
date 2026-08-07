import json
import os
import random
import re
import time
from pathlib import Path

import requests
from transformers import AutoConfig

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]

TARGET_LEN = int(os.getenv("TARGET_LEN", "8192"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))

RESULT_FILE = Path(
    os.getenv(
        "RESULT_FILE",
        "/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/tier_results.jsonl",
    )
)

TIMEOUT = 300

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

VOCAB_SIZE = config.vocab_size

if VOCAB_SIZE < 2000:
    raise RuntimeError(f"Unexpected vocab size: {VOCAB_SIZE}")


def make_random_ids(length: int, seed: int):
    """
    生成彼此不共享前缀的合法 token id。
    不使用自然语言是为了避免不同干扰请求意外共享第一页 Radix Prefix。
    """
    rng = random.Random(seed)

    low = 1000
    high = min(VOCAB_SIZE - 1, 50000)

    return [
        rng.randrange(low, high)
        for _ in range(length)
    ]


def flush_cache():
    resp = requests.post(
        f"{BASE_URL}/flush_cache",
        timeout=30,
    )
    resp.raise_for_status()

    # 等待 scheduler 处理 flush
    time.sleep(1.0)


def get_metrics_text():
    resp = requests.get(
        f"{BASE_URL}/metrics",
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def get_metric(name: str) -> float:
    """
    对同名、不同 label 的 Prometheus metric 求和。
    """
    text = get_metrics_text()
    values = []

    pattern = re.compile(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+"
        rf"([-+0-9.eE]+)$"
    )

    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            values.append(float(m.group(1)))

    return sum(values) if values else 0.0


def snapshot_metrics():
    return {
        "host_used": get_metric(
            "sglang:hicache_host_used_tokens"
        ),
        "host_total": get_metric(
            "sglang:hicache_host_total_tokens"
        ),
        "evicted": get_metric(
            "sglang:evicted_tokens_total"
        ),
        "load_back": get_metric(
            "sglang:load_back_tokens_total"
        ),
    }


def generate(input_ids, name: str):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
        },
        "stream": False,
    }

    begin = time.perf_counter()

    resp = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        timeout=TIMEOUT,
    )

    elapsed_ms = (
                         time.perf_counter() - begin
                 ) * 1000.0

    resp.raise_for_status()

    body = resp.json()

    if isinstance(body, list):
        body = body[0]

    meta = body.get("meta_info", {})

    cached_tokens = meta.get(
        "cached_tokens",
        None,
    )

    server_e2e = meta.get(
        "e2e_latency",
        None,
    )

    server_e2e_ms = (
        server_e2e * 1000.0
        if server_e2e is not None
        else None
    )

    result = {
        "name": name,
        "target_len": TARGET_LEN,
        "client_latency_ms": elapsed_ms,
        "server_e2e_ms": server_e2e_ms,
        "cached_tokens": cached_tokens,
        "text": body.get("text", ""),
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
        )
    )

    return result


def wait_host_backup(
        old_used: float,
        timeout=10,
):
    deadline = time.time() + timeout

    while time.time() < deadline:
        cur = get_metric(
            "sglang:hicache_host_used_tokens"
        )

        if cur > old_used:
            return cur

        time.sleep(0.5)

    return get_metric(
        "sglang:hicache_host_used_tokens"
    )


def append_result(row):
    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_FILE.open(
            "a",
            encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    print(
        f"""
===== CONFIG =====
BASE_URL      = {BASE_URL}
MODEL_PATH    = {MODEL_PATH}
TARGET_LEN    = {TARGET_LEN}
EVICTOR_LEN   = {EVICTOR_LEN}
NUM_EVICTORS  = {NUM_EVICTORS}
VOCAB_SIZE    = {VOCAB_SIZE}
==================
"""
    )

    # -------------------------------------------------
    # 0. 运行时热身，避免第一次请求包含额外初始化开销
    # -------------------------------------------------

    runtime_warmup = make_random_ids(
        128,
        seed=999999,
    )

    generate(
        runtime_warmup,
        "runtime_warmup",
    )

    flush_cache()

    base_metrics = snapshot_metrics()

    print(
        "Metrics after flush:",
        base_metrics,
    )

    if base_metrics["host_total"] <= 0:
        raise RuntimeError(
            "Host KV cache capacity is zero. "
            "HiCache may not be enabled."
        )

    # -------------------------------------------------
    # 1. Cold Recompute
    # -------------------------------------------------

    target_ids = make_random_ids(
        TARGET_LEN,
        seed=20260807,
    )

    cold = generate(
        target_ids,
        "cold_recompute",
    )

    time.sleep(1)

    metrics_after_cold = snapshot_metrics()

    # -------------------------------------------------
    # 2. L1 Hit
    #
    # 这里同时再次访问一次 target。
    # 对 chunked prefill，write-through 的 hit_count
    # 在 chunked 中不会更新，因此重复访问有助于确保
    # 对应节点真正完成 Host Backup。
    # -------------------------------------------------

    l1 = generate(
        target_ids,
        "l1_hit",
    )

    # 等待异步 write-through 完成
    host_used = wait_host_backup(
        base_metrics["host_used"],
        timeout=10,
    )

    # 某些节点如果还没有完成 Host Backup，
    # 再访问一次并等待。
    warm_round = 0

    while (
            host_used <= base_metrics["host_used"]
            and warm_round < 3
    ):
        warm_round += 1

        print(
            f"Host backup not observed, "
            f"extra warm round={warm_round}"
        )

        generate(
            target_ids,
            f"extra_l1_warm_{warm_round}",
        )

        host_used = wait_host_backup(
            base_metrics["host_used"],
            timeout=10,
        )

    if host_used <= base_metrics["host_used"]:
        raise RuntimeError(
            "Target KV has not been observed in Host KV cache. "
            "Do NOT continue the L2 experiment."
        )

    print(
        f"Host backup confirmed. "
        f"host_used={host_used}"
    )

    metrics_before_pressure = snapshot_metrics()

    # -------------------------------------------------
    # 3. 制造 GPU KV Cache 压力
    #
    # 当前 GPU KV capacity = 633280 tokens。
    #
    # 24 * 30000 = 720000 tokens，
    # 已经超过当前 L1 容量。
    # -------------------------------------------------

    for i in range(NUM_EVICTORS):

        ids = make_random_ids(
            EVICTOR_LEN,
            seed=100000 + i,
        )

        generate(
            ids,
            f"evictor_{i:02d}",
        )

        if (i + 1) % 4 == 0:
            cur = snapshot_metrics()

            print(
                f"[pressure {i + 1}/{NUM_EVICTORS}] "
                f"{cur}"
            )

    metrics_after_pressure = snapshot_metrics()

    evicted_delta = (
            metrics_after_pressure["evicted"]
            - metrics_before_pressure["evicted"]
    )

    print(
        f"Evicted delta = {evicted_delta}"
    )

    if evicted_delta <= 0:
        raise RuntimeError(
            "No GPU KV eviction happened. "
            "Increase NUM_EVICTORS or EVICTOR_LEN."
        )

    # -------------------------------------------------
    # 4. 再次访问 Target
    #
    # 如果 Target 已经从 L1 淘汰但 L2 仍在，
    # SGLang 应调用 load_back。
    # -------------------------------------------------

    load_back_before = get_metric(
        "sglang:load_back_tokens_total"
    )

    l2 = generate(
        target_ids,
        "l2_revisit",
    )

    time.sleep(0.5)

    load_back_after = get_metric(
        "sglang:load_back_tokens_total"
    )

    load_back_delta = (
            load_back_after
            - load_back_before
    )

    final_metrics = snapshot_metrics()

    print(
        f"""
========== RESULT ==========
Cold latency : {cold['client_latency_ms']:.3f} ms
L1 latency   : {l1['client_latency_ms']:.3f} ms
L2 latency   : {l2['client_latency_ms']:.3f} ms

Cold cached  : {cold['cached_tokens']}
L1 cached    : {l1['cached_tokens']}
L2 cached    : {l2['cached_tokens']}

Evicted delta   : {evicted_delta}
Load-back delta : {load_back_delta}

Final metrics:
{final_metrics}
============================
"""
    )

    rows = [
        {
            **cold,
            "path": "cold",
            "evicted_delta": 0,
            "load_back_delta": 0,
        },
        {
            **l1,
            "path": "l1",
            "evicted_delta": 0,
            "load_back_delta": 0,
        },
        {
            **l2,
            "path": "l2",
            "evicted_delta": evicted_delta,
            "load_back_delta": load_back_delta,
        },
    ]

    for row in rows:
        append_result(row)

    if load_back_delta <= 0:
        print(
            "\n[FAIL] L2 Restore was NOT confirmed."
        )
        print(
            "Increase NUM_EVICTORS, e.g. NUM_EVICTORS=32."
        )
        raise SystemExit(2)

    print(
        "\n[PASS] Cold / L1 Hit / L2 Restore "
        "three paths are confirmed."
    )


if __name__ == "__main__":
    main()
