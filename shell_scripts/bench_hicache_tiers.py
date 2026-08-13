# 用于初步对比recompute\restore耗时
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

def refresh_hicache_metrics(seed: int = 987654):
    """
    hicache_host_used_tokens 是周期更新的 Gauge，
    /metrics 本身不会主动读取 Host Pool。

    因此发送一个不共享前缀的短 Prefill 请求，
    强制 Scheduler 执行 report_prefill_stats()，
    从而调用 _log_hicache_stats() 更新 Host Pool Gauge。

    注意：
    这个 probe 自己产生的 Host Backup 在该次 Prefill
    metrics 采样之后才发生，因此当前采样主要反映 probe
    开始前已经存在的 Host KV。
    """

    probe_ids = make_random_ids(
        128,
        seed=seed,
    )

    generate(
        probe_ids,
        f"metrics_refresh_{seed}",
    )

    time.sleep(0.2)

    return snapshot_metrics()

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
        "(host_used may be stale):",
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
    # 2. 等待 Cold Request 的异步 Write-through
    # -------------------------------------------------

    # HiCache CPU write-through 是异步执行的。
    # 给 DMA / ack 一点处理时间。
    time.sleep(1.0)

    # 主动制造一次 Prefill，让 Prometheus 中的
    # hicache_host_used_tokens 更新。
    metrics_after_backup = refresh_hicache_metrics(
        seed=888001
    )

    host_used = metrics_after_backup["host_used"]

    print(
        "Metrics after target backup refresh:",
        metrics_after_backup,
    )

    if host_used <= 0:
        raise RuntimeError(
            "Host KV is still zero after an explicit metrics refresh. "
            "Now this is likely a real L2 backup problem, not a stale "
            "Prometheus gauge."
        )

    print(
        f"Host backup observed after metrics refresh: "
        f"host_used={host_used}"
    )

    # -------------------------------------------------
    # 3. 再请求一次 Target，验证 L1 Hit
    # -------------------------------------------------

    l1 = generate(
        target_ids,
        "l1_hit",
    )

    time.sleep(0.2)

    if (
            l1["cached_tokens"] is not None
            and l1["cached_tokens"] <= 0
    ):
        raise RuntimeError(
            "The second target request did not hit L1 cache."
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
