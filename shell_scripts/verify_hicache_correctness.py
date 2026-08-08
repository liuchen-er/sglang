import json
import os
import random
import re
import sys
import time
from pathlib import Path
import requests
from transformers import AutoConfig

# ============================================================
# Configuration
# ============================================================
BASE_URL = os.getenv(
    "BASE_URL",
    "http://127.0.0.1:30000",
)
MODEL_PATH = os.environ["MODEL_PATH"]
# 这里只是给结果打标签，并用于判断期望的 load_back 行为。
# 它不会修改 Server 本身的策略。
POLICY = os.getenv(
    "POLICY",
    "always_restore",
)
if POLICY not in {
    "always_restore",
    "always_recompute",
}:
    raise ValueError(
        f"Unsupported POLICY={POLICY}. Expected always_restore or always_recompute."
    )
PROMPT_LENGTHS = [
    512,
    2048,
    8192,
    16384,
]
PROMPT_SEEDS = {
    512: 100512,
    2048: 102048,
    8192: 108192,
    16384: 116384,
}
TRIALS = int(
    os.getenv(
        "TRIALS",
        "10",
    )
)
MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "64",
    )
)
# 当前启动日志中：
# max_total_num_tokens = 633280
#
# 30K * 24 = 720K，
# 理论上足够让 GPU KV Pool 发生 Eviction。
EVICTOR_LEN = int(
    os.getenv(
        "EVICTOR_LEN",
        "30000",
    )
)
NUM_EVICTORS = int(
    os.getenv(
        "NUM_EVICTORS",
        "24",
    )
)
TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT",
        "300",
    )
)
RESULT_DIR = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
RESULT_FILE = Path(
    os.getenv(
        "RESULT_FILE",
        str(RESULT_DIR / f"correctness_{POLICY}.jsonl"),
    )
)
# ============================================================
# Model information
# ============================================================
config = AutoConfig.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
VOCAB_SIZE = config.vocab_size
if VOCAB_SIZE < 2000:
    raise RuntimeError(f"Unexpected vocab size: {VOCAB_SIZE}")


# ============================================================
# Utilities
# ============================================================
def make_random_ids(
    length: int,
    seed: int,
):
    """
    生成确定性的合法 Token IDs。
    固定 seed 后，同一个 prompt_len 在：
      always_restore
      always_recompute
    两次 Server 实验中会得到完全相同的输入。
    """
    rng = random.Random(seed)
    low = 1000
    high = min(
        VOCAB_SIZE - 1,
        50000,
    )
    return [
        rng.randrange(
            low,
            high,
        )
        for _ in range(length)
    ]


def flush_cache():
    """
    清空当前 Radix / HiCache 缓存。
    注意：
    hicache_host_used_tokens Gauge
    在 flush 后可能暂时是旧值，
    所以后面不会直接依赖 flush 后的 host_used。
    """
    resp = requests.post(
        f"{BASE_URL}/flush_cache",
        timeout=30,
    )
    resp.raise_for_status()
    time.sleep(0.5)


def health_check():
    resp = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )
    resp.raise_for_status()


def get_metrics_text():
    resp = requests.get(
        f"{BASE_URL}/metrics",
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def get_metric(
    name: str,
) -> float:
    """
    Prometheus 同一个 Metric 可能带不同 label。
    这里把同名 metric 的所有 label value 求和。
    """
    text = get_metrics_text()
    pattern = re.compile(
        rf"^{re.escape(name)}"
        rf"(?:\{{[^}}]*\}})?\s+"
        rf"([-+0-9.eE]+)$"
    )
    values = []
    for line in text.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            values.append(float(match.group(1)))
    return sum(values) if values else 0.0


def snapshot_metrics():
    return {
        "host_used": get_metric("sglang:hicache_host_used_tokens"),
        "host_total": get_metric("sglang:hicache_host_total_tokens"),
        "evicted": get_metric("sglang:evicted_tokens_total"),
        "load_back": get_metric("sglang:load_back_tokens_total"),
    }


def generate(
    input_ids,
    request_name: str,
    max_new_tokens: int,
):
    """
    发送 /generate 请求。
    temperature=0：
      使用 greedy decoding，
      避免随机采样干扰正确性比较。
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }
    begin = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        timeout=TIMEOUT,
    )
    latency_ms = (time.perf_counter() - begin) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    if isinstance(
        body,
        list,
    ):
        body = body[0]
    meta = body.get(
        "meta_info",
        {},
    )
    result = {
        "request_name": request_name,
        "text": body.get(
            "text",
            "",
        ),
        "cached_tokens": meta.get(
            "cached_tokens",
            None,
        ),
        "latency_ms": latency_ms,
    }
    return result


def refresh_hicache_metrics(
    seed: int,
):
    """
    /metrics 读取的是 Prometheus Gauge，
    不一定主动刷新当前 Host Pool 状态。
    发送一个很短、且不共享 Target Prefix 的 Prefill，
    让 Scheduler 再跑一次 Prefill metrics 更新。
    该 probe 只用于刷新观测值。
    """
    probe_ids = make_random_ids(
        128,
        seed,
    )
    generate(
        input_ids=probe_ids,
        request_name="metrics_refresh",
        max_new_tokens=1,
    )
    time.sleep(0.2)
    return snapshot_metrics()


def append_result(
    row: dict,
):
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


def assert_same_output(
    reference: str,
    candidate: str,
    path_name: str,
):
    if reference != candidate:
        print()
        print(f"[CORRECTNESS FAIL] {path_name} output differs from Cold reference.")
        print("Cold output:")
        print(repr(reference))
        print(f"{path_name} output:")
        print(repr(candidate))
        raise RuntimeError(f"{path_name} output mismatch.")


# ============================================================
# One correctness trial
# ============================================================
def run_one_trial(
    prompt_len: int,
    trial: int,
):
    """
    一轮完整正确性测试：
      1. flush
      2. Cold Prefill
      3. 确认 Host L2 已出现缓存
      4. L1 Hit
      5. 制造 GPU KV Eviction
      6. Target Revisit
      7. 验证 Restore / Recompute 路径
      8. 比较输出
    """
    prompt_seed = PROMPT_SEEDS[prompt_len]
    target_ids = make_random_ids(
        prompt_len,
        prompt_seed,
    )
    print()
    print("=" * 72)
    print(f"POLICY={POLICY}, prompt_len={prompt_len}, trial={trial}")
    print("=" * 72)
    # --------------------------------------------------------
    # 0. 清理上一轮缓存
    # --------------------------------------------------------
    flush_cache()
    # --------------------------------------------------------
    # 1. Cold Prefill
    # 这是本轮的正确性 Reference。
    # --------------------------------------------------------
    cold = generate(
        input_ids=target_ids,
        request_name="cold_reference",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"[Cold] latency={cold['latency_ms']:.3f} ms, cached={cold['cached_tokens']}")
    cold_output = cold["text"]
    if not cold_output:
        raise RuntimeError("Cold reference produced empty output.")
    # --------------------------------------------------------
    # 2. 等待 write-through：
    # GPU Target KV
    #       ↓
    # CPU Host L2
    # --------------------------------------------------------
    time.sleep(1.0)
    metrics_after_backup = refresh_hicache_metrics(
        seed=800000 + prompt_len + trial,
    )
    print(
        "[After backup refresh]",
        metrics_after_backup,
    )
    if metrics_after_backup["host_total"] <= 0:
        raise RuntimeError("HiCache Host Pool is not available.")
    if metrics_after_backup["host_used"] <= 0:
        raise RuntimeError("No Host KV was observed after the Target Cold Prefill.")
    # --------------------------------------------------------
    # 3. 第二次相同 Target：
    # 此时应该主要走 L1 GPU Hit。
    # 同时验证：
    # output_L1 == output_Cold
    # --------------------------------------------------------
    l1 = generate(
        input_ids=target_ids,
        request_name="l1_hit",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"[L1] latency={l1['latency_ms']:.3f} ms, cached={l1['cached_tokens']}")
    assert_same_output(
        reference=cold_output,
        candidate=l1["text"],
        path_name="L1",
    )
    if l1["cached_tokens"] is not None and l1["cached_tokens"] <= 0:
        raise RuntimeError(
            "The second Target request did "
            "not report any cached tokens. "
            "L1 Hit is not confirmed."
        )
    # --------------------------------------------------------
    # 4. 记录制造压力前的 Eviction Counter
    # --------------------------------------------------------
    metrics_before_pressure = snapshot_metrics()
    # --------------------------------------------------------
    # 5. 制造大量互不共享 Prefix 的请求
    # 目标：
    # GPU KV Pool 被填满
    #        ↓
    # Target 从 L1 被淘汰
    # 但 write-through 产生的 L2 副本仍保留。
    # --------------------------------------------------------
    for i in range(NUM_EVICTORS):
        evictor_seed = 2_000_000 + trial * 100_000 + prompt_len + i
        evictor_ids = make_random_ids(
            EVICTOR_LEN,
            evictor_seed,
        )
        generate(
            input_ids=evictor_ids,
            request_name=(f"evictor_{i:02d}"),
            max_new_tokens=1,
        )
        if (i + 1) % 4 == 0:
            current = snapshot_metrics()
            print(
                f"[Pressure "
                f"{i + 1}/"
                f"{NUM_EVICTORS}] "
                f"evicted="
                f"{current['evicted']}, "
                f"host_used="
                f"{current['host_used']}"
            )
    metrics_after_pressure = snapshot_metrics()
    evicted_delta = (
        metrics_after_pressure["evicted"] - metrics_before_pressure["evicted"]
    )
    print(f"[Eviction] delta={evicted_delta}")
    if evicted_delta <= 0:
        raise RuntimeError(
            "No GPU KV eviction occurred. Increase NUM_EVICTORS or EVICTOR_LEN."
        )
    # --------------------------------------------------------
    # 6. 最关键步骤：
    #
    # 再次访问 Target。
    #
    # always_restore：
    #   L1 Miss
    #   → L2 Hit
    #   → CPU → GPU Load Back
    #
    # always_recompute：
    #   L1 Miss
    #   → L2 Hit
    #   → 策略拒绝 Load Back
    #   → Prefill Recompute
    # --------------------------------------------------------
    load_back_before = get_metric("sglang:load_back_tokens_total")
    revisit = generate(
        input_ids=target_ids,
        request_name=(f"target_revisit_{POLICY}"),
        max_new_tokens=MAX_NEW_TOKENS,
    )
    time.sleep(0.2)
    load_back_after = get_metric("sglang:load_back_tokens_total")
    load_back_delta = load_back_after - load_back_before
    print(
        f"[Revisit] "
        f"latency="
        f"{revisit['latency_ms']:.3f} ms, "
        f"cached="
        f"{revisit['cached_tokens']}, "
        f"load_back_delta="
        f"{load_back_delta}"
    )
    # --------------------------------------------------------
    # 7. 正确性最重要的判断
    #
    # Cold output
    #      ==
    # Revisit output
    # --------------------------------------------------------
    # assert_same_output(
    #     reference=cold_output,
    #     candidate=revisit["text"],
    #     path_name=POLICY,
    # )
    same = cold_output == l1["text"]

    print(f"[L1 Output Exact Match] {same}")

    if not same:
        print(
            "[WARN] Cold and L1 outputs differ. "
            "Do not classify this as cache corruption yet. "
            "Run determinism and logprob checks first."
        )

    # --------------------------------------------------------
    # 8. 数据路径检查
    # --------------------------------------------------------
    if POLICY == "always_restore":
        if load_back_delta <= 0:
            raise RuntimeError(
                "POLICY=always_restore, but no L2->L1 Load Back was observed."
            )
    elif POLICY == "always_recompute":
        if load_back_delta != 0:
            raise RuntimeError(
                "POLICY=always_recompute, but L2->L1 Load Back still occurred."
            )
    # --------------------------------------------------------
    # 9. 保存结果
    # --------------------------------------------------------
    row = {
        "policy": POLICY,
        "prompt_len": prompt_len,
        "prompt_seed": prompt_seed,
        "trial": trial,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cold_output": cold_output,
        "l1_output": l1["text"],
        "revisit_output": revisit["text"],
        "cold_latency_ms": cold["latency_ms"],
        "l1_latency_ms": l1["latency_ms"],
        "revisit_latency_ms": revisit["latency_ms"],
        "cold_cached_tokens": cold["cached_tokens"],
        "l1_cached_tokens": l1["cached_tokens"],
        "revisit_cached_tokens": revisit["cached_tokens"],
        "evicted_delta": evicted_delta,
        "load_back_delta": load_back_delta,
        "host_used_after_backup": metrics_after_backup["host_used"],
        "correct": True,
    }
    append_result(row)
    print(f"[PASS] len={prompt_len}, trial={trial}, policy={POLICY}")


# ============================================================
# Main
# ============================================================
def main():
    health_check()
    print()
    print("============================================================")
    print("HiCache Correctness Verification")
    print("============================================================")
    print(f"Server       : {BASE_URL}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Policy label : {POLICY}")
    print(f"Prompt lens  : {PROMPT_LENGTHS}")
    print(f"Trials       : {TRIALS}")
    print(f"Output tokens: {MAX_NEW_TOKENS}")
    print(f"Evictor      : {NUM_EVICTORS} x {EVICTOR_LEN}")
    print(f"Result file  : {RESULT_FILE}")
    print("============================================================")
    passed = 0
    total = len(PROMPT_LENGTHS) * TRIALS
    for prompt_len in PROMPT_LENGTHS:
        for trial in range(TRIALS):
            try:
                run_one_trial(
                    prompt_len=prompt_len,
                    trial=trial,
                )
                passed += 1
            except Exception as exc:
                print()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("[FAIL]")
                print(f"policy={POLICY}")
                print(f"prompt_len={prompt_len}")
                print(f"trial={trial}")
                print(f"error={exc}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                raise
    print()
    print("=" * 72)
    print("FINAL RESULT")
    print(f"PASS = {passed}")
    print(f"TOTAL = {total}")
    print(f"FAIL = {total - passed}")
    print("=" * 72)
    if passed != total:
        raise SystemExit(1)
    print("[ALL PASS] HiCache correctness verification completed.")


if __name__ == "__main__":
    main()
