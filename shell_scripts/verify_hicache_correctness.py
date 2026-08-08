import json
import os
import random
import re
import time
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

# ==================== Configuration ====================

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")

if POLICY not in {"always_restore", "always_recompute"}:
    raise ValueError(
        f"Unsupported POLICY={POLICY}. Expected always_restore or always_recompute."
    )


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


PROMPT_LENGTHS = parse_int_list(os.getenv("PROMPT_LENGTHS", "512,2048,8192,16384"))
TRIALS = int(os.getenv("TRIALS", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "64"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
CHECK_COLD_DETERMINISM = os.getenv("CHECK_COLD_DETERMINISM", "1") == "1"
STRICT_TEXT_MATCH = os.getenv("STRICT_TEXT_MATCH", "0") == "1"
BACKUP_WAIT_SECONDS = float(os.getenv("BACKUP_WAIT_SECONDS", "1.0"))
MIN_HOST_RATIO = float(os.getenv("MIN_HOST_RATIO", "0.75"))

RESULT_DIR = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = Path(
    os.getenv("RESULT_FILE", str(RESULT_DIR / f"correctness_{POLICY}.jsonl"))
)

# ==================== Model / Tokenizer ====================

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size

if VOCAB_SIZE < 2000:
    raise RuntimeError(f"Unexpected vocab size: {VOCAB_SIZE}")

# ==================== Prompt ====================

CORRECTNESS_TEXT = """
Large language model inference systems use KV cache to avoid recomputing the key and value tensors of previously processed tokens.
During autoregressive generation, the model stores attention states for earlier tokens so that subsequent decoding steps only need to process newly generated tokens.
Hierarchical KV cache extends GPU memory with host memory. Frequently used KV states can remain in GPU memory, while colder states can be stored in CPU memory and restored when the same context is reused.
A correct hierarchical cache implementation must guarantee that restoring cached key and value tensors produces the same logical model state as recomputing the corresponding prefix.
Modern inference engines also use continuous batching, paged memory management, prefix caching, CUDA Graphs, chunked prefill, and optimized attention kernels.
The purpose of this text is to create a deterministic and meaningful long-context input for validating large language model inference systems.
"""


def build_natural_prompt_ids(target_len: int):
    text = (CORRECTNESS_TEXT + "\n") * 64
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_len:
        text += (CORRECTNESS_TEXT + "\n") * 32
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_len]


def make_random_ids(length: int, seed: int):
    rng = random.Random(seed)
    low, high = 1000, min(VOCAB_SIZE - 1, 50000)
    return [rng.randrange(low, high) for _ in range(length)]


# ==================== HTTP / Metrics ====================


def health_check():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()


def flush_cache():
    resp = requests.post(f"{BASE_URL}/flush_cache", timeout=30)
    resp.raise_for_status()
    time.sleep(0.5)


def get_metrics_text():
    resp = requests.get(f"{BASE_URL}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.text


def get_metric(name: str) -> float:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    values = []
    for line in get_metrics_text().splitlines():
        match = pattern.match(line.strip())
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


# ==================== Generate ====================


def generate(
    input_ids, request_name: str, max_new_tokens: int, return_logprob: bool = True
):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }

    if return_logprob:
        payload["return_logprob"] = True
        payload["top_logprobs_num"] = 5

    begin = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=TIMEOUT)
    latency_ms = (time.perf_counter() - begin) * 1000.0
    resp.raise_for_status()

    body = resp.json()
    if isinstance(body, list):
        body = body[0]
    meta = body.get("meta_info", {})

    return {
        "request_name": request_name,
        "text": body.get("text", ""),
        "cached_tokens": meta.get("cached_tokens"),
        "latency_ms": latency_ms,
        "output_token_logprobs": meta.get("output_token_logprobs"),
        "output_top_logprobs": meta.get("output_top_logprobs"),
    }


def refresh_hicache_metrics(seed: int):
    probe_ids = make_random_ids(128, seed)
    generate(probe_ids, "metrics_refresh", 1, return_logprob=False)
    time.sleep(0.2)
    return snapshot_metrics()


# ==================== Output comparison ====================


def first_difference(a: str, b: str):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return {"index": i, "a_char": repr(a[i]), "b_char": repr(b[i])}
    if len(a) != len(b):
        return {
            "index": limit,
            "a_char": "<END>" if limit >= len(a) else repr(a[limit]),
            "b_char": "<END>" if limit >= len(b) else repr(b[limit]),
        }
    return None


def compare_outputs(reference, candidate, path_name: str, exact_match_required: bool):
    same = reference["text"] == candidate["text"]
    print(f"[{path_name} exact text match] {same}")
    if same:
        return True

    print(f"[WARN] {path_name} output differs from Cold reference.")
    print(f"First difference: {first_difference(reference['text'], candidate['text'])}")
    print("Cold:", repr(reference["text"]))
    print(f"{path_name}:", repr(candidate["text"]))

    if reference["output_top_logprobs"] is not None:
        print("Cold output_top_logprobs:")
        print(
            json.dumps(
                reference["output_top_logprobs"][:3],
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    if candidate["output_top_logprobs"] is not None:
        print(f"{path_name} output_top_logprobs:")
        print(
            json.dumps(
                candidate["output_top_logprobs"][:3],
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    if exact_match_required:
        raise RuntimeError(
            f"{path_name} output differs from deterministic Cold reference."
        )
    return False


def append_result(row: dict):
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ==================== Cold determinism ====================


def check_cold_determinism(prompt_len: int, target_ids):
    if not CHECK_COLD_DETERMINISM:
        print("[Determinism] check disabled")
        return False

    print(f"\n[Determinism] checking prompt_len={prompt_len}")
    flush_cache()
    cold_a = generate(target_ids, "cold_det_a", MAX_NEW_TOKENS)
    flush_cache()
    cold_b = generate(target_ids, "cold_det_b", MAX_NEW_TOKENS)
    same = cold_a["text"] == cold_b["text"]

    print(f"[Determinism] Cold A == Cold B: {same}")
    if not same:
        print("[Determinism WARN] Cold path itself is not text deterministic.")
        print("Cold A:", repr(cold_a["text"]))
        print("Cold B:", repr(cold_b["text"]))
    return same


# ==================== One trial ====================


def run_one_trial(prompt_len: int, trial: int, target_ids, cold_is_deterministic: bool):
    print("\n" + "=" * 72)
    print(f"POLICY={POLICY}, prompt_len={prompt_len}, trial={trial}")
    print("=" * 72)

    exact_match_required = STRICT_TEXT_MATCH or cold_is_deterministic
    flush_cache()

    # 1. Cold Prefill
    cold = generate(target_ids, "cold_reference", MAX_NEW_TOKENS)
    print(f"[Cold] latency={cold['latency_ms']:.3f} ms, cached={cold['cached_tokens']}")
    if not cold["text"]:
        raise RuntimeError("Cold reference generated empty output.")

    # 2. 等待 write-through 到 Host
    time.sleep(BACKUP_WAIT_SECONDS)
    metrics_after_backup = refresh_hicache_metrics(800000 + prompt_len * 10 + trial)
    print(f"[After backup refresh] {metrics_after_backup}")

    if metrics_after_backup["host_total"] <= 0:
        raise RuntimeError("HiCache Host Pool is not available.")

    min_expected_host = max(PAGE_SIZE, int(prompt_len * MIN_HOST_RATIO))
    if metrics_after_backup["host_used"] < min_expected_host:
        raise RuntimeError(
            f"Target Host backup is not sufficiently observed. "
            f"host_used={metrics_after_backup['host_used']}, expected_at_least={min_expected_host}"
        )

    # 3. L1 Hit
    load_back_before_l1 = get_metric("sglang:load_back_tokens_total")
    l1 = generate(target_ids, "l1_hit", MAX_NEW_TOKENS)
    load_back_after_l1 = get_metric("sglang:load_back_tokens_total")
    l1_load_back_delta = load_back_after_l1 - load_back_before_l1

    print(
        f"[L1] latency={l1['latency_ms']:.3f} ms, "
        f"cached={l1['cached_tokens']}, load_back_delta={l1_load_back_delta}"
    )

    if l1["cached_tokens"] is not None and l1["cached_tokens"] <= 0:
        raise RuntimeError("Second Target request did not hit GPU prefix cache.")
    if l1_load_back_delta > 0:
        raise RuntimeError("L1 verification unexpectedly triggered Host->GPU LoadBack.")

    l1_exact_match = compare_outputs(cold, l1, "L1", exact_match_required)
    metrics_before_pressure = snapshot_metrics()

    # 4. 制造 GPU KV 压力
    for i in range(NUM_EVICTORS):
        evictor_seed = 2_000_000 + trial * 100_000 + prompt_len * 10 + i
        evictor_ids = make_random_ids(EVICTOR_LEN, evictor_seed)
        generate(evictor_ids, f"evictor_{i:02d}", 1, return_logprob=False)

        if (i + 1) % 4 == 0:
            cur = snapshot_metrics()
            print(
                f"[Pressure {i + 1}/{NUM_EVICTORS}] "
                f"evicted={cur['evicted']}, host_used={cur['host_used']}, load_back={cur['load_back']}"
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

    # 5. Target Revisit
    load_back_before = get_metric("sglang:load_back_tokens_total")
    revisit = generate(target_ids, f"target_revisit_{POLICY}", MAX_NEW_TOKENS)
    time.sleep(0.2)
    load_back_after = get_metric("sglang:load_back_tokens_total")
    load_back_delta = load_back_after - load_back_before

    print(
        f"[Revisit] latency={revisit['latency_ms']:.3f} ms, "
        f"cached={revisit['cached_tokens']}, load_back_delta={load_back_delta}"
    )

    # 6. 数据路径检查
    if POLICY == "always_restore":
        if load_back_delta <= 0:
            raise RuntimeError(
                "POLICY=always_restore, but Target revisit did not trigger L2->L1 LoadBack."
            )

    elif POLICY == "always_recompute":
        if load_back_delta != 0:
            raise RuntimeError(
                "POLICY=always_recompute, but Target revisit still triggered L2->L1 LoadBack."
            )

        if (
            revisit["cached_tokens"] is not None
            and l1["cached_tokens"] is not None
            and revisit["cached_tokens"] >= l1["cached_tokens"]
        ):
            raise RuntimeError(
                "always_recompute revisit still reports an L1-sized prefix hit. "
                "Target may not have been evicted from GPU. Increase NUM_EVICTORS."
            )

    # 7. Output correctness
    revisit_exact_match = compare_outputs(cold, revisit, POLICY, exact_match_required)

    row = {
        "policy": POLICY,
        "prompt_len": prompt_len,
        "trial": trial,
        "cold_is_deterministic": cold_is_deterministic,
        "strict_text_match": STRICT_TEXT_MATCH,
        "cold_output": cold["text"],
        "l1_output": l1["text"],
        "revisit_output": revisit["text"],
        "l1_exact_match": l1_exact_match,
        "revisit_exact_match": revisit_exact_match,
        "cold_latency_ms": cold["latency_ms"],
        "l1_latency_ms": l1["latency_ms"],
        "revisit_latency_ms": revisit["latency_ms"],
        "cold_cached_tokens": cold["cached_tokens"],
        "l1_cached_tokens": l1["cached_tokens"],
        "revisit_cached_tokens": revisit["cached_tokens"],
        "host_used_after_backup": metrics_after_backup["host_used"],
        "evicted_delta": evicted_delta,
        "l1_load_back_delta": l1_load_back_delta,
        "revisit_load_back_delta": load_back_delta,
        "cold_output_top_logprobs": cold["output_top_logprobs"],
        "l1_output_top_logprobs": l1["output_top_logprobs"],
        "revisit_output_top_logprobs": revisit["output_top_logprobs"],
    }
    append_result(row)

    print(f"[PASS] policy={POLICY}, len={prompt_len}, trial={trial}")
    if not l1_exact_match:
        print(
            "[PASS WITH WARNING] L1 text differs from Cold, but control-path checks passed."
        )
    if not revisit_exact_match:
        print(
            "[PASS WITH WARNING] Revisit text differs from Cold, but HiCache control-path checks passed."
        )
    return row


# ==================== Main ====================


def main():
    health_check()

    print("=" * 72)
    print("HiCache Correctness Verification")
    print("=" * 72)
    print(f"Server            : {BASE_URL}")
    print(f"Model             : {MODEL_PATH}")
    print(f"Policy            : {POLICY}")
    print(f"Prompt lengths    : {PROMPT_LENGTHS}")
    print(f"Trials            : {TRIALS}")
    print(f"Max new tokens    : {MAX_NEW_TOKENS}")
    print(f"Page size         : {PAGE_SIZE}")
    print(f"Evictor           : {NUM_EVICTORS} x {EVICTOR_LEN}")
    print(f"Cold determinism  : {CHECK_COLD_DETERMINISM}")
    print(f"Strict text match : {STRICT_TEXT_MATCH}")
    print(f"Result file       : {RESULT_FILE}")
    print("=" * 72)

    targets = {}
    for prompt_len in PROMPT_LENGTHS:
        target_ids = build_natural_prompt_ids(prompt_len)
        if len(target_ids) != prompt_len:
            raise RuntimeError(
                f"Target length mismatch: expected={prompt_len}, actual={len(target_ids)}"
            )
        targets[prompt_len] = target_ids

    determinism = {}
    for prompt_len in PROMPT_LENGTHS:
        determinism[prompt_len] = check_cold_determinism(
            prompt_len, targets[prompt_len]
        )

    passed = 0
    warning_count = 0
    total = len(PROMPT_LENGTHS) * TRIALS

    for prompt_len in PROMPT_LENGTHS:
        for trial in range(TRIALS):
            try:
                row = run_one_trial(
                    prompt_len=prompt_len,
                    trial=trial,
                    target_ids=targets[prompt_len],
                    cold_is_deterministic=determinism[prompt_len],
                )
                passed += 1
                warning_count += int(not row["l1_exact_match"])
                warning_count += int(not row["revisit_exact_match"])
            except Exception as exc:
                print("\n" + "!" * 72)
                print("[FAIL]")
                print(f"policy={POLICY}")
                print(f"prompt_len={prompt_len}")
                print(f"trial={trial}")
                print(f"error={exc}")
                print("!" * 72)
                raise

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print(f"PASS          = {passed}")
    print(f"TOTAL         = {total}")
    print(f"FAIL          = {total - passed}")
    print(f"TEXT WARNINGS = {warning_count}")
    print("Cold determinism:")
    for prompt_len in PROMPT_LENGTHS:
        print(f"  {prompt_len:6d}: {determinism[prompt_len]}")
    print("=" * 72)

    if passed != total:
        raise SystemExit(1)

    print("[ALL PATH CHECKS PASS]")


if __name__ == "__main__":
    main()
