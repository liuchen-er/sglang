import json
import os
import random
import re
import time
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:30000")
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
PROMPT_LENGTHS = [int(x) for x in os.getenv("PROMPT_LENGTHS", "512,1024,2048,4096,8192,16384").split(",")]
TRIALS = int(os.getenv("TRIALS", "5"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))
TIMEOUT = 300
DATA = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
RESULT_FILE = Path(os.getenv("RESULT_FILE", str(DATA / f"policy_{POLICY}.jsonl")))

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size
TEXT = (
    "Large language model inference systems use KV cache to avoid recomputing previous attention states. "
    "Hierarchical KV cache stores colder states in CPU memory and restores them to GPU when reused. "
    "This benchmark measures hierarchical cache restore and recomputation latency under controlled conditions. "
)

def target_ids(n):
    ids = tokenizer.encode((TEXT + "\n") * 256, add_special_tokens=False)
    while len(ids) < n:
        ids += ids
    return ids[:n]

def random_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, min(VOCAB_SIZE - 1, 50000)) for _ in range(n)]

def generate(ids, rid):
    payload = {
        "rid": rid,
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0, "top_k": 1, "top_p": 1.0,
            "repetition_penalty": 1.0, "max_new_tokens": 1,
        },
        "stream": False,
    }
    begin = time.perf_counter()
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=TIMEOUT)
    latency_ms = (time.perf_counter() - begin) * 1000
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list):
        body = body[0]
    return latency_ms, body.get("meta_info", {}).get("cached_tokens")

def metric(name):
    text = requests.get(f"{BASE_URL}/metrics", timeout=10).text
    p = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    return sum(float(m.group(1)) for line in text.splitlines() if (m := p.match(line.strip())))

def metrics():
    return {
        "host_used": metric("sglang:hicache_host_used_tokens"),
        "evicted": metric("sglang:evicted_tokens_total"),
        "load_back": metric("sglang:load_back_tokens_total"),
    }

def flush():
    requests.post(f"{BASE_URL}/flush_cache", timeout=30).raise_for_status()
    time.sleep(0.5)

def refresh(seed):
    generate(random_ids(128, seed), f"metric_probe_{seed}")
    time.sleep(0.2)

def save(row):
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def run_case(n, trial):
    flush()
    ids = target_ids(n)
    cold_ms, _ = generate(ids, f"prep_{POLICY}_len{n}_trial{trial}")
    time.sleep(1.0)
    refresh(800000 + n + trial)

    before = metrics()
    for i in range(NUM_EVICTORS):
        generate(random_ids(EVICTOR_LEN, 2_000_000 + n * 100 + trial * 1000 + i), f"evict_{n}_{trial}_{i}")
    after = metrics()
    evicted_delta = after["evicted"] - before["evicted"]
    if evicted_delta <= 0:
        raise RuntimeError(f"No eviction: len={n}, trial={trial}")

    load_before = metric("sglang:load_back_tokens_total")
    rid = f"target_{POLICY}_c1_len{n}_trial{trial}"
    revisit_ms, cached = generate(ids, rid)
    load_delta = metric("sglang:load_back_tokens_total") - load_before

    if POLICY == "always_restore" and load_delta <= 0:
        raise RuntimeError(f"Restore not observed: len={n}, trial={trial}")
    if POLICY == "always_recompute" and load_delta != 0:
        raise RuntimeError(f"Unexpected restore: len={n}, trial={trial}")

    action = "restore" if load_delta > 0 else "recompute"
    row = {
        "rid": rid, "policy": POLICY, "prompt_len": n, "trial": trial,
        "cold_latency_ms": cold_ms, "revisit_latency_ms": revisit_ms,
        "revisit_cached_tokens": cached, "load_back_delta": load_delta,
        "evicted_delta": evicted_delta, "action": action,
    }
    save(row)
    print(f"[PASS] policy={POLICY} len={n} trial={trial} cached={cached} load={load_delta:.0f} latency={revisit_ms:.3f} ms")

def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    print(f"POLICY={POLICY}, lengths={PROMPT_LENGTHS}, trials={TRIALS}, result={RESULT_FILE}")
    for n in PROMPT_LENGTHS:
        for trial in range(TRIALS):
            run_case(n, trial)

if __name__ == "__main__":
    main()
