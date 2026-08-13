import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

BASE_URL = "http://127.0.0.1:30000"
MODEL_PATH = os.environ["MODEL_PATH"]
POLICY = os.getenv("POLICY", "always_restore")
CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))
PROMPT_LENGTHS = [int(x) for x in os.getenv("PROMPT_LENGTHS", "512,1024,2048,4096,8192,16384").split(",")]
TRIALS = int(os.getenv("TRIALS", "3"))
EVICTOR_LEN = int(os.getenv("EVICTOR_LEN", "30000"))
NUM_EVICTORS = int(os.getenv("NUM_EVICTORS", "24"))
RESULT_FILE = Path(os.getenv(
    "RESULT_FILE",
    f"/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/load_{POLICY}.jsonl",
))

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size
TEXT = (
    "Large language model inference uses hierarchical KV caching to reuse previous attention states. "
    "This request is used to evaluate concurrent host cache restoration and prefill recomputation. "
)

def target_ids(n, unique_id):
    ids = tokenizer.encode((TEXT + "\n") * 512, add_special_tokens=False)
    while len(ids) < n:
        ids += ids
    ids = ids[:n]
    ids[0] = 1000 + unique_id % min(20000, VOCAB_SIZE - 1001)
    return ids

def random_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, min(VOCAB_SIZE - 1, 50000)) for _ in range(n)]

def generate(ids, rid):
    payload = {
        "rid": rid, "input_ids": ids,
        "sampling_params": {"temperature": 0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 1},
        "stream": False,
    }
    begin = time.perf_counter()
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=300)
    latency = (time.perf_counter() - begin) * 1000
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list):
        body = body[0]
    return {"rid": rid, "latency_ms": latency, "cached_tokens": body.get("meta_info", {}).get("cached_tokens")}

def metric(name):
    text = requests.get(f"{BASE_URL}/metrics", timeout=10).text
    p = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    return sum(float(m.group(1)) for line in text.splitlines() if (m := p.match(line.strip())))

def flush():
    requests.post(f"{BASE_URL}/flush_cache", timeout=30).raise_for_status()
    time.sleep(0.5)

def run_case(n, trial):
    flush()
    targets = [target_ids(n, trial * 100 + i) for i in range(CONCURRENCY)]

    for i, ids in enumerate(targets):
        generate(ids, f"prep_c{CONCURRENCY}_len{n}_trial{trial}_{i}")

    time.sleep(1.0)
    generate(random_ids(128, 900000 + n + trial), f"probe_{n}_{trial}")

    evict_before = metric("sglang:evicted_tokens_total")
    for i in range(NUM_EVICTORS):
        generate(random_ids(EVICTOR_LEN, 3_000_000 + n * 100 + trial * 1000 + i), f"evict_{n}_{trial}_{i}")
    evicted_delta = metric("sglang:evicted_tokens_total") - evict_before
    if evicted_delta <= 0:
        raise RuntimeError("No GPU eviction observed.")

    load_before = metric("sglang:load_back_tokens_total")
    barrier = threading.Barrier(CONCURRENCY)

    def worker(i):
        rid = f"target_c{CONCURRENCY}_len{n}_trial{trial}_{i}"
        barrier.wait()
        return generate(targets[i], rid)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        rows = list(pool.map(worker, range(CONCURRENCY)))

    load_delta = metric("sglang:load_back_tokens_total") - load_before
    if POLICY == "always_restore" and load_delta <= 0:
        raise RuntimeError("Concurrent restore was not observed.")
    if POLICY == "always_recompute" and load_delta != 0:
        raise RuntimeError(f"Unexpected load-back under recompute: {load_delta}")

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        for row in rows:
            row.update({
                "policy": POLICY, "concurrency": CONCURRENCY, "prompt_len": n,
                "trial": trial, "global_load_back_delta": load_delta, "evicted_delta": evicted_delta,
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    med = sorted(x["latency_ms"] for x in rows)[len(rows) // 2]
    print(f"[PASS] policy={POLICY} c={CONCURRENCY} len={n} trial={trial} median={med:.3f} ms load={load_delta:.0f}")

def main():
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()
    for n in PROMPT_LENGTHS:
        for trial in range(TRIALS):
            run_case(n, trial)

if __name__ == "__main__":
    main()
