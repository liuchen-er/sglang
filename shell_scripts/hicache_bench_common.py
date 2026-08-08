import asyncio
import json
import math
import random
import re
import time

import aiohttp
import requests

TIMEOUT = 300

def metric(base_url, name):
    text = requests.get(f"{base_url}/metrics", timeout=10).text
    p = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    return sum(float(m.group(1)) for line in text.splitlines() if (m := p.match(line.strip())))

def metrics(base_url):
    return {
        "host_used": metric(base_url, "sglang:hicache_host_used_tokens"),
        "evicted": metric(base_url, "sglang:evicted_tokens_total"),
        "load_back": metric(base_url, "sglang:load_back_tokens_total"),
    }

def flush(base_url):
    r = requests.post(f"{base_url}/flush_cache", timeout=30)
    r.raise_for_status()
    time.sleep(0.5)

def random_ids(n, seed, vocab_size):
    rng = random.Random(seed)
    upper = min(vocab_size - 1, 50000)
    return [rng.randrange(1000, upper) for _ in range(n)]

def fit_text_ids(tokenizer, header, body, n):
    h = tokenizer.encode(header, add_special_tokens=False)
    b = tokenizer.encode(body, add_special_tokens=False)
    if not b:
        raise RuntimeError("Body tokenization returned empty ids.")
    ids = list(h)
    while len(ids) < n:
        ids.extend(b)
    return ids[:n]

def sync_generate(base_url, ids, rid, max_new_tokens=1):
    payload = {
        "rid": rid,
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0, "top_k": 1, "top_p": 1.0,
            "ignore_eos": True, "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }
    r = requests.post(f"{base_url}/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list):
        body = body[0]
    return body

def runtime_warmup(base_url, vocab_size):
    sync_generate(base_url, random_ids(128, 991337, vocab_size), "runtime_warmup", 16)
    flush(base_url)

def refresh_metrics(base_url, vocab_size, seed):
    sync_generate(base_url, random_ids(128, seed, vocab_size), f"metric_probe_{seed}", 1)
    time.sleep(0.3)
    return metrics(base_url)

def force_eviction(base_url, vocab_size, evictor_len, num_evictors, seed):
    before = metrics(base_url)
    for i in range(num_evictors):
        sync_generate(base_url, random_ids(evictor_len, seed + i, vocab_size), f"evict_{seed}_{i}", 1)
    after = metrics(base_url)
    delta = after["evicted"] - before["evicted"]
    if delta <= 0:
        raise RuntimeError("No GPU KV eviction observed.")
    return delta

async def stream_generate(session, base_url, ids, rid, max_new_tokens, extra=None):
    payload = {
        "rid": rid,
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0, "top_k": 1, "top_p": 1.0,
            "ignore_eos": True, "max_new_tokens": max_new_tokens,
        },
        "stream": True,
    }
    st = time.perf_counter()
    ttft = None
    most_recent = st
    last_output_len = 0
    output_len = 0
    cached_tokens = 0
    itls = []
    generated_text = ""

    async with session.post(f"{base_url}/generate", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"{rid}: HTTP {response.status}: {await response.text()}")

        async for raw in response.content:
            raw = raw.strip()
            if not raw:
                continue

            chunk = raw.decode("utf-8")
            if chunk.startswith("data: "):
                chunk = chunk[6:]
            elif chunk.startswith("data:"):
                chunk = chunk[5:]

            if chunk == "[DONE]":
                continue

            data = json.loads(chunk)
            meta = data.get("meta_info") or {}
            cached_tokens = meta.get("cached_tokens", cached_tokens)
            text = data.get("text", "")
            current_output_len = int(meta.get("completion_tokens", output_len))

            if text:
                ts = time.perf_counter()
                generated_text = text

                if ttft is None:
                    ttft = ts - st
                else:
                    new_tokens = current_output_len - last_output_len
                    if new_tokens > 0:
                        gap = ts - most_recent
                        itls.extend([gap / new_tokens] * new_tokens)

                most_recent = ts
                last_output_len = current_output_len
                output_len = current_output_len

    end = time.perf_counter()
    latency = end - st
    ttft = latency if ttft is None else ttft
    tpot = (latency - ttft) / (output_len - 1) if output_len > 1 else 0.0

    row = {
        "rid": rid,
        "input_tokens": len(ids),
        "output_tokens": output_len,
        "cached_tokens": cached_tokens,
        "ttft_ms": ttft * 1000,
        "e2e_ms": latency * 1000,
        "tpot_ms": tpot * 1000,
        "itls_ms": [x * 1000 for x in itls],
    }
    if extra:
        row.update(extra)
    return row

async def run_requests(specs, base_url, max_concurrency, request_rate, seed):
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0)
    sem = asyncio.Semaphore(max_concurrency)
    rng = random.Random(seed)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def one(spec):
            async with sem:
                return await stream_generate(
                    session, base_url, spec["ids"], spec["rid"],
                    spec["max_new_tokens"], spec.get("extra"),
                )

        tasks = []
        for spec in specs:
            tasks.append(asyncio.create_task(one(spec)))
            if not math.isinf(request_rate):
                await asyncio.sleep(rng.expovariate(request_rate))

        return await asyncio.gather(*tasks)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
