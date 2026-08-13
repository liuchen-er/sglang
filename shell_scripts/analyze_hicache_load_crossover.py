import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")

def load(name):
    d = defaultdict(list)
    with (BASE / name).open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            d[(x["concurrency"], x["prompt_len"])].append(x["latency_ms"])
    return d

r = load("load_always_restore.jsonl")
c = load("load_always_recompute.jsonl")
concurrencies = sorted(set(k[0] for k in r) & set(k[0] for k in c))

for cc in concurrencies:
    print(f"\nConcurrency={cc}")
    print(f"{'Len':>7} {'Restore':>11} {'Recompute':>11} {'R-C':>10} {'Winner':>10}")
    cross = None
    lens = sorted(k[1] for k in r if k[0] == cc and k in c)
    for n in lens:
        rm = statistics.median(r[(cc, n)])
        cm = statistics.median(c[(cc, n)])
        winner = "restore" if rm < cm else "recompute"
        if cross is None and winner == "restore":
            cross = n
        print(f"{n:7d} {rm:11.3f} {cm:11.3f} {rm-cm:10.3f} {winner:>10}")
    print(f"Approx crossover prompt length: {cross}")
