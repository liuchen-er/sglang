import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
PAGE_SIZE = 64
MARGIN_MS = 1.0

def load(path):
    out = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out[x["prompt_len"]].append(x)
    return out

r = load(BASE / "policy_always_restore.jsonl")
c = load(BASE / "policy_always_recompute.jsonl")
lengths = sorted(set(r) & set(c))
rows = []

print(f"{'Prompt':>8} {'HostHit':>8} {'Restore':>11} {'Recompute':>11} {'R-C':>9} {'Winner':>10}")
for n in lengths:
    rm = statistics.median(x["revisit_latency_ms"] for x in r[n])
    cm = statistics.median(x["revisit_latency_ms"] for x in c[n])
    hits = [x["revisit_cached_tokens"] for x in r[n] if x["revisit_cached_tokens"]]
    host_hit = int(statistics.median(hits)) if hits else max(0, n - PAGE_SIZE)
    winner = "restore" if rm + MARGIN_MS < cm else ("recompute" if cm + MARGIN_MS < rm else "tie")
    rows.append((n, host_hit, rm, cm, winner))
    print(f"{n:8d} {host_hit:8d} {rm:11.3f} {cm:11.3f} {rm-cm:9.3f} {winner:>10}")

candidate = None
for i, row in enumerate(rows):
    if all(x[4] == "restore" for x in rows[i:]):
        candidate = math.ceil(row[1] / PAGE_SIZE) * PAGE_SIZE
        break

print()
if candidate is None:
    print("RECOMMENDED_THRESHOLD=NONE")
    print("No stable restore-winning region was found.")
else:
    print(f"RECOMMENDED_THRESHOLD={candidate}")
    print("Verify this value against host_hit_length in server_always_restore.log.")
