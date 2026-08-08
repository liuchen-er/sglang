import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
POLICIES = os.getenv("POLICIES", "always_restore,token_threshold,cost_model").split(",")

def load(policy):
    path = BASE / f"agent_{policy}.jsonl"
    rows, summaries = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["record_type"] == "request":
                rows.append(x)
            else:
                summaries.append(x)
    return rows, summaries

def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")

data = {}
for p in POLICIES:
    path = BASE / f"agent_{p}.jsonl"
    if path.exists():
        data[p] = load(p)

if "always_restore" not in data:
    raise RuntimeError("agent_always_restore.jsonl is required as baseline.")

print("=" * 128)
print("OVERALL AGENT L2-HIT RESULTS")
print("=" * 128)
print(
    f"{'Policy':>18} {'P50 TTFT':>10} {'P95 TTFT':>10} {'P99 TTFT':>10} "
    f"{'P95 TPOT':>10} {'P95 ITL':>10} {'P99 ITL':>10} "
    f"{'Out tok/s':>11} {'LoadBack/Req':>13} {'TTFT P95 Δ':>12}"
)

base_rows, _ = data["always_restore"]
base_p95 = pct([x["ttft_ms"] for x in base_rows], 95)

for policy, (rows, summaries) in data.items():
    ttft = [x["ttft_ms"] for x in rows]
    tpot = [x["tpot_ms"] for x in rows if x["output_tokens"] > 1]
    itl = [v for x in rows for v in x["itls_ms"]]
    throughput = np.median([x["output_throughput"] for x in summaries])
    load_per_req = sum(x["load_back_delta"] for x in summaries) / max(sum(x["requests"] for x in summaries), 1)
    improve = (base_p95 - pct(ttft, 95)) / base_p95 * 100

    print(
        f"{policy:>18} {pct(ttft,50):10.3f} {pct(ttft,95):10.3f} {pct(ttft,99):10.3f} "
        f"{pct(tpot,95):10.3f} {pct(itl,95):10.3f} {pct(itl,99):10.3f} "
        f"{throughput:11.1f} {load_per_req:13.1f} {improve:11.2f}%"
    )

print("\n" + "=" * 104)
print("TTFT BY REUSABLE PREFIX LENGTH")
print("=" * 104)
print(f"{'Prefix':>8}", end="")
for p in data:
    print(f" {p + ' P50':>18} {p + ' P95':>18}", end="")
print()

prefixes = sorted(set(x["prefix_len"] for rows, _ in data.values() for x in rows))
for n in prefixes:
    print(f"{n:8d}", end="")
    for p, (rows, _) in data.items():
        vals = [x["ttft_ms"] for x in rows if x["prefix_len"] == n]
        print(f" {pct(vals,50):18.3f} {pct(vals,95):18.3f}", end="")
    print()
