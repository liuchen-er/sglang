import json
import os
from pathlib import Path

import numpy as np

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
POLICIES = os.getenv("POLICIES", "always_restore,token_threshold,cost_model").split(",")

def load(path):
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

control_rows, control_summaries = load(BASE / "mixed_control.jsonl")
control_decode = [x for x in control_rows if x["group"] == "decode"]
control_tpot = [x["tpot_ms"] for x in control_decode if x["output_tokens"] > 1]
control_itl = [v for x in control_decode for v in x["itls_ms"]]

control_p95_tpot = pct(control_tpot, 95)
control_p95_itl = pct(control_itl, 95)
control_p99_itl = pct(control_itl, 99)

data = {}
for policy in POLICIES:
    path = BASE / f"mixed_{policy}.jsonl"
    if path.exists():
        data[policy] = load(path)

if "always_restore" not in data:
    raise RuntimeError("mixed_always_restore.jsonl is required.")

base_target = [x for x in data["always_restore"][0] if x["group"] == "target"]
base_target_p95 = pct([x["ttft_ms"] for x in base_target], 95)

print("=" * 156)
print("MIXED PREFILL + DECODE RESULTS")
print("=" * 156)
print(
    f"{'Policy':>18} {'Target P50':>11} {'Target P95':>11} {'Target P99':>11} "
    f"{'Decode P95 TPOT':>15} {'P95 ITL':>10} {'P99 ITL':>10} "
    f"{'P99 ITL Δ':>11} {'Out tok/s':>11} {'LoadBack':>11} "
    f"{'Evict':>10} {'TTFT P95 Δ':>12}"
)

for policy, (rows, summaries) in data.items():
    target = [x for x in rows if x["group"] == "target"]
    decode = [x for x in rows if x["group"] == "decode"]

    target_ttft = [x["ttft_ms"] for x in target]
    decode_tpot = [x["tpot_ms"] for x in decode if x["output_tokens"] > 1]
    decode_itl = [v for x in decode for v in x["itls_ms"]]

    p99_itl = pct(decode_itl, 99)
    itl_overhead = (p99_itl - control_p99_itl) / control_p99_itl * 100
    ttft_improve = (base_target_p95 - pct(target_ttft,95)) / base_target_p95 * 100

    throughput = np.median([x["output_throughput"] for x in summaries])
    loadback = np.median([x["load_back_delta"] for x in summaries])
    evicted = np.median([x["measurement_evicted_delta"] for x in summaries])

    print(
        f"{policy:>18} {pct(target_ttft,50):11.3f} {pct(target_ttft,95):11.3f} "
        f"{pct(target_ttft,99):11.3f} {pct(decode_tpot,95):15.3f} "
        f"{pct(decode_itl,95):10.3f} {p99_itl:10.3f} {itl_overhead:10.2f}% "
        f"{throughput:11.1f} {loadback:11.0f} {evicted:10.0f} {ttft_improve:11.2f}%"
    )

print("\nCONTROL DECODE-ONLY")
print(f"P95 TPOT = {control_p95_tpot:.3f} ms")
print(f"P95 ITL  = {control_p95_itl:.3f} ms")
print(f"P99 ITL  = {control_p99_itl:.3f} ms")

print("\nTARGET TTFT BY PREFIX LENGTH")
prefixes = sorted(set(
    x["prefix_len"]
    for rows, _ in data.values()
    for x in rows if x["group"] == "target"
))

print(f"{'Prefix':>8}", end="")
for p in data:
    print(f" {p + ' P50':>18} {p + ' P95':>18}", end="")
print()

for n in prefixes:
    print(f"{n:8d}", end="")
    for p, (rows, _) in data.items():
        vals = [
            x["ttft_ms"]
            for x in rows
            if x["group"] == "target" and x["prefix_len"] == n
        ]
        print(f" {pct(vals,50):18.3f} {pct(vals,95):18.3f}", end="")
    print()
