#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/root/autodl-tmp/sglang_pd_exp/results/raw/manual")

DIRS = {
    "colocated": ROOT / "colocated",
    "pd": ROOT / "pd",
}


def load_mode(mode):
    groups = defaultdict(list)

    for path in sorted(DIRS[mode].glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)

                # 只按 workload 匹配，不要求两边 num_prompts 相同
                key = (
                    int(r["random_input_len"]),
                    int(r["random_output_len"]),
                    float(r["request_rate"]),
                    int(r["max_concurrency"]),
                )

                ttfts = [
                    x * 1000.0
                    for x in r.get("ttfts", [])
                    if x is not None
                ]

                itls = [
                    x * 1000.0
                    for req in r.get("itls", [])
                    for x in req
                    if x is not None
                ]

                groups[key].append({
                    "completed": int(r["completed"]),
                    "req_s": float(r["request_throughput"]),
                    "out_tok_s": float(r["output_throughput"]),
                    "concurrency": float(r["concurrency"]),
                    "p99_tpot": float(r["p99_tpot_ms"]),
                    "ttfts": ttfts,
                    "itls": itls,
                    "file": path.name,
                })

    return groups


def aggregate(group):
    ttfts = [x for r in group for x in r["ttfts"]]
    itls = [x for r in group for x in r["itls"]]

    return {
        "requests": sum(r["completed"] for r in group),
        "req_s": np.mean([r["req_s"] for r in group]),
        "out_tok_s": np.mean([r["out_tok_s"] for r in group]),
        "concurrency": np.mean([r["concurrency"] for r in group]),

        "p99_ttft": np.percentile(ttfts, 99),
        "p99_tpot": np.mean([r["p99_tpot"] for r in group]),

        "p99_itl": np.percentile(itls, 99),
        "p999_itl": np.percentile(itls, 99.9),
        "max_itl": np.max(itls),
    }


col = load_mode("colocated")
pd = load_mode("pd")

keys = sorted(set(col) & set(pd))

print()
print("=" * 154)
print("EQUAL OFFERED LOAD: COLOCATED vs PD")
print("=" * 154)

header = (
    f"{'Input':>6} {'Rate':>6} "
    f"{'N Col':>6} {'N PD':>6} "
    f"{'Req Col':>8} {'Req PD':>8} "
    f"{'C Col':>7} {'C PD':>7} "
    f"{'TTFT Col':>10} {'TTFT PD':>10} "
    f"{'TPOT Col':>9} {'TPOT PD':>9} "
    f"{'ITL Col':>9} {'ITL PD':>9} "
    f"{'P99.9 Col':>10} {'P99.9 PD':>10}"
)

print(header)
print("-" * len(header))

for key in keys:
    c = aggregate(col[key])
    p = aggregate(pd[key])

    input_len, output_len, rate, max_conc = key

    print(
        f"{input_len:>6} {rate:>6.2f} "
        f"{c['requests']:>6} {p['requests']:>6} "
        f"{c['req_s']:>8.2f} {p['req_s']:>8.2f} "
        f"{c['concurrency']:>7.2f} {p['concurrency']:>7.2f} "
        f"{c['p99_ttft']:>10.1f} {p['p99_ttft']:>10.1f} "
        f"{c['p99_tpot']:>9.2f} {p['p99_tpot']:>9.2f} "
        f"{c['p99_itl']:>9.2f} {p['p99_itl']:>9.2f} "
        f"{c['p999_itl']:>10.2f} {p['p999_itl']:>10.2f}"
    )

print()
print("Matched cases:", len(keys))
print()
print("TTFT : pooled P99")
print("TPOT : benchmark P99")
print("ITL  : pooled P99")
print("P99.9: pooled ITL P99.9")
