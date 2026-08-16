#!/usr/bin/env python3

import csv
from pathlib import Path

ROOT = Path("/root/autodl-tmp/sglang_pd_exp/results/summary/capacity")

def load(path):
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (
                int(r["input_len"]),
                int(r["output_len"]),
                int(r["concurrency"]),
            )
            out[key] = r
    return out

col = load(ROOT / "colocated_capacity_summary.csv")
pd = load(ROOT / "pd_capacity_summary.csv")

def f(row, key):
    return float(row[key])

rows = []

for key in sorted(set(col) & set(pd)):
    c = col[key]
    p = pd[key]

    c_req = f(c, "request_throughput_mean")
    p_req = f(p, "request_throughput_mean")

    c_ttft = f(c, "pooled_p99_ttft_ms")
    p_ttft = f(p, "pooled_p99_ttft_ms")

    c_tpot = f(c, "p99_tpot_ms_mean")
    p_tpot = f(p, "p99_tpot_ms_mean")

    c_itl = f(c, "pooled_p99_itl_ms")
    p_itl = f(p, "pooled_p99_itl_ms")

    c_itl999 = f(c, "pooled_p999_itl_ms")
    p_itl999 = f(p, "pooled_p999_itl_ms")

    rows.append({
        "input_len": key[0],
        "output_len": key[1],
        "concurrency": key[2],

        "colocated_req_s": c_req,
        "pd_req_s": p_req,
        "req_s_change_pct": (p_req / c_req - 1) * 100,

        "colocated_p99_ttft_ms": c_ttft,
        "pd_p99_ttft_ms": p_ttft,
        "ttft_change_pct": (p_ttft / c_ttft - 1) * 100,

        "colocated_p99_tpot_ms": c_tpot,
        "pd_p99_tpot_ms": p_tpot,
        "tpot_reduction_pct": (1 - p_tpot / c_tpot) * 100,

        "colocated_p99_itl_ms": c_itl,
        "pd_p99_itl_ms": p_itl,
        "itl_reduction_pct": (1 - p_itl / c_itl) * 100,

        "colocated_p999_itl_ms": c_itl999,
        "pd_p999_itl_ms": p_itl999,
        "itl999_reduction_pct": (1 - p_itl999 / c_itl999) * 100,
    })

out = ROOT / "capacity_compare.csv"

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print()
print("=" * 112)
print("CAPACITY: COLOCATED vs PD")
print("=" * 112)
print(
    f"{'Input':>6} {'C':>3} "
    f"{'Req Col':>9} {'Req PD':>9} "
    f"{'P99 TTFT Col':>13} {'P99 TTFT PD':>12} "
    f"{'P99 ITL Col':>11} {'P99 ITL PD':>10} "
    f"{'P99.9↓':>9}"
)
print("-" * 112)

for r in rows:
    print(
        f"{r['input_len']:>6} "
        f"{r['concurrency']:>3} "
        f"{r['colocated_req_s']:>9.2f} "
        f"{r['pd_req_s']:>9.2f} "
        f"{r['colocated_p99_ttft_ms']:>13.2f} "
        f"{r['pd_p99_ttft_ms']:>12.2f} "
        f"{r['colocated_p99_itl_ms']:>11.2f} "
        f"{r['pd_p99_itl_ms']:>10.2f} "
        f"{r['itl999_reduction_pct']:>8.1f}%"
    )

print()
print("Saved:", out)
