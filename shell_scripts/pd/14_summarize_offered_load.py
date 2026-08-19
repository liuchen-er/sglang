#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================
EXP_ROOT = Path("/root/autodl-tmp/sglang_pd_exp")

RAW_ROOT = EXP_ROOT / "results" / "raw" / "offered_load"
SUMMARY_ROOT = EXP_ROOT / "results" / "summary" / "offered_load"

DIRS = {
    "colocated": RAW_ROOT / "colocated",
    "pd": RAW_ROOT / "pd",
}

SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def percentile(values, p):
    if not values:
        return math.nan

    return float(
        np.percentile(
            np.asarray(values, dtype=float),
            p,
        )
    )


def mean(values):
    if not values:
        return math.nan

    return float(
        np.mean(
            np.asarray(values, dtype=float)
        )
    )


def load_jsonl(path):
    records = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# Load one architecture
#
# Match key:
#   input_len
#   output_len
#   request_rate
#   max_concurrency
#
# num_prompts is intentionally NOT part of the key.
# Therefore n200 vs n300 can still be compared.
# ============================================================
def load_mode(mode):
    input_dir = DIRS[mode]

    if not input_dir.exists():
        raise RuntimeError(
            f"Directory does not exist: {input_dir}"
        )

    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        raise RuntimeError(
            f"No JSONL files found: {input_dir}"
        )

    groups = defaultdict(list)

    for path in files:
        for record in load_jsonl(path):

            input_len = int(record["random_input_len"])
            output_len = int(record["random_output_len"])
            request_rate = float(record["request_rate"])
            max_concurrency = int(record["max_concurrency"])

            key = (
                input_len,
                output_len,
                request_rate,
                max_concurrency,
            )

            # TTFT: seconds -> ms
            ttfts_ms = [
                float(x) * 1000.0
                for x in record.get("ttfts", [])
                if x is not None
            ]

            # ITL: List[List[seconds]] -> flat List[ms]
            itls_ms = [
                float(x) * 1000.0
                for req_itls in record.get("itls", [])
                for x in req_itls
                if x is not None
            ]

            groups[key].append(
                {
                    "file": path.name,

                    "completed":
                        int(record["completed"]),

                    "duration_s":
                        float(record["duration"]),

                    "request_throughput":
                        float(record["request_throughput"]),

                    "output_throughput":
                        float(record["output_throughput"]),

                    "concurrency":
                        float(record["concurrency"]),

                    "p99_tpot_ms":
                        float(record["p99_tpot_ms"]),

                    "ttfts_ms":
                        ttfts_ms,

                    "itls_ms":
                        itls_ms,
                }
            )

    return groups


# ============================================================
# Aggregate repeats of the same workload
# ============================================================
def aggregate(group):
    all_ttfts = [
        x
        for run in group
        for x in run["ttfts_ms"]
    ]

    all_itls = [
        x
        for run in group
        for x in run["itls_ms"]
    ]

    return {
        "runs": len(group),

        "requests":
            sum(run["completed"] for run in group),

        "request_throughput":
            mean([
                run["request_throughput"]
                for run in group
            ]),

        "output_throughput":
            mean([
                run["output_throughput"]
                for run in group
            ]),

        "concurrency":
            mean([
                run["concurrency"]
                for run in group
            ]),

        # Request-level raw samples pooled across repeats
        "p95_ttft_ms":
            percentile(all_ttfts, 95),

        "p99_ttft_ms":
            percentile(all_ttfts, 99),

        # Current JSONL does not contain per-request TPOT.
        # Therefore use mean of benchmark-reported per-run P99 TPOT.
        "p99_tpot_ms":
            mean([
                run["p99_tpot_ms"]
                for run in group
            ]),

        # Token-level raw samples pooled across repeats
        "p95_itl_ms":
            percentile(all_itls, 95),

        "p99_itl_ms":
            percentile(all_itls, 99),

        "p999_itl_ms":
            percentile(all_itls, 99.9),

        "max_itl_ms":
            max(all_itls)
            if all_itls
            else math.nan,

        "itl_samples":
            len(all_itls),
    }


# ============================================================
# Load and match
# ============================================================
colocated_groups = load_mode("colocated")
pd_groups = load_mode("pd")

matched_keys = sorted(
    set(colocated_groups.keys())
    & set(pd_groups.keys())
)

only_colocated = sorted(
    set(colocated_groups.keys())
    - set(pd_groups.keys())
)

only_pd = sorted(
    set(pd_groups.keys())
    - set(colocated_groups.keys())
)

if not matched_keys:
    raise RuntimeError(
        "No matching Colocated / PD workloads found."
    )


rows = []


# ============================================================
# Build comparison rows
# ============================================================
for key in matched_keys:
    input_len, output_len, rate, max_concurrency = key

    c = aggregate(colocated_groups[key])
    p = aggregate(pd_groups[key])

    ttft_change_pct = (
        (p["p99_ttft_ms"] / c["p99_ttft_ms"] - 1.0)
        * 100.0
    )

    tpot_reduction_pct = (
        (1.0 - p["p99_tpot_ms"] / c["p99_tpot_ms"])
        * 100.0
    )

    itl_reduction_pct = (
        (1.0 - p["p99_itl_ms"] / c["p99_itl_ms"])
        * 100.0
    )

    itl999_reduction_pct = (
        (1.0 - p["p999_itl_ms"] / c["p999_itl_ms"])
        * 100.0
    )

    rows.append(
        {
            "input_len": input_len,
            "output_len": output_len,
            "request_rate": rate,
            "max_concurrency": max_concurrency,

            "colocated_runs": c["runs"],
            "pd_runs": p["runs"],

            "colocated_requests": c["requests"],
            "pd_requests": p["requests"],

            "colocated_req_s":
                c["request_throughput"],

            "pd_req_s":
                p["request_throughput"],

            "colocated_output_tok_s":
                c["output_throughput"],

            "pd_output_tok_s":
                p["output_throughput"],

            "colocated_concurrency":
                c["concurrency"],

            "pd_concurrency":
                p["concurrency"],

            "colocated_p99_ttft_ms":
                c["p99_ttft_ms"],

            "pd_p99_ttft_ms":
                p["p99_ttft_ms"],

            "ttft_change_pct":
                ttft_change_pct,

            "colocated_p99_tpot_ms":
                c["p99_tpot_ms"],

            "pd_p99_tpot_ms":
                p["p99_tpot_ms"],

            "tpot_reduction_pct":
                tpot_reduction_pct,

            "colocated_p99_itl_ms":
                c["p99_itl_ms"],

            "pd_p99_itl_ms":
                p["p99_itl_ms"],

            "itl_reduction_pct":
                itl_reduction_pct,

            "colocated_p999_itl_ms":
                c["p999_itl_ms"],

            "pd_p999_itl_ms":
                p["p999_itl_ms"],

            "itl999_reduction_pct":
                itl999_reduction_pct,

            "colocated_max_itl_ms":
                c["max_itl_ms"],

            "pd_max_itl_ms":
                p["max_itl_ms"],

            "colocated_itl_samples":
                c["itl_samples"],

            "pd_itl_samples":
                p["itl_samples"],
        }
    )


# ============================================================
# Save CSV
# ============================================================
output_file = (
    SUMMARY_ROOT
    / "offered_load_compare.csv"
)

with output_file.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Console table
# ============================================================
print()
print("=" * 164)
print("EQUAL OFFERED LOAD: COLOCATED vs PD")
print("=" * 164)

header = (
    f"{'Input':>6} "
    f"{'Rate':>6} "
    f"{'N Col':>6} "
    f"{'N PD':>6} "
    f"{'Req Col':>8} "
    f"{'Req PD':>8} "
    f"{'C Col':>7} "
    f"{'C PD':>7} "
    f"{'TTFT Col':>10} "
    f"{'TTFT PD':>10} "
    f"{'TPOT Col':>9} "
    f"{'TPOT PD':>9} "
    f"{'ITL Col':>9} "
    f"{'ITL PD':>9} "
    f"{'P99.9 Col':>10} "
    f"{'P99.9 PD':>10}"
)

print(header)
print("-" * len(header))

for r in rows:
    print(
        f"{r['input_len']:>6} "
        f"{r['request_rate']:>6.2f} "

        f"{r['colocated_requests']:>6} "
        f"{r['pd_requests']:>6} "

        f"{r['colocated_req_s']:>8.2f} "
        f"{r['pd_req_s']:>8.2f} "

        f"{r['colocated_concurrency']:>7.2f} "
        f"{r['pd_concurrency']:>7.2f} "

        f"{r['colocated_p99_ttft_ms']:>10.1f} "
        f"{r['pd_p99_ttft_ms']:>10.1f} "

        f"{r['colocated_p99_tpot_ms']:>9.2f} "
        f"{r['pd_p99_tpot_ms']:>9.2f} "

        f"{r['colocated_p99_itl_ms']:>9.2f} "
        f"{r['pd_p99_itl_ms']:>9.2f} "

        f"{r['colocated_p999_itl_ms']:>10.2f} "
        f"{r['pd_p999_itl_ms']:>10.2f}"
    )


# ============================================================
# Improvement table
# ============================================================
print()
print("=" * 92)
print("PD CHANGE RELATIVE TO COLOCATED")
print("=" * 92)

header2 = (
    f"{'Input':>6} "
    f"{'Rate':>6} "
    f"{'TTFT change':>14} "
    f"{'TPOT reduction':>16} "
    f"{'P99 ITL reduction':>19} "
    f"{'P99.9 reduction':>18}"
)

print(header2)
print("-" * len(header2))

for r in rows:
    print(
        f"{r['input_len']:>6} "
        f"{r['request_rate']:>6.2f} "

        f"{r['ttft_change_pct']:>13.1f}% "
        f"{r['tpot_reduction_pct']:>15.1f}% "
        f"{r['itl_reduction_pct']:>18.1f}% "
        f"{r['itl999_reduction_pct']:>17.1f}%"
    )


# ============================================================
# Diagnostics
# ============================================================
print()
print("=" * 92)
print("SUMMARY")
print("=" * 92)

print(f"Matched workloads : {len(matched_keys)}")
print(f"Only Colocated    : {len(only_colocated)}")
print(f"Only PD           : {len(only_pd)}")

if only_colocated:
    print()
    print("Unmatched Colocated workloads:")
    for x in only_colocated:
        print(" ", x)

if only_pd:
    print()
    print("Unmatched PD workloads:")
    for x in only_pd:
        print(" ", x)

print()
print("Metric definitions:")
print("  TTFT   : pooled request-level P99")
print("  TPOT   : mean of benchmark per-run P99 TPOT")
print("  ITL    : pooled token-level P99")
print("  P99.9  : pooled token-level P99.9")
print()
print("Saved:")
print(output_file)
