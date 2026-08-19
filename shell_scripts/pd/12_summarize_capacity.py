#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TAG_RE = re.compile(
    r"^(?P<mode>colocated|pd)"
    r"_i(?P<input>\d+)"
    r"_o(?P<output>\d+)"
    r"_c(?P<conc>\d+)"
    r"_r(?P<rate>[^_]+)"
    r"_n(?P<num>\d+)"
)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def pct(values, p):
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=float), p))


def load_jsonl(path):
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_tag(record, path):
    tag = record.get("tag") or path.stem
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Cannot parse tag: {tag}")

    d = m.groupdict()
    return {
        "mode": d["mode"],
        "input_len": int(d["input"]),
        "output_len": int(d["output"]),
        "concurrency": int(d["conc"]),
        "request_rate": d["rate"],
        "num_prompts": int(d["num"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["colocated", "pd"])
    parser.add_argument(
        "--exp-root",
        default="/root/autodl-tmp/sglang_pd_exp",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    input_dir = exp_root / "results" / "raw" / "capacity" / args.mode
    output_dir = exp_root / "results" / "summary" / "capacity"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    runs = []

    for path in files:
        records = load_jsonl(path)

        if len(records) != 1:
            print(
                f"WARNING: {path.name} contains {len(records)} JSON records; "
                f"using each record as one run."
            )

        for rec in records:
            meta = parse_tag(rec, path)

            if meta["mode"] != args.mode:
                continue

            completed = int(rec["completed"])
            expected = meta["num_prompts"]

            expected_input = expected * meta["input_len"]
            expected_output = expected * meta["output_len"]

            valid = (
                completed == expected
                and int(rec["total_input_tokens"]) == expected_input
                and int(rec["total_output_tokens"]) == expected_output
            )

            ttfts_ms = [
                float(x) * 1000.0
                for x in rec.get("ttfts", [])
                if x is not None
            ]

            raw_itls = rec.get("itls", [])
            itls_ms = [
                float(x) * 1000.0
                for request_itls in raw_itls
                for x in request_itls
                if x is not None
            ]

            runs.append(
                {
                    **meta,
                    "file": path.name,
                    "valid": valid,
                    "completed": completed,
                    "duration_s": float(rec["duration"]),
                    "request_throughput": float(rec["request_throughput"]),
                    "input_throughput": float(rec["input_throughput"]),
                    "output_throughput": float(rec["output_throughput"]),
                    "total_throughput": float(rec["total_throughput"]),
                    "mean_ttft_ms": float(rec["mean_ttft_ms"]),
                    "p95_ttft_ms": float(rec["p95_ttft_ms"]),
                    "p99_ttft_ms": float(rec["p99_ttft_ms"]),
                    "mean_tpot_ms": float(rec["mean_tpot_ms"]),
                    "p95_tpot_ms": float(rec["p95_tpot_ms"]),
                    "p99_tpot_ms": float(rec["p99_tpot_ms"]),
                    "mean_itl_ms": float(rec["mean_itl_ms"]),
                    "p95_itl_ms": float(rec["p95_itl_ms"]),
                    "p99_itl_ms": float(rec["p99_itl_ms"]),
                    "max_itl_ms": max(itls_ms) if itls_ms else math.nan,
                    "_ttfts_ms": ttfts_ms,
                    "_itls_ms": itls_ms,
                }
            )

    # --------------------------------------------------------
    # Per-run CSV
    # --------------------------------------------------------
    per_run_path = output_dir / f"{args.mode}_per_run.csv"

    public_fields = [
        k for k in runs[0].keys()
        if not k.startswith("_")
    ]

    with per_run_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: r[k] for k in public_fields})

    # --------------------------------------------------------
    # Group by workload
    # --------------------------------------------------------
    groups = defaultdict(list)

    for r in runs:
        key = (
            r["input_len"],
            r["output_len"],
            r["concurrency"],
            r["request_rate"],
            r["num_prompts"],
        )
        groups[key].append(r)

    summary_rows = []

    for key in sorted(groups):
        group = groups[key]

        input_len, output_len, conc, rate, num_prompts = key

        valid_runs = [r for r in group if r["valid"]]

        all_ttfts = [
            x
            for r in valid_runs
            for x in r["_ttfts_ms"]
        ]

        all_itls = [
            x
            for r in valid_runs
            for x in r["_itls_ms"]
        ]

        row = {
            "mode": args.mode,
            "input_len": input_len,
            "output_len": output_len,
            "concurrency": conc,
            "request_rate": rate,
            "num_prompts_per_run": num_prompts,
            "runs": len(group),
            "valid_runs": len(valid_runs),
            "pooled_requests": len(all_ttfts),
            "pooled_itl_samples": len(all_itls),
        }

        metrics = [
            "request_throughput",
            "output_throughput",
            "total_throughput",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p95_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "p95_itl_ms",
            "p99_itl_ms",
            "max_itl_ms",
        ]

        for metric in metrics:
            vals = [r[metric] for r in valid_runs]
            mean, std = mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        # Recompute percentiles from pooled raw samples.
        row["pooled_p50_ttft_ms"] = pct(all_ttfts, 50)
        row["pooled_p95_ttft_ms"] = pct(all_ttfts, 95)
        row["pooled_p99_ttft_ms"] = pct(all_ttfts, 99)

        row["pooled_p50_itl_ms"] = pct(all_itls, 50)
        row["pooled_p95_itl_ms"] = pct(all_itls, 95)
        row["pooled_p99_itl_ms"] = pct(all_itls, 99)
        row["pooled_p999_itl_ms"] = pct(all_itls, 99.9)
        row["pooled_max_itl_ms"] = max(all_itls) if all_itls else math.nan

        summary_rows.append(row)

    summary_path = output_dir / f"{args.mode}_capacity_summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print()
    print("=" * 110)
    print(f"{args.mode.upper()} CAPACITY SUMMARY")
    print("=" * 110)

    header = (
        f"{'Input':>6} "
        f"{'C':>3} "
        f"{'Runs':>5} "
        f"{'Req/s':>10} "
        f"{'Out tok/s':>11} "
        f"{'P99 TTFT':>11} "
        f"{'P99 TPOT':>11} "
        f"{'P99 ITL':>10} "
        f"{'P99.9 ITL':>11}"
    )
    print(header)
    print("-" * len(header))

    for r in summary_rows:
        print(
            f"{r['input_len']:>6} "
            f"{r['concurrency']:>3} "
            f"{r['valid_runs']:>2}/{r['runs']:<2} "
            f"{r['request_throughput_mean']:>10.2f} "
            f"{r['output_throughput_mean']:>11.2f} "
            f"{r['pooled_p99_ttft_ms']:>11.2f} "
            f"{r['p99_tpot_ms_mean']:>11.2f} "
            f"{r['pooled_p99_itl_ms']:>10.2f} "
            f"{r['pooled_p999_itl_ms']:>11.2f}"
        )

    invalid = [r for r in runs if not r["valid"]]

    print()
    print(f"Total runs : {len(runs)}")
    print(f"Valid runs : {len(runs) - len(invalid)}")
    print(f"Invalid    : {len(invalid)}")

    if invalid:
        print()
        print("INVALID RUNS:")
        for r in invalid:
            print(
                f"  {r['file']}: "
                f"completed={r['completed']}, "
                f"expected={r['num_prompts']}"
            )

    print()
    print(f"Per-run CSV : {per_run_path}")
    print(f"Summary CSV : {summary_path}")


if __name__ == "__main__":
    main()
