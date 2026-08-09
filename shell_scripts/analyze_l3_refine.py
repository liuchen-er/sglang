import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

RESULT_DIR = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_refine"
)

FILES = {
    (1, "restore"): RESULT_DIR / "l3_restore_c1_refine.jsonl",
    (1, "recompute"): RESULT_DIR / "l3_recompute_c1_refine.jsonl",
    (16, "restore"): RESULT_DIR / "l3_restore_c16_refine.jsonl",
    (16, "recompute"): RESULT_DIR / "l3_recompute_c16_refine.jsonl",
    (32, "restore"): RESULT_DIR / "l3_restore_c32_refine.jsonl",
    (32, "recompute"): RESULT_DIR / "l3_recompute_c32_refine.jsonl",
}

CSV_PATH = RESULT_DIR / "l3_refine_summary.csv"


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_file(path):
    requests = defaultdict(list)
    summaries = {}

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            prefix = int(x["prefix_len"])
            trial = int(x["trial"])

            if x.get("record_type") == "request":
                requests[(prefix, trial)].append(x)

            elif x.get("record_type") == "summary":
                summaries[(prefix, trial)] = x

    return requests, summaries


def metric_sign(diff, baseline):
    threshold = max(2.0, 0.03 * baseline)

    if diff < -threshold:
        return "restore"

    if diff > threshold:
        return "recompute"

    return "tie"


all_data = {}

for key, path in FILES.items():
    concurrency, policy = key

    reqs, summaries = load_file(path)

    prefixes = sorted(
        set(prefix for prefix, _ in reqs.keys())
    )

    for prefix in prefixes:
        trial_metrics = {}

        for trial in sorted(
            t for p, t in reqs.keys() if p == prefix
        ):
            rows = reqs[(prefix, trial)]
            ttft = [float(x["ttft_ms"]) for x in rows]

            summary = summaries.get((prefix, trial), {})

            if policy == "restore":
                prefetch = float(
                    summary.get("storage_prefetch_delta", -1)
                )
                expected = float(
                    summary.get("expected_storage_prefetch", -2)
                )
                valid = (
                    prefetch >= 0
                    and expected >= 0
                    and abs(prefetch - expected) < 1e-6
                )
                validation = (
                    f"{prefetch:.0f}/{expected:.0f}"
                )
            else:
                load_back = float(
                    summary.get("load_back_delta", -1)
                )
                valid = abs(load_back) < 1e-6
                validation = f"loadback={load_back:.0f}"

            trial_metrics[trial] = {
                "n": len(ttft),
                "mean": mean(ttft),
                "p50": median(ttft),
                "p95": percentile(ttft, 0.95),
                "max": max(ttft),
                "valid": valid,
                "validation": validation,
            }

        pooled = []

        for trial in sorted(trial_metrics):
            pooled.extend(
                float(x["ttft_ms"])
                for x in reqs[(prefix, trial)]
            )

        all_data[(concurrency, prefix, policy)] = {
            "n": len(pooled),
            "mean": mean(pooled),
            "p50": median(pooled),
            "p95": percentile(pooled, 0.95),
            "max": max(pooled),
            "trials": trial_metrics,
            "valid": all(
                x["valid"] for x in trial_metrics.values()
            ),
        }


print()
print("=" * 118)
print("L3 REFINE RAW RESULTS")
print("=" * 118)

print(
    f"{'C':>4} "
    f"{'Prefix':>8} "
    f"{'Policy':>10} "
    f"{'N':>4} "
    f"{'Mean':>12} "
    f"{'P50':>12} "
    f"{'P95':>12} "
    f"{'Max':>12} "
    f"{'Valid':>8}"
)

print("-" * 118)

for key in sorted(all_data):
    c, prefix, policy = key
    x = all_data[key]

    print(
        f"{c:4d} "
        f"{prefix:8d} "
        f"{policy:>10} "
        f"{x['n']:4d} "
        f"{x['mean']:12.3f} "
        f"{x['p50']:12.3f} "
        f"{x['p95']:12.3f} "
        f"{x['max']:12.3f} "
        f"{('PASS' if x['valid'] else 'FAIL'):>8}"
    )


print()
print("=" * 156)
print("RESTORE vs RECOMPUTE")
print("=" * 156)

header = (
    f"{'C':>4} "
    f"{'Prefix':>8} "
    f"{'R Mean':>11} "
    f"{'C Mean':>11} "
    f"{'ΔMean':>11} "
    f"{'R P50':>11} "
    f"{'C P50':>11} "
    f"{'ΔP50':>11} "
    f"{'R P95':>11} "
    f"{'C P95':>11} "
    f"{'ΔP95':>11} "
    f"{'Trial P50 Δ':>31} "
    f"{'Decision':>12}"
)

print(header)
print("-" * 156)

csv_rows = []
summary_by_c = defaultdict(list)

pairs = sorted(
    set(
        (c, prefix)
        for c, prefix, policy in all_data.keys()
    )
)

for c, prefix in pairs:
    r = all_data.get((c, prefix, "restore"))
    q = all_data.get((c, prefix, "recompute"))

    if r is None or q is None:
        continue

    mean_diff = r["mean"] - q["mean"]
    p50_diff = r["p50"] - q["p50"]
    p95_diff = r["p95"] - q["p95"]

    mean_sign = metric_sign(
        mean_diff,
        min(r["mean"], q["mean"]),
    )

    p50_sign = metric_sign(
        p50_diff,
        min(r["p50"], q["p50"]),
    )

    p95_sign = metric_sign(
        p95_diff,
        min(r["p95"], q["p95"]),
    )

    trial_diffs = []
    trial_votes = []

    common_trials = sorted(
        set(r["trials"]) & set(q["trials"])
    )

    for trial in common_trials:
        rd = r["trials"][trial]["p50"]
        qd = q["trials"][trial]["p50"]

        diff = rd - qd
        trial_diffs.append(diff)

        trial_votes.append(
            metric_sign(diff, min(rd, qd))
        )

    metric_votes = [mean_sign, p50_sign, p95_sign]

    non_tie_metric_votes = [
        x for x in metric_votes if x != "tie"
    ]

    non_tie_trial_votes = [
        x for x in trial_votes if x != "tie"
    ]

    if not r["valid"] or not q["valid"]:
        decision = "INVALID"

    elif (
        len(non_tie_metric_votes) == 3
        and len(set(non_tie_metric_votes)) == 1
        and len(non_tie_trial_votes) >= 2
        and len(set(non_tie_trial_votes)) == 1
        and non_tie_metric_votes[0] == non_tie_trial_votes[0]
    ):
        decision = non_tie_metric_votes[0]

    else:
        decision = "uncertain"

    trial_diff_text = ",".join(
        f"{x:+.1f}" for x in trial_diffs
    )

    print(
        f"{c:4d} "
        f"{prefix:8d} "
        f"{r['mean']:11.3f} "
        f"{q['mean']:11.3f} "
        f"{mean_diff:11.3f} "
        f"{r['p50']:11.3f} "
        f"{q['p50']:11.3f} "
        f"{p50_diff:11.3f} "
        f"{r['p95']:11.3f} "
        f"{q['p95']:11.3f} "
        f"{p95_diff:11.3f} "
        f"{trial_diff_text:>31} "
        f"{decision:>12}"
    )

    summary_by_c[c].append(
        {
            "prefix": prefix,
            "decision": decision,
            "p50_diff": p50_diff,
        }
    )

    csv_rows.append(
        {
            "concurrency": c,
            "prefix": prefix,
            "restore_mean_ms": r["mean"],
            "recompute_mean_ms": q["mean"],
            "mean_diff_ms": mean_diff,
            "restore_p50_ms": r["p50"],
            "recompute_p50_ms": q["p50"],
            "p50_diff_ms": p50_diff,
            "restore_p95_ms": r["p95"],
            "recompute_p95_ms": q["p95"],
            "p95_diff_ms": p95_diff,
            "trial_p50_diffs_ms": trial_diff_text,
            "restore_valid": r["valid"],
            "recompute_valid": q["valid"],
            "decision": decision,
        }
    )


print()
print("=" * 80)
print("CROSSOVER SUMMARY")
print("=" * 80)

for c in sorted(summary_by_c):
    print()
    print(f"Concurrency = {c}")

    rows = sorted(
        summary_by_c[c],
        key=lambda x: x["prefix"],
    )

    for row in rows:
        print(
            f"  prefix={row['prefix']:5d} "
            f"decision={row['decision']:10s} "
            f"ΔP50={row['p50_diff']:+9.3f} ms"
        )

    certain = [
        row
        for row in rows
        if row["decision"] in ("restore", "recompute")
    ]

    transitions = []

    for a, b in zip(certain, certain[1:]):
        if a["decision"] != b["decision"]:
            transitions.append(
                (
                    a["prefix"],
                    b["prefix"],
                    a["decision"],
                    b["decision"],
                )
            )

    if not transitions:
        print("  crossover: not clearly observed")
    else:
        for p0, p1, d0, d1 in transitions:
            print(
                f"  crossover candidate: "
                f"{p0} -> {p1} tokens "
                f"({d0} -> {d1})"
            )


with CSV_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "concurrency",
            "prefix",
            "restore_mean_ms",
            "recompute_mean_ms",
            "mean_diff_ms",
            "restore_p50_ms",
            "recompute_p50_ms",
            "p50_diff_ms",
            "restore_p95_ms",
            "recompute_p95_ms",
            "p95_diff_ms",
            "trial_p50_diffs_ms",
            "restore_valid",
            "recompute_valid",
            "decision",
        ],
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print()
print(f"CSV written to: {CSV_PATH}")
