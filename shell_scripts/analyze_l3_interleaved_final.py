import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")
RESULT_ROOT = ROOT / "results/l3_interleaved"
LOG_DIR = ROOT / "logs"

ROUNDS = (1, 2, 3)
POLICIES = ("restore", "recompute", "v3", "v31")
CONCURRENCIES = (1, 16, 32)

PREFIXES = {
    1: (256, 8192, 16384),
    16: (256, 4096, 8192, 16384),
    32: (256, 4096, 8192),
}

EXPECTED_REQUESTS = 32

CSV_OUT = RESULT_ROOT / "final_summary.csv"
JSON_OUT = RESULT_ROOT / "final_summary.json"


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


def load_jsonl(path):
    requests = defaultdict(list)
    summaries = {}

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON: {path}:{line_no}: {e}")

            prefix = int(row["prefix_len"])

            if row.get("record_type") == "request":
                requests[prefix].append(row)
            elif row.get("record_type") == "summary":
                summaries[prefix] = row

    return requests, summaries


def request_stats(rows):
    ttfts = [float(x["ttft_ms"]) for x in rows]
    return {
        "n": len(ttfts),
        "mean": mean(ttfts),
        "p50": median(ttfts),
        "p95": percentile(ttfts, 0.95),
        "max": max(ttfts),
    }


def spread_pct(values):
    m = median(values)
    if m <= 0:
        return 0.0
    return (max(values) - min(values)) / m * 100.0


def parse_actions(round_id, label):
    path = LOG_DIR / f"server_final_r{round_id}_{label}.log"

    if not path.exists():
        return {}, [f"Missing log: {path}"]

    pattern = re.compile(
        r"\[HiCacheEarlyDecision\] "
        r"policy=cost_model "
        r"action=(restore|recompute).*?"
        r"rid=agent_target_cost_model_c(\d+)_t\d+_s\d+_p(\d+)"
    )

    counts = defaultdict(lambda: {"restore": 0, "recompute": 0})

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            action = m.group(1)
            concurrency = int(m.group(2))
            prefix = int(m.group(3))
            counts[(concurrency, prefix)][action] += 1

    return counts, []


data = {}
validation_errors = []

print()
print("=" * 105)
print("FINAL 3-ROUND VALIDATION")
print("=" * 105)

for round_id in ROUNDS:
    print(f"\nRound {round_id}:")

    for policy in POLICIES:
        for c in CONCURRENCIES:
            path = RESULT_ROOT / f"round{round_id}" / f"{policy}_c{c}.jsonl"

            try:
                requests, summaries = load_jsonl(path)
            except Exception as e:
                validation_errors.append(str(e))
                continue

            expected_prefixes = list(PREFIXES[c])

            if sorted(requests) != expected_prefixes:
                validation_errors.append(
                    f"{path.name}: request prefixes={sorted(requests)}, "
                    f"expected={expected_prefixes}"
                )

            if sorted(summaries) != expected_prefixes:
                validation_errors.append(
                    f"{path.name}: summary prefixes={sorted(summaries)}, "
                    f"expected={expected_prefixes}"
                )

            for prefix in PREFIXES[c]:
                rows = requests.get(prefix, [])
                summary = summaries.get(prefix)

                if len(rows) != EXPECTED_REQUESTS:
                    validation_errors.append(
                        f"round={round_id} policy={policy} c={c} prefix={prefix}: "
                        f"requests={len(rows)}, expected={EXPECTED_REQUESTS}"
                    )
                    continue

                if summary is None:
                    validation_errors.append(
                        f"round={round_id} policy={policy} c={c} prefix={prefix}: "
                        f"summary missing"
                    )
                    continue

                prefetch = float(summary.get("storage_prefetch_delta", -1))
                expected_prefetch = float(
                    summary.get("expected_storage_prefetch", -1)
                )
                load_back = float(summary.get("load_back_delta", -1))

                if policy == "restore":
                    if abs(prefetch - expected_prefetch) > 1e-6:
                        validation_errors.append(
                            f"restore round={round_id} c={c} prefix={prefix}: "
                            f"prefetch={prefetch:.0f}, "
                            f"expected={expected_prefetch:.0f}"
                        )
                    if load_back <= 0:
                        validation_errors.append(
                            f"restore round={round_id} c={c} prefix={prefix}: "
                            f"load_back={load_back:.0f}, expected > 0"
                        )

                elif policy == "recompute":
                    if abs(prefetch) > 1e-6:
                        validation_errors.append(
                            f"recompute round={round_id} c={c} prefix={prefix}: "
                            f"prefetch={prefetch:.0f}, expected=0"
                        )
                    if abs(load_back) > 1e-6:
                        validation_errors.append(
                            f"recompute round={round_id} c={c} prefix={prefix}: "
                            f"load_back={load_back:.0f}, expected=0"
                        )

                data[(round_id, policy, c, prefix)] = {
                    "stats": request_stats(rows),
                    "summary": summary,
                }

            print(f"  [OK] {path.name}")

if validation_errors:
    print()
    print("[FAIL] Final validation failed:")
    for error in validation_errors:
        print("  -", error)
    raise SystemExit("\nDo not use these results for the final report.")

print()
print("[PASS] All 36 result files are complete.")
print("[PASS] All always_restore runs are pure L3 restore.")
print("[PASS] All always_recompute runs are true recompute.")


final = {}

for c in CONCURRENCIES:
    for prefix in PREFIXES[c]:
        for policy in POLICIES:
            round_p50 = [
                data[(r, policy, c, prefix)]["stats"]["p50"]
                for r in ROUNDS
            ]
            round_p95 = [
                data[(r, policy, c, prefix)]["stats"]["p95"]
                for r in ROUNDS
            ]
            round_mean = [
                data[(r, policy, c, prefix)]["stats"]["mean"]
                for r in ROUNDS
            ]
            round_throughput = [
                float(
                    data[(r, policy, c, prefix)]["summary"][
                        "request_throughput"
                    ]
                )
                for r in ROUNDS
            ]

            final[(policy, c, prefix)] = {
                "p50": median(round_p50),
                "p95": median(round_p95),
                "mean": median(round_mean),
                "throughput": median(round_throughput),
                "round_p50": round_p50,
                "round_p95": round_p95,
                "p50_spread_pct": spread_pct(round_p50),
            }


print()
print("=" * 165)
print("FINAL CROSS-ROUND P50 TTFT")
print("Final value = median(Round1 P50, Round2 P50, Round3 P50)")
print("=" * 165)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>11} {'Recompute':>11} "
    f"{'V3':>11} {'V3.1':>11} "
    f"{'Best':>7} "
    f"{'V3 Gap%':>10} {'V3.1 Gap%':>12} "
    f"{'V3.1 vs V3%':>14}"
)
print("-" * 165)

final_rows = []
v3_gaps = []
v31_gaps = []

for c in CONCURRENCIES:
    for prefix in PREFIXES[c]:
        r = final[("restore", c, prefix)]["p50"]
        q = final[("recompute", c, prefix)]["p50"]
        v3 = final[("v3", c, prefix)]["p50"]
        v31 = final[("v31", c, prefix)]["p50"]

        best_static = min(r, q)
        best_action = "R" if r <= q else "C"

        v3_gap = (v3 - best_static) / best_static * 100.0
        v31_gap = (v31 - best_static) / best_static * 100.0
        v31_vs_v3 = (v31 - v3) / v3 * 100.0

        v3_gaps.append(v3_gap)
        v31_gaps.append(v31_gap)

        row = {
            "concurrency": c,
            "prefix": prefix,
            "restore_p50_ms": r,
            "recompute_p50_ms": q,
            "v3_p50_ms": v3,
            "v31_p50_ms": v31,
            "best_static_action": best_action,
            "best_static_p50_ms": best_static,
            "v3_best_static_gap_pct": v3_gap,
            "v31_best_static_gap_pct": v31_gap,
            "v31_vs_v3_pct": v31_vs_v3,
        }
        final_rows.append(row)

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:11.3f} {q:11.3f} "
            f"{v3:11.3f} {v31:11.3f} "
            f"{best_action:>7} "
            f"{v3_gap:9.2f}% "
            f"{v31_gap:11.2f}% "
            f"{v31_vs_v3:13.2f}%"
        )


print()
print("=" * 135)
print("FINAL CROSS-ROUND P95 TTFT")
print("Final value = median(Round1 P95, Round2 P95, Round3 P95)")
print("=" * 135)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>12} {'Recompute':>12} "
    f"{'V3':>12} {'V3.1':>12} "
    f"{'Best Static':>12}"
)
print("-" * 135)

for c in CONCURRENCIES:
    for prefix in PREFIXES[c]:
        r = final[("restore", c, prefix)]["p95"]
        q = final[("recompute", c, prefix)]["p95"]
        v3 = final[("v3", c, prefix)]["p95"]
        v31 = final[("v31", c, prefix)]["p95"]

        best = "restore" if r <= q else "recompute"

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:12.3f} {q:12.3f} "
            f"{v3:12.3f} {v31:12.3f} "
            f"{best:>12}"
        )


print()
print("=" * 145)
print("V3 / V3.1 CROSS-ROUND P50 STABILITY")
print("=" * 145)

print(
    f"{'Policy':>7} {'C':>3} {'Prefix':>7} "
    f"{'R1':>11} {'R2':>11} {'R3':>11} "
    f"{'Median':>11} {'Min':>11} {'Max':>11} {'Spread%':>9}"
)
print("-" * 145)

for policy, name in (("v3", "V3"), ("v31", "V3.1")):
    for c in CONCURRENCIES:
        for prefix in PREFIXES[c]:
            x = final[(policy, c, prefix)]
            r1, r2, r3 = x["round_p50"]

            print(
                f"{name:>7} {c:3d} {prefix:7d} "
                f"{r1:11.3f} {r2:11.3f} {r3:11.3f} "
                f"{x['p50']:11.3f} "
                f"{min(x['round_p50']):11.3f} "
                f"{max(x['round_p50']):11.3f} "
                f"{x['p50_spread_pct']:8.1f}%"
            )


all_actions = {
    "v3": defaultdict(lambda: {"restore": 0, "recompute": 0}),
    "v31": defaultdict(lambda: {"restore": 0, "recompute": 0}),
}

round_action_pct = {
    "v3": defaultdict(dict),
    "v31": defaultdict(dict),
}

action_errors = []

for round_id in ROUNDS:
    for label in ("v3", "v31"):
        counts, errors = parse_actions(round_id, label)
        action_errors.extend(errors)

        for c in CONCURRENCIES:
            for prefix in PREFIXES[c]:
                x = counts.get(
                    (c, prefix),
                    {"restore": 0, "recompute": 0},
                )

                restore = x["restore"]
                recompute = x["recompute"]
                total = restore + recompute

                if total != EXPECTED_REQUESTS:
                    action_errors.append(
                        f"round={round_id} {label} c={c} prefix={prefix}: "
                        f"actions={total}, expected={EXPECTED_REQUESTS}"
                    )

                all_actions[label][(c, prefix)]["restore"] += restore
                all_actions[label][(c, prefix)]["recompute"] += recompute

                pct = 100.0 * restore / total if total else 0.0
                round_action_pct[label][(c, prefix)][round_id] = pct


print()
print("=" * 125)
print("FINAL COST-MODEL ACTION DISTRIBUTION")
print("Aggregated over 3 independent rounds: 96 requests per workload")
print("=" * 125)

print(
    f"{'Policy':>7} {'C':>3} {'Prefix':>7} "
    f"{'Restore':>8} {'Recompute':>10} {'Restore%':>9} "
    f"{'R1%':>7} {'R2%':>7} {'R3%':>7}"
)
print("-" * 125)

for label, name in (("v3", "V3"), ("v31", "V3.1")):
    for c in CONCURRENCIES:
        for prefix in PREFIXES[c]:
            x = all_actions[label][(c, prefix)]
            total = x["restore"] + x["recompute"]
            pct = 100.0 * x["restore"] / total if total else 0.0

            p1 = round_action_pct[label][(c, prefix)].get(1, 0.0)
            p2 = round_action_pct[label][(c, prefix)].get(2, 0.0)
            p3 = round_action_pct[label][(c, prefix)].get(3, 0.0)

            print(
                f"{name:>7} {c:3d} {prefix:7d} "
                f"{x['restore']:8d} {x['recompute']:10d} "
                f"{pct:8.1f}% "
                f"{p1:6.1f}% {p2:6.1f}% {p3:6.1f}%"
            )

if action_errors:
    print()
    print("[WARN] Action-log validation:")
    for error in action_errors:
        print("  -", error)
else:
    print()
    print("[PASS] All V3/V3.1 action logs cover 32 requests per round.")


print()
print("=" * 125)
print("FINAL POLICY SUMMARY")
print("=" * 125)

v3_better_points = sum(
    v3 < v31
    for v3, v31 in zip(v3_gaps, v31_gaps)
)
v31_better_points = sum(
    v31 < v3
    for v3, v31 in zip(v3_gaps, v31_gaps)
)

print(f"Number of workloads              = {len(v3_gaps)}")
print()
print(f"V3 median Best-Static Gap        = {median(v3_gaps):.2f}%")
print(f"V3.1 median Best-Static Gap      = {median(v31_gaps):.2f}%")
print()
print(f"V3 mean Best-Static Gap          = {mean(v3_gaps):.2f}%")
print(f"V3.1 mean Best-Static Gap        = {mean(v31_gaps):.2f}%")
print()
print(f"V3 worst positive Gap            = {max(v3_gaps):.2f}%")
print(f"V3.1 worst positive Gap          = {max(v31_gaps):.2f}%")
print()
print(f"V3 lower-gap workloads           = {v3_better_points}/{len(v3_gaps)}")
print(f"V3.1 lower-gap workloads         = {v31_better_points}/{len(v31_gaps)}")
print()
print(
    "Median-gap improvement V3→V3.1 = "
    f"{median(v3_gaps) - median(v31_gaps):.2f} percentage points"
)
print(
    "Mean-gap improvement V3→V3.1   = "
    f"{mean(v3_gaps) - mean(v31_gaps):.2f} percentage points"
)
print(
    "Worst-gap reduction V3→V3.1    = "
    f"{max(v3_gaps) - max(v31_gaps):.2f} percentage points"
)

print()
print(
    "Negative Best-Static Gap means the dynamic policy was faster "
    "than both static policies in that workload."
)


RESULT_ROOT.mkdir(parents=True, exist_ok=True)

csv_fields = [
    "concurrency",
    "prefix",
    "restore_p50_ms",
    "recompute_p50_ms",
    "v3_p50_ms",
    "v31_p50_ms",
    "best_static_action",
    "best_static_p50_ms",
    "v3_best_static_gap_pct",
    "v31_best_static_gap_pct",
    "v31_vs_v3_pct",
    "restore_p95_ms",
    "recompute_p95_ms",
    "v3_p95_ms",
    "v31_p95_ms",
    "v3_round1_p50_ms",
    "v3_round2_p50_ms",
    "v3_round3_p50_ms",
    "v31_round1_p50_ms",
    "v31_round2_p50_ms",
    "v31_round3_p50_ms",
    "v3_p50_spread_pct",
    "v31_p50_spread_pct",
    "v3_restore_pct",
    "v31_restore_pct",
]

with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()

    for row in final_rows:
        c = row["concurrency"]
        prefix = row["prefix"]

        v3_action = all_actions["v3"][(c, prefix)]
        v31_action = all_actions["v31"][(c, prefix)]

        v3_total = v3_action["restore"] + v3_action["recompute"]
        v31_total = v31_action["restore"] + v31_action["recompute"]

        writer.writerow(
            {
                **row,
                "restore_p95_ms": final[("restore", c, prefix)]["p95"],
                "recompute_p95_ms": final[("recompute", c, prefix)]["p95"],
                "v3_p95_ms": final[("v3", c, prefix)]["p95"],
                "v31_p95_ms": final[("v31", c, prefix)]["p95"],
                "v3_round1_p50_ms": final[("v3", c, prefix)]["round_p50"][0],
                "v3_round2_p50_ms": final[("v3", c, prefix)]["round_p50"][1],
                "v3_round3_p50_ms": final[("v3", c, prefix)]["round_p50"][2],
                "v31_round1_p50_ms": final[("v31", c, prefix)]["round_p50"][0],
                "v31_round2_p50_ms": final[("v31", c, prefix)]["round_p50"][1],
                "v31_round3_p50_ms": final[("v31", c, prefix)]["round_p50"][2],
                "v3_p50_spread_pct": final[("v3", c, prefix)][
                    "p50_spread_pct"
                ],
                "v31_p50_spread_pct": final[("v31", c, prefix)][
                    "p50_spread_pct"
                ],
                "v3_restore_pct": (
                    100.0 * v3_action["restore"] / v3_total
                    if v3_total else 0.0
                ),
                "v31_restore_pct": (
                    100.0 * v31_action["restore"] / v31_total
                    if v31_total else 0.0
                ),
            }
        )


json_rows = []

for row in final_rows:
    c = row["concurrency"]
    prefix = row["prefix"]

    item = dict(row)

    for policy in POLICIES:
        x = final[(policy, c, prefix)]
        item[policy] = {
            "final_p50_ms": x["p50"],
            "final_p95_ms": x["p95"],
            "round_p50_ms": x["round_p50"],
            "round_p95_ms": x["round_p95"],
            "p50_spread_pct": x["p50_spread_pct"],
        }

    for label in ("v3", "v31"):
        x = all_actions[label][(c, prefix)]
        total = x["restore"] + x["recompute"]

        item[f"{label}_actions"] = {
            "restore": x["restore"],
            "recompute": x["recompute"],
            "restore_pct": (
                100.0 * x["restore"] / total
                if total else 0.0
            ),
            "round_restore_pct": round_action_pct[label][(c, prefix)],
        }

    json_rows.append(item)


json_output = {
    "aggregation": (
        "For each policy/workload, compute P50/P95 independently in each "
        "round, then take the median of the three round-level values."
    ),
    "best_static_definition": "min(always_restore, true_recompute)",
    "summary": {
        "num_workloads": len(v3_gaps),
        "v3_median_best_static_gap_pct": median(v3_gaps),
        "v31_median_best_static_gap_pct": median(v31_gaps),
        "v3_mean_best_static_gap_pct": mean(v3_gaps),
        "v31_mean_best_static_gap_pct": mean(v31_gaps),
        "v3_worst_positive_gap_pct": max(v3_gaps),
        "v31_worst_positive_gap_pct": max(v31_gaps),
        "v3_lower_gap_workloads": v3_better_points,
        "v31_lower_gap_workloads": v31_better_points,
    },
    "workloads": json_rows,
}

with JSON_OUT.open("w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

print()
print("=" * 105)
print("OUTPUT FILES")
print("=" * 105)
print(f"CSV : {CSV_OUT}")
print(f"JSON: {JSON_OUT}")
