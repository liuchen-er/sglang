import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

RESULT_DIR = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/results/l3_final"
)

COST_LOG = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/server_l3_final_cost_model_v3.log"
)

POLICIES = (
    "always_restore",
    "always_recompute",
    "cost_model",
)

CONCURRENCIES = (1, 16, 32)


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

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            prefix = int(row["prefix_len"])
            trial = int(row["trial"])

            if row.get("record_type") == "request":
                requests[(prefix, trial)].append(row)
            elif row.get("record_type") == "summary":
                summaries[(prefix, trial)] = row

    return requests, summaries


def validate_baseline(policy, summaries):
    errors = []

    for (prefix, trial), row in sorted(summaries.items()):
        prefetch = float(row.get("storage_prefetch_delta", 0.0))
        expected = float(row.get("expected_storage_prefetch", 0.0))
        load_back = float(row.get("load_back_delta", 0.0))

        if policy == "always_restore":
            if abs(prefetch - expected) > 1e-6:
                errors.append(
                    f"restore prefix={prefix} trial={trial}: "
                    f"prefetch={prefetch:.0f}, expected={expected:.0f}"
                )
            if load_back <= 0:
                errors.append(
                    f"restore prefix={prefix} trial={trial}: "
                    f"load_back={load_back:.0f}"
                )

        elif policy == "always_recompute":
            if abs(prefetch) > 1e-6:
                errors.append(
                    f"recompute prefix={prefix} trial={trial}: "
                    f"prefetch={prefetch:.0f}, expected=0"
                )
            if abs(load_back) > 1e-6:
                errors.append(
                    f"recompute prefix={prefix} trial={trial}: "
                    f"load_back={load_back:.0f}, expected=0"
                )

    return errors


def aggregate(requests):
    out = {}

    prefixes = sorted({prefix for prefix, _ in requests})

    for prefix in prefixes:
        pooled = []
        trial_p50 = {}

        for p, trial in sorted(requests):
            if p != prefix:
                continue

            values = [
                float(row["ttft_ms"])
                for row in requests[(p, trial)]
            ]

            pooled.extend(values)
            trial_p50[trial] = median(values)

        out[prefix] = {
            "n": len(pooled),
            "mean": mean(pooled),
            "p50": median(pooled),
            "p95": percentile(pooled, 0.95),
            "trial_p50": trial_p50,
        }

    return out


def parse_cost_actions():
    actions = defaultdict(lambda: {"restore": 0, "recompute": 0})

    if not COST_LOG.exists():
        return actions

    pattern = re.compile(
        r"\[HiCacheEarlyDecision\] "
        r"policy=cost_model "
        r"action=(restore|recompute).*?"
        r"rid=agent_target_cost_model_c(\d+)_"
        r"t\d+_s\d+_p(\d+)"
    )

    with COST_LOG.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            action = m.group(1)
            concurrency = int(m.group(2))
            prefix = int(m.group(3))
            actions[(concurrency, prefix)][action] += 1

    return actions


all_data = {}
baseline_errors = []

for policy in POLICIES:
    for concurrency in CONCURRENCIES:
        path = RESULT_DIR / (
            f"l3_final_{policy}_c{concurrency}.jsonl"
        )

        requests, summaries = load_file(path)

        if policy in ("always_restore", "always_recompute"):
            baseline_errors.extend(
                validate_baseline(
                    policy,
                    summaries,
                )
            )

        agg = aggregate(requests)

        for prefix, row in agg.items():
            all_data[
                (concurrency, prefix, policy)
            ] = row


print()
print("=" * 80)
print("BASELINE VALIDATION")
print("=" * 80)

if baseline_errors:
    for error in baseline_errors:
        print("[FAIL]", error)
    raise SystemExit(
        "Baseline validation failed. Do not interpret V3 results."
    )
else:
    print("[PASS] always_restore is pure L3 restore.")
    print("[PASS] always_recompute is true recompute.")


print()
print("=" * 132)
print("FINAL L3 TTFT COMPARISON")
print("=" * 132)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore P50':>12} "
    f"{'Recomp P50':>12} "
    f"{'V3 P50':>12} "
    f"{'Oracle':>10} "
    f"{'V3-Regret':>11} "
    f"{'Regret%':>9} "
    f"{'V3 vs R%':>9} "
    f"{'V3 vs C%':>9}"
)

print("-" * 132)

pairs = sorted(
    {
        (c, p)
        for c, p, policy in all_data
        if policy == "cost_model"
    }
)

for concurrency, prefix in pairs:
    restore = all_data[
        (concurrency, prefix, "always_restore")
    ]
    recompute = all_data[
        (concurrency, prefix, "always_recompute")
    ]
    v3 = all_data[
        (concurrency, prefix, "cost_model")
    ]

    r = restore["p50"]
    c = recompute["p50"]
    v = v3["p50"]

    oracle = min(r, c)
    oracle_action = "R" if r <= c else "C"

    regret = v - oracle
    regret_pct = (
        regret / oracle * 100.0
        if oracle > 0
        else float("nan")
    )

    vs_restore = (v - r) / r * 100.0
    vs_recompute = (v - c) / c * 100.0

    print(
        f"{concurrency:3d} "
        f"{prefix:7d} "
        f"{r:12.3f} "
        f"{c:12.3f} "
        f"{v:12.3f} "
        f"{oracle_action:>10} "
        f"{regret:11.3f} "
        f"{regret_pct:8.2f}% "
        f"{vs_restore:8.2f}% "
        f"{vs_recompute:8.2f}%"
    )


print()
print("=" * 132)
print("P95 COMPARISON")
print("=" * 132)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore P95':>12} "
    f"{'Recomp P95':>12} "
    f"{'V3 P95':>12} "
    f"{'Best Static':>12}"
)

print("-" * 132)

for concurrency, prefix in pairs:
    restore = all_data[
        (concurrency, prefix, "always_restore")
    ]
    recompute = all_data[
        (concurrency, prefix, "always_recompute")
    ]
    v3 = all_data[
        (concurrency, prefix, "cost_model")
    ]

    r = restore["p95"]
    c = recompute["p95"]
    v = v3["p95"]

    best = "restore" if r <= c else "recompute"

    print(
        f"{concurrency:3d} "
        f"{prefix:7d} "
        f"{r:12.3f} "
        f"{c:12.3f} "
        f"{v:12.3f} "
        f"{best:>12}"
    )


actions = parse_cost_actions()

print()
print("=" * 80)
print("V3 EARLY ACTION DISTRIBUTION")
print("=" * 80)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>8} "
    f"{'Recompute':>10} "
    f"{'Total':>7} "
    f"{'Restore%':>9}"
)

print("-" * 80)

for concurrency, prefix in pairs:
    x = actions[(concurrency, prefix)]
    restore_count = x["restore"]
    recompute_count = x["recompute"]
    total = restore_count + recompute_count

    restore_pct = (
        100.0 * restore_count / total
        if total > 0
        else 0.0
    )

    print(
        f"{concurrency:3d} "
        f"{prefix:7d} "
        f"{restore_count:8d} "
        f"{recompute_count:10d} "
        f"{total:7d} "
        f"{restore_pct:8.2f}%"
    )


print()
print("Interpretation:")
print("  Oracle R/C = faster static baseline at that point.")
print("  V3-Regret = V3 P50 - Oracle P50.")
print("  Smaller regret is better; negative means V3 beat both static baselines.")
print("  V3 vs R/C < 0 means V3 is faster than that baseline.")
