import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

DIR = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/results/l3_true_profile"
)

FILES = {
    (1, "restore"): DIR / "l3_true_restore_c1.jsonl",
    (1, "recompute"): DIR / "l3_true_recompute_c1.jsonl",
    (16, "restore"): DIR / "l3_true_restore_c16.jsonl",
    (16, "recompute"): DIR / "l3_true_recompute_c16.jsonl",
    (32, "restore"): DIR / "l3_true_restore_c32.jsonl",
    (32, "recompute"): DIR / "l3_true_recompute_c32.jsonl",
}

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

def load(path):
    reqs = defaultdict(list)
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
                reqs[(prefix, trial)].append(x)
            elif x.get("record_type") == "summary":
                summaries[(prefix, trial)] = x

    return reqs, summaries

data = {}

for (c, policy), path in FILES.items():
    reqs, summaries = load(path)

    prefixes = sorted({p for p, _ in reqs})

    for prefix in prefixes:
        pooled = []
        trial_p50 = {}
        valid = True

        trials = sorted(t for p, t in reqs if p == prefix)

        for trial in trials:
            rows = reqs[(prefix, trial)]
            ttft = [float(x["ttft_ms"]) for x in rows]
            pooled.extend(ttft)
            trial_p50[trial] = median(ttft)

            s = summaries.get((prefix, trial), {})

            prefetch = float(
                s.get("storage_prefetch_delta", -1)
            )
            load_back = float(
                s.get("load_back_delta", -1)
            )

            if policy == "restore":
                expected = float(
                    s.get("expected_storage_prefetch", -2)
                )
                if (
                    prefetch < 0
                    or expected < 0
                    or abs(prefetch - expected) > 1e-6
                    or load_back <= 0
                ):
                    valid = False

            else:
                if (
                    abs(prefetch) > 1e-6
                    or abs(load_back) > 1e-6
                ):
                    valid = False

        data[(c, prefix, policy)] = {
            "n": len(pooled),
            "mean": mean(pooled),
            "p50": median(pooled),
            "p95": percentile(pooled, 0.95),
            "max": max(pooled),
            "trial_p50": trial_p50,
            "valid": valid,
        }

print()
print("=" * 112)
print("TRUE L3 PROFILE RAW")
print("=" * 112)

print(
    f"{'C':>4} {'Prefix':>8} {'Policy':>10} {'N':>4} "
    f"{'Mean':>11} {'P50':>11} {'P95':>11} "
    f"{'Max':>11} {'Valid':>8}"
)
print("-" * 112)

for key in sorted(data):
    c, prefix, policy = key
    x = data[key]

    print(
        f"{c:4d} {prefix:8d} {policy:>10} {x['n']:4d} "
        f"{x['mean']:11.3f} {x['p50']:11.3f} "
        f"{x['p95']:11.3f} {x['max']:11.3f} "
        f"{('PASS' if x['valid'] else 'FAIL'):>8}"
    )

print()
print("=" * 150)
print("TRUE RESTORE vs TRUE RECOMPUTE")
print("=" * 150)

print(
    f"{'C':>4} {'Prefix':>8} "
    f"{'R Mean':>11} {'C Mean':>11} {'ΔMean':>11} "
    f"{'R P50':>11} {'C P50':>11} {'ΔP50':>11} "
    f"{'R P95':>11} {'C P95':>11} {'ΔP95':>11} "
    f"{'Trial ΔP50':>28}"
)
print("-" * 150)

for c, prefix in sorted(
    {(c, p) for c, p, _ in data.keys()}
):
    r = data.get((c, prefix, "restore"))
    q = data.get((c, prefix, "recompute"))

    if r is None or q is None:
        continue

    dm = r["mean"] - q["mean"]
    d50 = r["p50"] - q["p50"]
    d95 = r["p95"] - q["p95"]

    common_trials = sorted(
        set(r["trial_p50"]) & set(q["trial_p50"])
    )

    trial_diffs = [
        r["trial_p50"][t] - q["trial_p50"][t]
        for t in common_trials
    ]

    trial_text = ",".join(
        f"{x:+.1f}" for x in trial_diffs
    )

    print(
        f"{c:4d} {prefix:8d} "
        f"{r['mean']:11.3f} {q['mean']:11.3f} {dm:11.3f} "
        f"{r['p50']:11.3f} {q['p50']:11.3f} {d50:11.3f} "
        f"{r['p95']:11.3f} {q['p95']:11.3f} {d95:11.3f} "
        f"{trial_text:>28}"
    )

print()
print("Δ = Restore - Recompute")
print("Δ < 0 : Restore faster")
print("Δ > 0 : Recompute faster")
