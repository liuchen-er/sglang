import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

DIR = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_next")
FILES = {
    "restore": DIR / "l3_restore_c1_boundary.jsonl",
    "recompute": DIR / "l3_recompute_c1_boundary.jsonl",
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
    sums = {}
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
                sums[(prefix, trial)] = x
    return reqs, sums

raw = {}
for policy, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    reqs, sums = load(path)

    for prefix in sorted({p for p, _ in reqs}):
        pooled = []
        trial_p50 = {}
        valid = True

        for trial in sorted(t for p, t in reqs if p == prefix):
            rows = reqs[(prefix, trial)]
            vals = [float(x["ttft_ms"]) for x in rows]
            pooled.extend(vals)
            trial_p50[trial] = median(vals)

            s = sums.get((prefix, trial), {})
            if policy == "restore":
                actual = float(s.get("storage_prefetch_delta", -1))
                expected = float(s.get("expected_storage_prefetch", -2))
                if actual < 0 or expected < 0 or abs(actual - expected) > 1e-6:
                    valid = False
            else:
                load_back = float(s.get("load_back_delta", -1))
                if abs(load_back) > 1e-6:
                    valid = False

        raw[(prefix, policy)] = {
            "n": len(pooled),
            "mean": mean(pooled),
            "p50": median(pooled),
            "p95": percentile(pooled, 0.95),
            "trial_p50": trial_p50,
            "valid": valid,
        }

print("\n================ C1 BOUNDARY RAW ================\n")
print(
    f"{'Prefix':>8} {'Policy':>10} {'N':>4} "
    f"{'Mean':>10} {'P50':>10} {'P95':>10} {'Valid':>7}"
)
print("-" * 70)

for prefix, policy in sorted(raw):
    x = raw[(prefix, policy)]
    print(
        f"{prefix:8d} {policy:>10} {x['n']:4d} "
        f"{x['mean']:10.3f} {x['p50']:10.3f} "
        f"{x['p95']:10.3f} "
        f"{('PASS' if x['valid'] else 'FAIL'):>7}"
    )

print("\n================ C1 RESTORE vs RECOMPUTE ================\n")
print(
    f"{'Prefix':>8} "
    f"{'R Mean':>10} {'C Mean':>10} {'ΔMean':>10} "
    f"{'R P50':>10} {'C P50':>10} {'ΔP50':>10} "
    f"{'R P95':>10} {'C P95':>10} {'ΔP95':>10} "
    f"{'Trial ΔP50':>28}"
)
print("-" * 135)

for prefix in sorted({p for p, _ in raw}):
    r = raw.get((prefix, "restore"))
    c = raw.get((prefix, "recompute"))
    if r is None or c is None:
        continue

    dm = r["mean"] - c["mean"]
    d50 = r["p50"] - c["p50"]
    d95 = r["p95"] - c["p95"]

    trials = sorted(set(r["trial_p50"]) & set(c["trial_p50"]))
    trial_diff = [
        r["trial_p50"][t] - c["trial_p50"][t]
        for t in trials
    ]
    trial_text = ",".join(f"{x:+.1f}" for x in trial_diff)

    print(
        f"{prefix:8d} "
        f"{r['mean']:10.3f} {c['mean']:10.3f} {dm:10.3f} "
        f"{r['p50']:10.3f} {c['p50']:10.3f} {d50:10.3f} "
        f"{r['p95']:10.3f} {c['p95']:10.3f} {d95:10.3f} "
        f"{trial_text:>28}"
    )

print("\nΔ = Restore - Recompute")
print("Δ < 0 : Restore faster")
print("Δ > 0 : Recompute faster")
