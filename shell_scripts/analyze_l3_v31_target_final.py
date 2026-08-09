import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")

OLD_DIR = ROOT / "results/l3_final"
V31_DIR = ROOT / "results/l3_v31_target_final"

FILES = {
    (16, "restore"): OLD_DIR / "l3_final_always_restore_c16.jsonl",
    (16, "recompute"): OLD_DIR / "l3_final_always_recompute_c16.jsonl",
    (16, "v3"): OLD_DIR / "l3_final_cost_model_c16.jsonl",
    (16, "v31"): V31_DIR / "l3_v31_c16.jsonl",

    (32, "restore"): OLD_DIR / "l3_final_always_restore_c32.jsonl",
    (32, "recompute"): OLD_DIR / "l3_final_always_recompute_c32.jsonl",
    (32, "v3"): OLD_DIR / "l3_final_cost_model_c32.jsonl",
    (32, "v31"): V31_DIR / "l3_v31_c32.jsonl",
}

TARGETS = {
    16: {4096, 16384},
    32: {4096, 8192},
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


def load_requests(path, targets):
    data = defaultdict(list)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("record_type") != "request":
                continue

            prefix = int(row["prefix_len"])

            if prefix not in targets:
                continue

            data[prefix].append(float(row["ttft_ms"]))

    return data


stats = {}

for (c, policy), path in FILES.items():
    rows = load_requests(path, TARGETS[c])

    for prefix, values in rows.items():
        stats[(c, prefix, policy)] = {
            "n": len(values),
            "mean": mean(values),
            "p50": median(values),
            "p95": percentile(values, 0.95),
        }


print()
print("=" * 150)
print("V3.1 TARGET FINAL - P50")
print("=" * 150)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>11} {'Recompute':>11} "
    f"{'V3':>11} {'V3.1':>11} "
    f"{'Oracle':>8} "
    f"{'V3 Regret%':>12} "
    f"{'V3.1 Regret%':>14} "
    f"{'V3.1 vs V3%':>14}"
)
print("-" * 150)

for c in sorted(TARGETS):
    for prefix in sorted(TARGETS[c]):
        r = stats[(c, prefix, "restore")]["p50"]
        q = stats[(c, prefix, "recompute")]["p50"]
        v3 = stats[(c, prefix, "v3")]["p50"]
        v31 = stats[(c, prefix, "v31")]["p50"]

        oracle = min(r, q)
        oracle_name = "R" if r <= q else "C"

        v3_regret = (v3 - oracle) / oracle * 100.0
        v31_regret = (v31 - oracle) / oracle * 100.0
        v31_vs_v3 = (v31 - v3) / v3 * 100.0

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:11.3f} {q:11.3f} "
            f"{v3:11.3f} {v31:11.3f} "
            f"{oracle_name:>8} "
            f"{v3_regret:11.2f}% "
            f"{v31_regret:13.2f}% "
            f"{v31_vs_v3:13.2f}%"
        )


print()
print("=" * 120)
print("V3.1 TARGET FINAL - P95")
print("=" * 120)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>12} {'Recompute':>12} "
    f"{'V3':>12} {'V3.1':>12}"
)
print("-" * 120)

for c in sorted(TARGETS):
    for prefix in sorted(TARGETS[c]):
        r = stats[(c, prefix, "restore")]["p95"]
        q = stats[(c, prefix, "recompute")]["p95"]
        v3 = stats[(c, prefix, "v3")]["p95"]
        v31 = stats[(c, prefix, "v31")]["p95"]

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:12.3f} {q:12.3f} "
            f"{v3:12.3f} {v31:12.3f}"
        )
