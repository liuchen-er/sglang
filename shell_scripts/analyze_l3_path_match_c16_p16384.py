import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/hicache"
)

FILES = {
    "always_restore": (
        ROOT
        / "results/l3_final/l3_final_always_restore_c16.jsonl"
    ),
    "true_recompute": (
        ROOT
        / "results/l3_final/l3_final_always_recompute_c16.jsonl"
    ),
    "v3": (
        ROOT
        / "results/l3_final/l3_final_cost_model_c16.jsonl"
    ),
    "v31": (
        ROOT
        / "results/l3_v31_target_final/l3_v31_c16.jsonl"
    ),
    "forced_early_restore": (
        ROOT
        / "results/l3_path_match/"
        "forced_early_restore_c16_p16384.jsonl"
    ),
    "forced_early_recompute": (
        ROOT
        / "results/l3_path_match/"
        "forced_early_recompute_c16_p16384.jsonl"
    ),
}


def percentile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    return (
        xs[lo] * (hi - pos)
        + xs[hi] * (pos - lo)
    )


def load(path):
    values = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("record_type") != "request":
                continue

            if int(x["prefix_len"]) != 16384:
                continue

            values.append(float(x["ttft_ms"]))

    return values


print(
    f"{'Policy':>24} {'N':>4} "
    f"{'Mean':>12} {'P50':>12} {'P95':>12}"
)
print("-" * 70)

for name, path in FILES.items():
    values = load(path)

    print(
        f"{name:>24} "
        f"{len(values):4d} "
        f"{mean(values):12.3f} "
        f"{median(values):12.3f} "
        f"{percentile(values, 0.95):12.3f}"
    )
