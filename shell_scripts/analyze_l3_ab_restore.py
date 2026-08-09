import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/results/l3_ab_restore"
)


def percentile(values, q):
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]

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
            x = json.loads(line)

            if x.get("record_type") == "request":
                values.append(float(x["ttft_ms"]))

    return values


print(
    f"{'Run':>5} {'Forced P50':>12} {'V31 P50':>12} "
    f"{'Delta':>12} {'Forced P95':>12} {'V31 P95':>12}"
)
print("-" * 80)

forced_p50s = []
v31_p50s = []

for i in range(1, 6):
    forced = load(ROOT / f"forced_early_restore_{i}.jsonl")
    v31 = load(ROOT / f"v31_restore_{i}.jsonl")

    f50 = median(forced)
    v50 = median(v31)

    forced_p50s.append(f50)
    v31_p50s.append(v50)

    print(
        f"{i:5d} "
        f"{f50:12.3f} "
        f"{v50:12.3f} "
        f"{v50-f50:12.3f} "
        f"{percentile(forced, 0.95):12.3f} "
        f"{percentile(v31, 0.95):12.3f}"
    )

print()
print(
    "Forced trial-P50 median =",
    f"{median(forced_p50s):.3f}",
)

print(
    "V3.1 trial-P50 median   =",
    f"{median(v31_p50s):.3f}",
)

print(
    "Median paired delta     =",
    f"{median(v-f for f, v in zip(forced_p50s, v31_p50s)):.3f}",
)
