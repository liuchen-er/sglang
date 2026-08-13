import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/hicache"
)

LOG = ROOT / "logs/server_l3_v31_recompute_profile.log"
RESULT_DIR = ROOT / "results/l3_v31_recompute"

RESULT_FILES = [
    RESULT_DIR / "l3_v31_recompute_c1.jsonl",
    RESULT_DIR / "l3_v31_recompute_c16.jsonl",
    RESULT_DIR / "l3_v31_recompute_c32.jsonl",
]

log_pattern = re.compile(
    r"\[HiCacheEarlyDecision\] "
    r"policy=always_recompute "
    r"action=skip_l3_prefetch "
    r"rid=(agent_target_always_recompute_c(\d+)_t(\d+)_s(\d+)_p(\d+)).*?"
    r"waiting_queue_len=(\d+).*?"
    r"running_bs=(\d+)"
)

signals = {}

with LOG.open(encoding="utf-8", errors="replace") as f:
    for line in f:
        m = log_pattern.search(line)
        if not m:
            continue

        rid = m.group(1)
        signals[rid] = {
            "concurrency": int(m.group(2)),
            "trial": int(m.group(3)),
            "session": int(m.group(4)),
            "prefix": int(m.group(5)),
            "waiting": int(m.group(6)),
            "running_bs": int(m.group(7)),
        }

rows = []

for path in RESULT_FILES:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            x = json.loads(line)

            if x.get("record_type") != "request":
                continue

            rid = x["rid"]

            if rid not in signals:
                raise RuntimeError(
                    f"Missing HiCacheEarlyDecision signal for rid={rid}"
                )

            s = signals[rid]

            rows.append(
                {
                    "rid": rid,
                    "concurrency": s["concurrency"],
                    "trial": s["trial"],
                    "prefix": s["prefix"],
                    "waiting": s["waiting"],
                    "running_bs": s["running_bs"],
                    "ttft_ms": float(x["ttft_ms"]),
                }
            )


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def linear_fit(points):
    if len(points) < 2:
        return None

    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]

    x_mean = mean(xs)
    y_mean = mean(ys)

    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return None

    beta = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    ) / denom

    intercept = y_mean - beta * x_mean

    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum(
        (y - (intercept + beta * x)) ** 2
        for x, y in zip(xs, ys)
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return intercept, beta, r2


print()
print("=" * 118)
print("TRUE RECOMPUTE GPU-QUEUE PROFILE")
print("=" * 118)

print(
    f"{'C':>3} {'Prefix':>7} {'N':>4} "
    f"{'WaitP50':>8} {'WaitP95':>8} "
    f"{'RunP50':>8} "
    f"{'TTFT P50':>10} {'TTFT P95':>10}"
)
print("-" * 118)

groups = defaultdict(list)

for row in rows:
    groups[(row["concurrency"], row["prefix"])].append(row)

for (c, prefix), subset in sorted(groups.items()):
    waits = [x["waiting"] for x in subset]
    runs = [x["running_bs"] for x in subset]
    ttfts = [x["ttft_ms"] for x in subset]

    print(
        f"{c:3d} {prefix:7d} {len(subset):4d} "
        f"{median(waits):8.1f} "
        f"{percentile(waits, 0.95):8.1f} "
        f"{median(runs):8.1f} "
        f"{median(ttfts):10.3f} "
        f"{percentile(ttfts, 0.95):10.3f}"
    )


print()
print("=" * 118)
print("PREFIX-DEPENDENT GPU QUEUE FIT")
print("=" * 118)

print(
    f"{'Prefix':>7} {'N':>4} "
    f"{'Intercept':>12} "
    f"{'Beta(ms/req)':>14} "
    f"{'R2':>8}"
)
print("-" * 118)

fit_results = {}

prefixes = sorted({x["prefix"] for x in rows})

for prefix in prefixes:
    subset = [x for x in rows if x["prefix"] == prefix]

    # Aggregate by waiting_queue_len first. This prevents a frequently
    # occurring queue length from dominating the regression.
    by_wait = defaultdict(list)

    for x in subset:
        by_wait[x["waiting"]].append(x["ttft_ms"])

    points = [
        (waiting, median(values))
        for waiting, values in sorted(by_wait.items())
    ]

    fit = linear_fit(points)

    if fit is None:
        print(
            f"{prefix:7d} {len(subset):4d} "
            f"{'NA':>12} {'NA':>14} {'NA':>8}"
        )
        continue

    intercept, beta, r2 = fit

    # Queue cost cannot be negative in the runtime model.
    beta = max(0.0, beta)

    fit_results[prefix] = {
        "storage_hit": prefix,
        "ms_per_waiting_req": beta,
        "fit_intercept_ms": intercept,
        "fit_r2": r2,
    }

    print(
        f"{prefix:7d} {len(subset):4d} "
        f"{intercept:12.3f} "
        f"{beta:14.3f} "
        f"{r2:8.4f}"
    )


output = ROOT / "profiles/l3_v31_gpu_queue_fit.json"

with output.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "source": "true_recompute_queue_profile",
            "entries": [
                fit_results[p]
                for p in sorted(fit_results)
            ],
        },
        f,
        indent=2,
    )

print()
print(f"Wrote: {output}")
