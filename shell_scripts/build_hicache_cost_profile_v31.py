import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")

V3_PROFILE = ROOT / "profiles/hicache_cost_profile_v3.json"
V31_PROFILE = ROOT / "profiles/hicache_cost_profile_v31.json"

LOG = ROOT / "logs/server_l3_v31_recompute_profile.log"

RESULT_FILES = [
    ROOT / "results/l3_v31_recompute/l3_v31_recompute_c1.jsonl",
    ROOT / "results/l3_v31_recompute/l3_v31_recompute_c16.jsonl",
    ROOT / "results/l3_v31_recompute/l3_v31_recompute_c32.jsonl",
]

MIN_GPU_QUEUE_STORAGE_HIT = 4096

with V3_PROFILE.open(encoding="utf-8") as f:
    profile = json.load(f)

if profile.get("version") != 3:
    raise ValueError(
        f"Expected V3 profile, got version={profile.get('version')}"
    )

recompute_base = {
    int(x["storage_hit"]): float(x["latency_ms"])
    for x in profile["l3"]["recompute_base"]
}

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

        signals[m.group(1)] = {
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
                    f"Missing early-decision signal for rid={rid}"
                )

            s = signals[rid]

            rows.append(
                {
                    "prefix": s["prefix"],
                    "waiting": s["waiting"],
                    "ttft_ms": float(x["ttft_ms"]),
                }
            )


def anchored_fit(points, base_ms):
    if not points:
        return None

    denom = sum(q * q for q, _ in points)

    if denom <= 0:
        return None

    beta = sum(
        q * (ttft - base_ms)
        for q, ttft in points
    ) / denom

    beta = max(0.0, beta)

    predictions = [
        base_ms + beta * q
        for q, _ in points
    ]
    actual = [
        ttft
        for _, ttft in points
    ]

    mse = sum(
        (y - yhat) ** 2
        for y, yhat in zip(actual, predictions)
    ) / len(actual)

    rmse = math.sqrt(mse)

    mean_y = sum(actual) / len(actual)
    ss_tot = sum((y - mean_y) ** 2 for y in actual)
    ss_res = sum(
        (y - yhat) ** 2
        for y, yhat in zip(actual, predictions)
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return beta, rmse, r2


entries = []

print()
print("=" * 100)
print("ANCHORED GPU QUEUE FIT")
print("=" * 100)

print(
    f"{'Prefix':>7} {'Base':>10} "
    f"{'Beta(ms/req)':>14} {'RMSE':>10} {'R2':>10}"
)
print("-" * 100)

for prefix in sorted({x["prefix"] for x in rows}):
    if prefix not in recompute_base:
        raise RuntimeError(
            f"No recompute_base for prefix={prefix}"
        )

    base_ms = recompute_base[prefix]

    by_wait = defaultdict(list)

    for row in rows:
        if row["prefix"] != prefix:
            continue

        by_wait[row["waiting"]].append(
            row["ttft_ms"]
        )

    # Use one median point per queue depth to avoid high-frequency
    # queue values dominating the regression.
    points = [
        (waiting, median(values))
        for waiting, values in sorted(by_wait.items())
    ]

    fit = anchored_fit(points, base_ms)

    if fit is None:
        raise RuntimeError(
            f"Unable to fit prefix={prefix}"
        )

    beta, rmse, r2 = fit

    print(
        f"{prefix:7d} {base_ms:10.3f} "
        f"{beta:14.3f} {rmse:10.3f} {r2:10.4f}"
    )

    entries.append(
        {
            "storage_hit": prefix,
            "ms_per_waiting_req": beta,
            "fit_rmse_ms": rmse,
            "fit_r2": r2,
        }
    )

v31 = copy.deepcopy(profile)

v31["l3"]["model_revision"] = "v3.1"

v31["l3"]["gpu_queue"] = {
    "min_storage_hit": MIN_GPU_QUEUE_STORAGE_HIT,
    "entries": entries,
}

with V31_PROFILE.open("w", encoding="utf-8") as f:
    json.dump(v31, f, indent=2)

print()
print(f"Wrote V3.1 profile: {V31_PROFILE}")
print(
    f"GPU queue penalty enabled only for "
    f"storage_hit >= {MIN_GPU_QUEUE_STORAGE_HIT}"
)
