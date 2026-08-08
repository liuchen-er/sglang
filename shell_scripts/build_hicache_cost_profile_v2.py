import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")
BS_BUCKETS = [0, 4, 16, 32]
EXPECTED_HITS = [512, 1024, 2048, 4096, 8192, 16384]

RESULT_FILES = {
    "restore": BASE / "results/paired_always_restore.jsonl",
    "recompute": BASE / "results/paired_always_recompute.jsonl",
}

LOG_FILES = {
    "restore": BASE / "logs/server_paired_always_restore.log",
    "recompute": BASE / "logs/server_paired_always_recompute.log",
}

DECISION_RE = re.compile(
    r"\[HiCacheDecision\].*?rid=(\S+).*?"
    r"host_hit_length=(\d+).*?running_bs=(\d+)"
)

def load_results(path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get("record_type") == "request":
                rows[x["rid"]] = x
    return rows

def load_decisions(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = DECISION_RE.search(line)
            if m:
                rid, host_hit, running_bs = m.groups()
                out[rid] = {
                    "host_hit_length": int(host_hit),
                    "actual_running_bs": int(running_bs),
                }
    return out

groups = {
    "restore": defaultdict(list),
    "recompute": defaultdict(list),
}

for action in ("restore", "recompute"):
    results = load_results(RESULT_FILES[action])
    decisions = load_decisions(LOG_FILES[action])

    missing_log = 0
    for rid, row in results.items():
        d = decisions.get(rid)
        if d is None:
            missing_log += 1
            continue

        expected_hit = row["prefix_len"]
        actual_hit = d["host_hit_length"]
        if actual_hit != expected_hit:
            print(
                f"[WARN] rid={rid}: expected host_hit={expected_hit}, "
                f"actual={actual_hit}"
            )

        key = (expected_hit, row["requested_running_bs"])
        groups[action][key].append({
            "latency_ms": row["ttft_ms"],
            "actual_running_bs": d["actual_running_bs"],
        })

    if missing_log:
        raise RuntimeError(
            f"{action}: {missing_log} target requests have no HiCacheDecision log."
        )

missing = []
for action in ("restore", "recompute"):
    for hit in EXPECTED_HITS:
        for bs in BS_BUCKETS:
            if not groups[action][(hit, bs)]:
                missing.append((action, hit, bs))

if missing:
    print("Missing profile cells:")
    for x in missing:
        print("  ", x)
    raise RuntimeError("Profile matrix is incomplete.")

profile = {
    "version": 2,
    "dimensions": ["host_hit", "running_bs"],
    "host_hit_points": EXPECTED_HITS,
    "running_bs_buckets": BS_BUCKETS,
    "entries": {
        "restore": [],
        "recompute": [],
    },
}

print(f"{'Action':>10} {'HostHit':>8} {'ReqBS':>6} {'ActualBS':>9} {'Median(ms)':>12} {'P95(ms)':>10} {'N':>5}")
print("-" * 72)

for action in ("restore", "recompute"):
    for hit in EXPECTED_HITS:
        for bs in BS_BUCKETS:
            values = groups[action][(hit, bs)]
            latencies = [x["latency_ms"] for x in values]
            actual_bs = [x["actual_running_bs"] for x in values]

            median_latency = statistics.median(latencies)
            actual_bs_median = statistics.median(actual_bs)
            sorted_lat = sorted(latencies)
            p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))
            p95_latency = sorted_lat[p95_idx]

            profile["entries"][action].append({
                "host_hit": hit,
                "running_bs": bs,
                "latency_ms": median_latency,
                "p95_latency_ms": p95_latency,
                "actual_running_bs_median": actual_bs_median,
                "samples": len(values),
            })

            print(
                f"{action:>10} {hit:8d} {bs:6d} {actual_bs_median:9.1f} "
                f"{median_latency:12.3f} {p95_latency:10.3f} {len(values):5d}"
            )

out = BASE / "profiles/hicache_cost_profile_v2.json"
out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
print(f"\nSaved: {out}")
