import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

parser = argparse.ArgumentParser()
parser.add_argument("round_id", type=int, choices=(1, 2, 3))
args = parser.parse_args()

ROUND = args.round_id

ROOT = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")
RESULT_DIR = ROOT / f"results/l3_interleaved/round{ROUND}"
LOG_DIR = ROOT / "logs"

POLICIES = ("restore", "recompute", "v3", "v31")
CONCURRENCIES = (1, 16, 32)

EXPECTED_PREFIXES = {
    1: [256, 8192, 16384],
    16: [256, 4096, 8192, 16384],
    32: [256, 4096, 8192],
}

EXPECTED_REQUESTS_PER_PREFIX = 32


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
                raise RuntimeError(
                    f"Invalid JSON: {path}:{line_no}: {e}"
                )

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


def parse_cost_actions(policy_label):
    log = LOG_DIR / f"server_final_r{ROUND}_{policy_label}.log"

    counts = defaultdict(
        lambda: {"restore": 0, "recompute": 0}
    )

    if not log.exists():
        return counts, [f"Missing log: {log}"]

    pattern = re.compile(
        r"\[HiCacheEarlyDecision\] "
        r"policy=cost_model "
        r"action=(restore|recompute).*?"
        r"rid=agent_target_cost_model_c(\d+)_"
        r"t\d+_s\d+_p(\d+)"
    )

    with log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            action = m.group(1)
            concurrency = int(m.group(2))
            prefix = int(m.group(3))

            counts[(concurrency, prefix)][action] += 1

    return counts, []


print()
print("=" * 100)
print(f"ROUND {ROUND} FILE / BASELINE VALIDATION")
print("=" * 100)

data = {}
validation_errors = []

for policy in POLICIES:
    for c in CONCURRENCIES:
        path = RESULT_DIR / f"{policy}_c{c}.jsonl"

        try:
            requests, summaries = load_jsonl(path)
        except Exception as e:
            validation_errors.append(str(e))
            continue

        expected_prefixes = EXPECTED_PREFIXES[c]

        actual_request_prefixes = sorted(requests)
        actual_summary_prefixes = sorted(summaries)

        if actual_request_prefixes != expected_prefixes:
            validation_errors.append(
                f"{path.name}: request prefixes="
                f"{actual_request_prefixes}, expected={expected_prefixes}"
            )

        if actual_summary_prefixes != expected_prefixes:
            validation_errors.append(
                f"{path.name}: summary prefixes="
                f"{actual_summary_prefixes}, expected={expected_prefixes}"
            )

        for prefix in expected_prefixes:
            rows = requests.get(prefix, [])

            if len(rows) != EXPECTED_REQUESTS_PER_PREFIX:
                validation_errors.append(
                    f"{path.name} prefix={prefix}: "
                    f"requests={len(rows)}, "
                    f"expected={EXPECTED_REQUESTS_PER_PREFIX}"
                )
                continue

            summary = summaries.get(prefix)
            if summary is None:
                continue

            if int(summary.get("requests", -1)) != EXPECTED_REQUESTS_PER_PREFIX:
                validation_errors.append(
                    f"{path.name} prefix={prefix}: "
                    f"summary requests={summary.get('requests')}"
                )

            prefetch = float(
                summary.get("storage_prefetch_delta", -1)
            )
            expected_prefetch = float(
                summary.get("expected_storage_prefetch", -1)
            )
            load_back = float(
                summary.get("load_back_delta", -1)
            )

            if policy == "restore":
                if abs(prefetch - expected_prefetch) > 1e-6:
                    validation_errors.append(
                        f"restore c={c} prefix={prefix}: "
                        f"prefetch={prefetch:.0f}, "
                        f"expected={expected_prefetch:.0f}"
                    )

                if load_back <= 0:
                    validation_errors.append(
                        f"restore c={c} prefix={prefix}: "
                        f"load_back={load_back:.0f}, expected > 0"
                    )

            elif policy == "recompute":
                if abs(prefetch) > 1e-6:
                    validation_errors.append(
                        f"recompute c={c} prefix={prefix}: "
                        f"prefetch={prefetch:.0f}, expected=0"
                    )

                if abs(load_back) > 1e-6:
                    validation_errors.append(
                        f"recompute c={c} prefix={prefix}: "
                        f"load_back={load_back:.0f}, expected=0"
                    )

            data[(policy, c, prefix)] = {
                "stats": request_stats(rows),
                "summary": summary,
            }

        print(
            f"[OK] {path.name}: "
            f"prefixes={actual_request_prefixes}"
        )

print()

if validation_errors:
    print("[FAIL] Round validation failed:")
    for error in validation_errors:
        print("  -", error)
    raise SystemExit(
        "\nDo not use this round for final comparison."
    )

print("[PASS] 12 result files are complete.")
print("[PASS] always_restore is pure L3 restore.")
print("[PASS] always_recompute is true recompute.")


print()
print("=" * 155)
print(f"ROUND {ROUND} P50 TTFT COMPARISON")
print("=" * 155)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>11} {'Recompute':>11} "
    f"{'V3':>11} {'V3.1':>11} "
    f"{'Oracle':>8} "
    f"{'V3 Regret%':>12} "
    f"{'V3.1 Regret%':>14} "
    f"{'V3.1 vs V3%':>14}"
)
print("-" * 155)

for c in CONCURRENCIES:
    for prefix in EXPECTED_PREFIXES[c]:
        r = data[("restore", c, prefix)]["stats"]["p50"]
        q = data[("recompute", c, prefix)]["stats"]["p50"]
        v3 = data[("v3", c, prefix)]["stats"]["p50"]
        v31 = data[("v31", c, prefix)]["stats"]["p50"]

        oracle = min(r, q)
        oracle_action = "R" if r <= q else "C"

        v3_regret = (v3 - oracle) / oracle * 100.0
        v31_regret = (v31 - oracle) / oracle * 100.0
        v31_vs_v3 = (v31 - v3) / v3 * 100.0

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:11.3f} {q:11.3f} "
            f"{v3:11.3f} {v31:11.3f} "
            f"{oracle_action:>8} "
            f"{v3_regret:11.2f}% "
            f"{v31_regret:13.2f}% "
            f"{v31_vs_v3:13.2f}%"
        )


print()
print("=" * 120)
print(f"ROUND {ROUND} P95 TTFT COMPARISON")
print("=" * 120)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>12} {'Recompute':>12} "
    f"{'V3':>12} {'V3.1':>12} "
    f"{'Best Static':>12}"
)
print("-" * 120)

for c in CONCURRENCIES:
    for prefix in EXPECTED_PREFIXES[c]:
        r = data[("restore", c, prefix)]["stats"]["p95"]
        q = data[("recompute", c, prefix)]["stats"]["p95"]
        v3 = data[("v3", c, prefix)]["stats"]["p95"]
        v31 = data[("v31", c, prefix)]["stats"]["p95"]

        best = "restore" if r <= q else "recompute"

        print(
            f"{c:3d} {prefix:7d} "
            f"{r:12.3f} {q:12.3f} "
            f"{v3:12.3f} {v31:12.3f} "
            f"{best:>12}"
        )


print()
print("=" * 110)
print(f"ROUND {ROUND} REQUEST THROUGHPUT")
print("=" * 110)

print(
    f"{'C':>3} {'Prefix':>7} "
    f"{'Restore':>12} {'Recompute':>12} "
    f"{'V3':>12} {'V3.1':>12}"
)
print("-" * 110)

for c in CONCURRENCIES:
    for prefix in EXPECTED_PREFIXES[c]:
        values = []

        for policy in POLICIES:
            summary = data[(policy, c, prefix)]["summary"]
            values.append(
                float(summary["request_throughput"])
            )

        print(
            f"{c:3d} {prefix:7d} "
            f"{values[0]:12.3f} "
            f"{values[1]:12.3f} "
            f"{values[2]:12.3f} "
            f"{values[3]:12.3f}"
        )


print()
print("=" * 90)
print(f"ROUND {ROUND} COST-MODEL ACTION DISTRIBUTION")
print("=" * 90)

v3_actions, v3_log_errors = parse_cost_actions("v3")
v31_actions, v31_log_errors = parse_cost_actions("v31")

for error in v3_log_errors + v31_log_errors:
    print("[WARN]", error)

print(
    f"{'Policy':>7} {'C':>3} {'Prefix':>7} "
    f"{'Restore':>8} {'Recompute':>10} "
    f"{'Total':>7} {'Restore%':>9}"
)
print("-" * 90)

action_errors = []

for label, actions in (
    ("V3", v3_actions),
    ("V3.1", v31_actions),
):
    for c in CONCURRENCIES:
        for prefix in EXPECTED_PREFIXES[c]:
            x = actions[(c, prefix)]

            restore = x["restore"]
            recompute = x["recompute"]
            total = restore + recompute

            pct = (
                100.0 * restore / total
                if total else 0.0
            )

            print(
                f"{label:>7} {c:3d} {prefix:7d} "
                f"{restore:8d} {recompute:10d} "
                f"{total:7d} {pct:8.1f}%"
            )

            if total != EXPECTED_REQUESTS_PER_PREFIX:
                action_errors.append(
                    f"{label} c={c} prefix={prefix}: "
                    f"actions={total}, expected=32"
                )

print()

if action_errors:
    print("[WARN] Action log count mismatch:")
    for error in action_errors:
        print("  -", error)
else:
    print("[PASS] V3/V3.1 action logs cover all target requests.")


v3_regrets = []
v31_regrets = []

for c in CONCURRENCIES:
    for prefix in EXPECTED_PREFIXES[c]:
        r = data[("restore", c, prefix)]["stats"]["p50"]
        q = data[("recompute", c, prefix)]["stats"]["p50"]
        v3 = data[("v3", c, prefix)]["stats"]["p50"]
        v31 = data[("v31", c, prefix)]["stats"]["p50"]

        oracle = min(r, q)

        v3_regrets.append(
            (v3 - oracle) / oracle * 100.0
        )
        v31_regrets.append(
            (v31 - oracle) / oracle * 100.0
        )

print()
print("=" * 90)
print(f"ROUND {ROUND} SUMMARY")
print("=" * 90)

print(
    "V3 median Oracle Regret   = "
    f"{median(v3_regrets):.2f}%"
)
print(
    "V3.1 median Oracle Regret = "
    f"{median(v31_regrets):.2f}%"
)

print(
    "V3 mean Oracle Regret     = "
    f"{mean(v3_regrets):.2f}%"
)
print(
    "V3.1 mean Oracle Regret   = "
    f"{mean(v31_regrets):.2f}%"
)

better = sum(
    v31 < v3
    for v3, v31 in zip(v3_regrets, v31_regrets)
)

print(
    f"V3.1 lower regret points   = "
    f"{better}/{len(v3_regrets)}"
)

print()
print(
    "Note: this is one independent round only. "
    "Do not use it as the final report result until rounds 2 and 3 finish."
)
