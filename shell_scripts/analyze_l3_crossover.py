import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

RESULT_DIR = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/hicache/results"
)

FILES = {
    (1, "restore"): RESULT_DIR / "l3_restore_c1.jsonl",
    (1, "recompute"): RESULT_DIR / "l3_recompute_c1.jsonl",
    (16, "restore"): RESULT_DIR / "l3_restore_c16.jsonl",
    (16, "recompute"): RESULT_DIR / "l3_recompute_c16.jsonl",
    (32, "restore"): RESULT_DIR / "l3_restore_c32.jsonl",
    (32, "recompute"): RESULT_DIR / "l3_recompute_c32.jsonl",
}


def percentile(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
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
        return requests, summaries

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            prefix = int(x.get("prefix_len", -1))

            if x.get("record_type") == "request":
                requests[prefix].append(x)
            elif x.get("record_type") == "summary":
                summaries[prefix] = x

    return requests, summaries


def fmt(x):
    if x is None or math.isnan(x):
        return "-"
    return f"{x:.3f}"


data = {}

for (c, policy), path in FILES.items():
    reqs, summaries = load_file(path)

    if not path.exists():
        print(f"[WARN] missing: {path}")

    for prefix, rows in reqs.items():
        ttfts = [float(x["ttft_ms"]) for x in rows]
        summary = summaries.get(prefix, {})

        prefetch = summary.get("storage_prefetch_delta")
        expected = summary.get("expected_storage_prefetch")
        load_back = summary.get("load_back_delta")

        if policy == "restore":
            if prefetch is not None and expected is not None:
                valid = abs(float(prefetch) - float(expected)) < 1e-6
                validation = (
                    f"{float(prefetch):.0f}/{float(expected):.0f}"
                )
            else:
                valid = None
                validation = "legacy"
        else:
            if load_back is not None:
                valid = abs(float(load_back)) < 1e-6
                validation = f"loadback={float(load_back):.0f}"
            else:
                valid = None
                validation = "legacy"

        data[(c, prefix, policy)] = {
            "n": len(ttfts),
            "mean": mean(ttfts),
            "p50": median(ttfts),
            "p90": percentile(ttfts, 0.90),
            "p95": percentile(ttfts, 0.95),
            "max": max(ttfts),
            "valid": valid,
            "validation": validation,
            "prefetch": prefetch,
            "expected": expected,
            "load_back": load_back,
        }


print("\n================ L3 RAW RESULTS ================\n")

print(
    f"{'C':>4} {'Prefix':>8} {'Policy':>10} {'N':>4} "
    f"{'Mean':>10} {'P50':>10} {'P95':>10} {'Max':>10} "
    f"{'Valid':>7} {'Validation':>20}"
)
print("-" * 112)

for key in sorted(data):
    c, prefix, policy = key
    x = data[key]

    valid = (
        "PASS" if x["valid"] is True
        else "FAIL" if x["valid"] is False
        else "N/A"
    )

    print(
        f"{c:4d} {prefix:8d} {policy:>10} {x['n']:4d} "
        f"{x['mean']:10.3f} {x['p50']:10.3f} "
        f"{x['p95']:10.3f} {x['max']:10.3f} "
        f"{valid:>7} {x['validation']:>20}"
    )


print("\n================ RESTORE vs RECOMPUTE ================\n")

print(
    f"{'C':>4} {'Prefix':>8} "
    f"{'Restore P50':>14} {'Recomp P50':>14} {'R-C':>12} "
    f"{'Restore P95':>14} {'Recomp P95':>14} "
    f"{'Winner':>12} {'Status':>12}"
)
print("-" * 122)

comparisons = []

pairs = sorted(
    set((c, prefix) for c, prefix, _ in data)
)

for c, prefix in pairs:
    r = data.get((c, prefix, "restore"))
    q = data.get((c, prefix, "recompute"))

    if not r or not q:
        continue

    diff = r["p50"] - q["p50"]

    if abs(diff) < 2.0:
        winner = "tie"
    elif diff < 0:
        winner = "restore"
    else:
        winner = "recompute"

    # Restore validation is the critical one for pure-L3 experiments.
    if r["valid"] is False:
        status = "INVALID"
    elif q["valid"] is False:
        status = "INVALID"
    elif r["valid"] is None or q["valid"] is None:
        status = "LEGACY"
    else:
        status = "VALID"

    print(
        f"{c:4d} {prefix:8d} "
        f"{r['p50']:14.3f} {q['p50']:14.3f} {diff:12.3f} "
        f"{r['p95']:14.3f} {q['p95']:14.3f} "
        f"{winner:>12} {status:>12}"
    )

    comparisons.append((c, prefix, winner, diff, status))


print("\n================ CROSSOVER SUMMARY ================\n")

by_c = defaultdict(list)
for c, prefix, winner, diff, status in comparisons:
    if status in ("VALID", "LEGACY"):
        by_c[c].append((prefix, winner, diff, status))

for c in sorted(by_c):
    print(f"Concurrency = {c}")

    rows = sorted(by_c[c])

    for prefix, winner, diff, status in rows:
        print(
            f"  prefix={prefix:5d}: "
            f"{winner:9s}  R-C={diff:+8.3f} ms  [{status}]"
        )

    valid_rows = [
        (p, w, d)
        for p, w, d, status in rows
        if status in ("VALID", "LEGACY")
    ]

    transitions = []
    for a, b in zip(valid_rows, valid_rows[1:]):
        if a[1] != b[1] and "tie" not in (a[1], b[1]):
            transitions.append((a[0], b[0], a[1], b[1]))

    if transitions:
        for p0, p1, w0, w1 in transitions:
            print(
                f"  crossover candidate: "
                f"{p0} -> {p1} tokens "
                f"({w0} -> {w1})"
            )
    else:
        print("  crossover candidate: not observed")

    print()
