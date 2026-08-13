import json
import os
from pathlib import Path
import numpy as np

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
POLICIES = [x.strip() for x in os.getenv("POLICIES", "always_restore,token_threshold,cost_model").split(",") if x.strip()]
CONCURRENCIES = [int(x) for x in os.getenv("CONCURRENCIES", "4,16,32").split(",")]

def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")

def load(policy, concurrency):
    path = BASE / f"agent_{policy}_c{concurrency}.jsonl"
    if not path.exists():
        return None
    rows, summaries = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if x["record_type"] == "request":
                rows.append(x)
            elif x["record_type"] == "summary":
                summaries.append(x)
    return rows, summaries

def fmt_change(new, base):
    if not np.isfinite(new) or not np.isfinite(base) or base == 0:
        return "N/A"
    return f"{(base - new) / base * 100:+.2f}%"

data = {}
for c in CONCURRENCIES:
    for p in POLICIES:
        loaded = load(p, c)
        if loaded is not None:
            data[(c, p)] = loaded

for c in CONCURRENCIES:
    if (c, "always_restore") not in data:
        print(f"[WARN] concurrency={c}: missing always_restore baseline, skip.")
        continue

    base_rows, base_summaries = data[(c, "always_restore")]
    base_ttft = [x["ttft_ms"] for x in base_rows]
    base_p50 = pct(base_ttft, 50)
    base_p95 = pct(base_ttft, 95)
    base_p99 = pct(base_ttft, 99)

    print("\n" + "=" * 158)
    print(f"AGENT L2-HIT RESULTS — MAX_CONCURRENCY={c}")
    print("=" * 158)
    print(
        f"{'Policy':>18} {'Req':>6} {'P50 TTFT':>10} {'P95 TTFT':>10} {'P99 TTFT':>10} "
        f"{'P95 Δ':>10} {'P99 Δ':>10} {'P95 TPOT':>10} {'P95 ITL':>10} {'P99 ITL':>10} "
        f"{'Out tok/s':>11} {'LoadBack/Req':>13} {'Evict/Req':>11}"
    )

    for p in POLICIES:
        if (c, p) not in data:
            continue

        rows, summaries = data[(c, p)]
        ttft = [x["ttft_ms"] for x in rows]
        tpot = [x["tpot_ms"] for x in rows if x.get("output_tokens", 0) > 1]
        itl = [v for x in rows for v in x.get("itls_ms", [])]

        p50 = pct(ttft, 50)
        p95 = pct(ttft, 95)
        p99 = pct(ttft, 99)

        throughput = pct([x["output_throughput"] for x in summaries], 50)
        total_req = sum(x.get("requests", 0) for x in summaries)
        load_per_req = sum(x.get("load_back_delta", 0) for x in summaries) / max(total_req, 1)
        evict_per_req = sum(x.get("measurement_evicted_delta", 0) for x in summaries) / max(total_req, 1)

        print(
            f"{p:>18} {len(rows):6d} {p50:10.3f} {p95:10.3f} {p99:10.3f} "
            f"{fmt_change(p95, base_p95):>10} {fmt_change(p99, base_p99):>10} "
            f"{pct(tpot,95):10.3f} {pct(itl,95):10.3f} {pct(itl,99):10.3f} "
            f"{throughput:11.1f} {load_per_req:13.1f} {evict_per_req:11.1f}"
        )

    print("\nTTFT BY PREFIX LENGTH")
    print("-" * 138)
    print(f"{'Prefix':>8}", end="")
    available = [p for p in POLICIES if (c, p) in data]
    for p in available:
        print(f" {p + ' P50':>18} {p + ' P95':>18} {p + ' ΔP95':>13}", end="")
    print()

    prefixes = sorted({
        x["prefix_len"]
        for p in available
        for x in data[(c, p)][0]
        if "prefix_len" in x
    })

    for n in prefixes:
        base_vals = [x["ttft_ms"] for x in base_rows if x.get("prefix_len") == n]
        base_len_p95 = pct(base_vals, 95)

        print(f"{n:8d}", end="")
        for p in available:
            rows, _ = data[(c, p)]
            vals = [x["ttft_ms"] for x in rows if x.get("prefix_len") == n]
            p50 = pct(vals, 50)
            p95 = pct(vals, 95)
            print(f" {p50:18.3f} {p95:18.3f} {fmt_change(p95, base_len_p95):>13}", end="")
        print()

print("\n" + "=" * 122)
print("CROSS-CONCURRENCY P95 TTFT SUMMARY")
print("=" * 122)
print(f"{'Concurrency':>12}", end="")
for p in POLICIES:
    print(f" {p + ' P95':>20} {p + ' Δ':>12}", end="")
print()

for c in CONCURRENCIES:
    if (c, "always_restore") not in data:
        continue
    base_rows, _ = data[(c, "always_restore")]
    base_p95 = pct([x["ttft_ms"] for x in base_rows], 95)

    print(f"{c:12d}", end="")
    for p in POLICIES:
        if (c, p) not in data:
            print(f" {'N/A':>20} {'N/A':>12}", end="")
            continue
        rows, _ = data[(c, p)]
        p95 = pct([x["ttft_ms"] for x in rows], 95)
        print(f" {p95:20.3f} {fmt_change(p95, base_p95):>12}", end="")
    print()
