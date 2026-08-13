import math
import re
from collections import defaultdict
from statistics import mean, median

LOG = (
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/server_l3_v31_signal_probe.log"
)

records = defaultdict(list)

rid_re = re.compile(
    r"rid=agent_target_cost_model_c(\d+)_t\d+_s\d+_p(\d+)"
)

fields = {
    "action": re.compile(r"action=(restore|recompute)"),
    "storage_hit": re.compile(r"storage_hit_length=(\d+)"),
    "io_pending": re.compile(r"io_pending_tokens=(\d+)"),
    "waiting": re.compile(r"waiting_queue_len=(\d+)"),
    "running_bs": re.compile(r"running_bs=(\d+)"),
    "query_ms": re.compile(r"query_ms=([0-9.]+)"),
}


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


with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        if "[HiCacheEarlyDecision]" not in line:
            continue
        if "policy=cost_model" not in line:
            continue

        rid = rid_re.search(line)
        if rid is None:
            continue

        c = int(rid.group(1))
        prefix = int(rid.group(2))

        row = {}
        valid = True

        for name, pattern in fields.items():
            m = pattern.search(line)
            if m is None:
                valid = False
                break
            row[name] = m.group(1)

        if not valid:
            continue

        row["storage_hit"] = int(row["storage_hit"])
        row["io_pending"] = int(row["io_pending"])
        row["waiting"] = int(row["waiting"])
        row["running_bs"] = int(row["running_bs"])
        row["query_ms"] = float(row["query_ms"])

        records[(c, prefix)].append(row)


print()
print("=" * 125)
print("V3.1 EARLY-DECISION SIGNALS")
print("=" * 125)

print(
    f"{'C':>3} {'Prefix':>7} {'Action':>10} {'N':>4} "
    f"{'WaitMean':>9} {'WaitP50':>8} {'WaitP95':>8} {'WaitMax':>8} "
    f"{'RunP50':>8} {'IOP50':>10} {'IOP95':>10} {'QueryP50':>10}"
)
print("-" * 125)

for (c, prefix), rows in sorted(records.items()):
    for action in ("restore", "recompute"):
        subset = [x for x in rows if x["action"] == action]
        if not subset:
            continue

        waits = [x["waiting"] for x in subset]
        runs = [x["running_bs"] for x in subset]
        ios = [x["io_pending"] for x in subset]
        queries = [x["query_ms"] for x in subset]

        print(
            f"{c:3d} {prefix:7d} {action:>10} {len(subset):4d} "
            f"{mean(waits):9.2f} {median(waits):8.1f} "
            f"{percentile(waits, 0.95):8.1f} {max(waits):8d} "
            f"{median(runs):8.1f} "
            f"{median(ios):10.0f} {percentile(ios, 0.95):10.0f} "
            f"{median(queries):10.3f}"
        )

print()
print("=" * 88)
print("ACTION DISTRIBUTION BY WAITING-QUEUE BUCKET")
print("=" * 88)

buckets = [
    (0, 0, "0"),
    (1, 4, "1-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 24, "17-24"),
    (25, 10**9, "25+"),
]

print(
    f"{'C':>3} {'Prefix':>7} {'Bucket':>8} "
    f"{'Restore':>8} {'Recompute':>10} {'Total':>7}"
)
print("-" * 88)

for (c, prefix), rows in sorted(records.items()):
    for lo, hi, label in buckets:
        subset = [
            x for x in rows
            if lo <= x["waiting"] <= hi
        ]

        if not subset:
            continue

        restore = sum(x["action"] == "restore" for x in subset)
        recompute = sum(x["action"] == "recompute" for x in subset)

        print(
            f"{c:3d} {prefix:7d} {label:>8} "
            f"{restore:8d} {recompute:10d} {len(subset):7d}"
        )
