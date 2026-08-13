import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

LOG = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/l3_stage_probe_restore_server.log"
)

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

def fields(line):
    return dict(re.findall(r"([A-Za-z_]+)=([^\s,]+)", line))

records = {}

with LOG.open(errors="replace") as f:
    for line in f:
        if "[HiCachePrefetchQuery]" in line:
            x = fields(line)
            rid = x.get("request_id")
            if rid:
                records.setdefault(rid, {}).update({
                    "query_ms": float(x["query_ms"]),
                    "io_queue_depth": int(x["io_queue_depth"]),
                })

        elif "[HiCachePrefetchIO]" in line:
            x = fields(line)
            rid = x.get("request_id")
            if rid:
                records.setdefault(rid, {}).update({
                    "io_wait_ms": float(x["io_wait_ms"]),
                    "transfer_ms": float(x["transfer_ms"]),
                    "total_prefetch_ms": float(x["total_prefetch_ms"]),
                    "completed_tokens": int(x["completed_tokens"]),
                })

groups = defaultdict(list)

for rid, x in records.items():
    m = re.search(r"_c(\d+).*?_p(\d+)", rid)
    if not m:
        continue

    required = [
        "query_ms",
        "io_queue_depth",
        "io_wait_ms",
        "transfer_ms",
        "total_prefetch_ms",
        "completed_tokens",
    ]

    if any(k not in x for k in required):
        continue

    c = int(m.group(1))
    prefix = int(m.group(2))
    groups[(c, prefix)].append(x)

print()
print("=" * 160)
print("HICACHE L3 PREFETCH STAGE TIMING")
print("=" * 160)

print(
    f"{'C':>4} {'Prefix':>8} {'N':>5} "
    f"{'Query P50':>11} {'Query P95':>11} "
    f"{'IOQ P50':>9} {'IOQ P95':>9} "
    f"{'Wait P50':>11} {'Wait P95':>11} "
    f"{'Xfer P50':>11} {'Xfer P95':>11} "
    f"{'Total P50':>12} {'Total P95':>12} "
    f"{'Total Max':>11}"
)
print("-" * 160)

for (c, prefix) in sorted(groups):
    rows = groups[(c, prefix)]

    query = [x["query_ms"] for x in rows]
    ioq = [x["io_queue_depth"] for x in rows]
    wait = [x["io_wait_ms"] for x in rows]
    xfer = [x["transfer_ms"] for x in rows]
    total = [x["total_prefetch_ms"] for x in rows]

    print(
        f"{c:4d} {prefix:8d} {len(rows):5d} "
        f"{median(query):11.3f} {percentile(query,0.95):11.3f} "
        f"{median(ioq):9.1f} {percentile(ioq,0.95):9.1f} "
        f"{median(wait):11.3f} {percentile(wait,0.95):11.3f} "
        f"{median(xfer):11.3f} {percentile(xfer,0.95):11.3f} "
        f"{median(total):12.3f} {percentile(total,0.95):12.3f} "
        f"{max(total):11.3f}"
    )
