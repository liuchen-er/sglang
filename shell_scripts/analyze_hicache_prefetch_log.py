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

# 只有出现在 HiCachePrefetchQuery 中的请求，才是真正命中 L3、
# 进入正式 Restore I/O 路径的请求。
successful_l3_rids = set()

with LOG.open(errors="replace") as f:
    for line in f:
        if "[HiCachePrefetchQuery]" not in line:
            continue
        x = fields(line)
        rid = x.get("request_id")
        if rid:
            successful_l3_rids.add(rid)

pressure_records = {}

with LOG.open(errors="replace") as f:
    for line in f:
        if "[HiCachePrefetchPressure]" not in line:
            continue

        x = fields(line)
        rid = x.get("request_id")

        if not rid or rid not in successful_l3_rids:
            continue

        required = [
            "prefetch_tokens",
            "occupied",
            "capacity",
            "storage_pressure",
            "prefetch_queue_depth",
            "ongoing_prefetch",
        ]

        if any(k not in x for k in required):
            continue

        pressure_records[rid] = {
            "prefetch_tokens": int(x["prefetch_tokens"]),
            "occupied": int(x["occupied"]),
            "capacity": int(x["capacity"]),
            "pressure": float(x["storage_pressure"]),
            "queue_depth": int(x["prefetch_queue_depth"]),
            "ongoing": int(x["ongoing_prefetch"]),
        }

groups = defaultdict(list)

for rid, x in pressure_records.items():
    m = re.search(r"_c(\d+).*?_p(\d+)", rid)
    if not m:
        continue

    c = int(m.group(1))
    prefix = int(m.group(2))
    groups[(c, prefix)].append(x)

print(f"[Parser] successful L3 rids={len(successful_l3_rids)}")
print(f"[Parser] matched pressure rids={len(pressure_records)}")

print()
print("=" * 160)
print("HICACHE STORAGE PRESSURE - SUCCESSFUL L3 RESTORE ONLY")
print("=" * 160)

print(
    f"{'C':>4} "
    f"{'Prefix':>8} "
    f"{'N':>5} "
    f"{'Pressure Mean':>14} "
    f"{'Pressure P50':>14} "
    f"{'Pressure P95':>14} "
    f"{'Pressure Max':>14} "
    f"{'Queue P50':>10} "
    f"{'Queue P95':>10} "
    f"{'Queue Max':>10} "
    f"{'Ongoing P50':>12} "
    f"{'Ongoing P95':>12} "
    f"{'Ongoing Max':>12} "
    f"{'Occ Max':>10} "
    f"{'Capacity':>10}"
)

print("-" * 180)

for (c, prefix) in sorted(groups):
    rows = groups[(c, prefix)]

    pressures = [x["pressure"] for x in rows]
    queues = [x["queue_depth"] for x in rows]
    ongoing = [x["ongoing"] for x in rows]
    occupied = [x["occupied"] for x in rows]
    capacity = max(x["capacity"] for x in rows)

    print(
        f"{c:4d} "
        f"{prefix:8d} "
        f"{len(rows):5d} "
        f"{mean(pressures):14.4f} "
        f"{median(pressures):14.4f} "
        f"{percentile(pressures,0.95):14.4f} "
        f"{max(pressures):14.4f} "
        f"{median(queues):10.1f} "
        f"{percentile(queues,0.95):10.1f} "
        f"{max(queues):10d} "
        f"{median(ongoing):12.1f} "
        f"{percentile(ongoing,0.95):12.1f} "
        f"{max(ongoing):12d} "
        f"{max(occupied):10d} "
        f"{capacity:10d}"
    )

print()
print("Only requests confirmed by [HiCachePrefetchQuery] are included.")
print("pressure = prefetch_tokens_occupied / prefetch_capacity_limit")
