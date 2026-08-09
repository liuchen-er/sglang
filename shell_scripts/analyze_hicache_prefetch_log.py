import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

LOG = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/l3_pressure_probe_restore_server.log"
)

PATTERN = re.compile(
    r"\[HiCachePrefetchPressure\].*?"
    r"request_id=(\S+)\s+"
    r"request_tokens=(\d+)\s+"
    r"occupied_at_enqueue=(\d+)\s+"
    r"capacity=(\d+)\s+"
    r"storage_pressure=([0-9.]+)\s+"
    r"queue_depth_at_enqueue=(\d+)"
)

RID_PATTERN = re.compile(
    r"_c(\d+).*?_p(\d+)"
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

if not LOG.exists():
    raise FileNotFoundError(LOG)

groups = defaultdict(list)

with LOG.open(errors="replace") as f:
    for line in f:
        m = PATTERN.search(line)
        if not m:
            continue

        rid = m.group(1)
        request_tokens = int(m.group(2))
        occupied = int(m.group(3))
        capacity = int(m.group(4))
        pressure = float(m.group(5))
        queue_depth = int(m.group(6))
        ongoing_prefetch = int(m.group(7))

        rm = RID_PATTERN.search(rid)
        if not rm:
            continue

        concurrency = int(rm.group(1))
        prefix = int(rm.group(2))

        groups[(concurrency, prefix)].append(
            {
                "rid": rid,
                "request_tokens": request_tokens,
                "occupied": occupied,
                "capacity": capacity,
                "pressure": pressure,
                "queue_depth": queue_depth,
            }
        )

print()
print("=" * 120)
print("HICACHE STORAGE PRESSURE AT PREFETCH ENQUEUE")
print("=" * 120)

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
    f"{'Occ Max':>10} "
    f"{'Capacity':>10}"
)

print("-" * 145)

for (c, prefix) in sorted(groups):
    rows = groups[(c, prefix)]

    pressures = [x["pressure"] for x in rows]
    queues = [x["queue_depth"] for x in rows]
    occupied = [x["occupied"] for x in rows]
    capacity = max(x["capacity"] for x in rows)

    print(
        f"{c:4d} "
        f"{prefix:8d} "
        f"{len(rows):5d} "
        f"{mean(pressures):14.4f} "
        f"{median(pressures):14.4f} "
        f"{percentile(pressures, 0.95):14.4f} "
        f"{max(pressures):14.4f} "
        f"{median(queues):10.1f} "
        f"{percentile(queues, 0.95):10.1f} "
        f"{max(queues):10d} "
        f"{max(occupied):10d} "
        f"{capacity:10d}"
    )

print()
print("pressure = prefetch_tokens_occupied / prefetch_capacity_limit")
