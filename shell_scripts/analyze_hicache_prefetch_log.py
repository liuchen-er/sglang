import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

LOG = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/l3_pressure_probe_restore_server.log"
)

MARKER = "[HiCachePrefetchPressure]"

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
matched_lines = 0
failed_lines = []

with LOG.open(errors="replace") as f:
    for line in f:
        if MARKER not in line:
            continue

        matched_lines += 1

        fields = dict(
            re.findall(r"([A-Za-z_]+)=([^\s,]+)", line)
        )

        rid = fields.get("request_id")
        if not rid:
            failed_lines.append(line.rstrip())
            continue

        rid_match = re.search(r"_c(\d+).*?_p(\d+)", rid)
        if not rid_match:
            failed_lines.append(line.rstrip())
            continue

        required = [
            "prefetch_tokens",
            "occupied",
            "capacity",
            "storage_pressure",
            "prefetch_queue_depth",
            "ongoing_prefetch",
        ]

        if any(k not in fields for k in required):
            failed_lines.append(line.rstrip())
            continue

        concurrency = int(rid_match.group(1))
        prefix = int(rid_match.group(2))

        groups[(concurrency, prefix)].append(
            {
                "rid": rid,
                "prefetch_tokens": int(fields["prefetch_tokens"]),
                "occupied": int(fields["occupied"]),
                "capacity": int(fields["capacity"]),
                "pressure": float(fields["storage_pressure"]),
                "queue_depth": int(fields["prefetch_queue_depth"]),
                "ongoing_prefetch": int(fields["ongoing_prefetch"]),
            }
        )

print(f"[Parser] marker lines={matched_lines}")
print(f"[Parser] parsed lines={sum(len(v) for v in groups.values())}")
print(f"[Parser] failed lines={len(failed_lines)}")

if failed_lines:
    print("\nFirst failed lines:")
    for line in failed_lines[:5]:
        print(line)

print()
print("=" * 150)
print("HICACHE STORAGE PRESSURE AT PREFETCH ENQUEUE")
print("=" * 150)

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
    f"{'Ongoing Max':>12} "
    f"{'Occ Max':>10} "
    f"{'Capacity':>10}"
)

print("-" * 170)

for (c, prefix) in sorted(groups):
    rows = groups[(c, prefix)]

    pressures = [x["pressure"] for x in rows]
    queues = [x["queue_depth"] for x in rows]
    ongoing = [x["ongoing_prefetch"] for x in rows]
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
        f"{median(ongoing):12.1f} "
        f"{max(ongoing):12d} "
        f"{max(occupied):10d} "
        f"{capacity:10d}"
    )

print()
print("pressure = prefetch_tokens_occupied / prefetch_capacity_limit")
