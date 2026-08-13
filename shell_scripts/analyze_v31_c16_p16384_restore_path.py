import math
import re
from pathlib import Path
from statistics import median

ROOT = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")

CASES = {
    "always_restore": {
        "log": ROOT / "logs/server_l3_final_always_restore.log",
        "rid": re.compile(
            r"agent_target_always_restore_c16_t\d+_s\d+_p16384"
        ),
    },
    "v31": {
        "log": ROOT / "logs/server_l3_v31_target_final.log",
        "rid": re.compile(
            r"agent_target_cost_model_c16_t\d+_s\d+_p16384"
        ),
    },
}


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


def stat(values):
    if not values:
        return ("NA", "NA", "NA")
    return (
        f"{median(values):.3f}",
        f"{percentile(values, 0.95):.3f}",
        f"{max(values):.3f}",
    )


def parse_case(name, cfg):
    path = cfg["log"]
    rid_pattern = cfg["rid"]

    if not path.exists():
        raise FileNotFoundError(path)

    query_ms = []
    io_queue_depth = []
    io_pending = []

    io_wait_ms = []
    transfer_ms = []
    total_prefetch_ms = []

    occupied = []
    pressure = []
    prefetch_queue_depth = []
    ongoing_prefetch = []

    query_source = {
        "worker": 0,
        "precomputed": 0,
    }

    early_restore = 0
    early_recompute = 0

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if rid_pattern.search(line) is None:
                continue

            if "[HiCacheEarlyDecision]" in line:
                if "action=restore" in line:
                    early_restore += 1
                elif "action=recompute" in line:
                    early_recompute += 1

            if "[HiCachePrefetchQuery]" in line:
                m = re.search(r"query_source=(\w+)", line)
                if m and m.group(1) in query_source:
                    query_source[m.group(1)] += 1

                m = re.search(r"query_ms=([0-9.]+)", line)
                if m:
                    query_ms.append(float(m.group(1)))

                m = re.search(r"io_queue_depth=(\d+)", line)
                if m:
                    io_queue_depth.append(int(m.group(1)))

                m = re.search(r"io_pending_tokens=(\d+)", line)
                if m:
                    io_pending.append(int(m.group(1)))

            if "[HiCachePrefetchIO]" in line:
                m = re.search(r"io_wait_ms=([0-9.]+)", line)
                if m:
                    io_wait_ms.append(float(m.group(1)))

                m = re.search(r"transfer_ms=([0-9.]+)", line)
                if m:
                    transfer_ms.append(float(m.group(1)))

                m = re.search(r"total_prefetch_ms=([0-9.]+)", line)
                if m:
                    total_prefetch_ms.append(float(m.group(1)))

            if "[HiCachePrefetchPressure]" in line:
                m = re.search(r"occupied=(\d+)", line)
                if m:
                    occupied.append(int(m.group(1)))

                m = re.search(r"storage_pressure=([0-9.]+)", line)
                if m:
                    pressure.append(float(m.group(1)))

                m = re.search(r"prefetch_queue_depth=(\d+)", line)
                if m:
                    prefetch_queue_depth.append(int(m.group(1)))

                m = re.search(r"ongoing_prefetch=(\d+)", line)
                if m:
                    ongoing_prefetch.append(int(m.group(1)))

    return {
        "name": name,
        "query_ms": query_ms,
        "io_queue_depth": io_queue_depth,
        "io_pending": io_pending,
        "io_wait_ms": io_wait_ms,
        "transfer_ms": transfer_ms,
        "total_prefetch_ms": total_prefetch_ms,
        "occupied": occupied,
        "pressure": pressure,
        "prefetch_queue_depth": prefetch_queue_depth,
        "ongoing_prefetch": ongoing_prefetch,
        "query_source": query_source,
        "early_restore": early_restore,
        "early_recompute": early_recompute,
    }


results = [
    parse_case(name, cfg)
    for name, cfg in CASES.items()
]

print()
print("=" * 145)
print("c16 / prefix=16384 RESTORE PATH COMPARISON")
print("=" * 145)

print(
    f"{'Case':>16} {'NIO':>5} "
    f"{'QueryP50':>10} {'QueryP95':>10} "
    f"{'IOQ P50':>9} {'IOQ P95':>9} "
    f"{'PendingP50':>12} {'PendingP95':>12} "
    f"{'WaitP50':>11} {'WaitP95':>11} "
    f"{'XferP50':>11} {'XferP95':>11} "
    f"{'TotalP50':>11} {'TotalP95':>11}"
)
print("-" * 145)

for x in results:
    q50, q95, _ = stat(x["query_ms"])
    iq50, iq95, _ = stat(x["io_queue_depth"])
    ip50, ip95, _ = stat(x["io_pending"])
    w50, w95, _ = stat(x["io_wait_ms"])
    t50, t95, _ = stat(x["transfer_ms"])
    p50, p95, _ = stat(x["total_prefetch_ms"])

    print(
        f"{x['name']:>16} "
        f"{len(x['io_wait_ms']):5d} "
        f"{q50:>10} {q95:>10} "
        f"{iq50:>9} {iq95:>9} "
        f"{ip50:>12} {ip95:>12} "
        f"{w50:>11} {w95:>11} "
        f"{t50:>11} {t95:>11} "
        f"{p50:>11} {p95:>11}"
    )

print()
print("=" * 110)
print("ADMISSION-SIDE PREFETCH STATE")
print("=" * 110)

print(
    f"{'Case':>16} "
    f"{'OccupiedP50':>13} {'OccupiedP95':>13} {'OccupiedMax':>13} "
    f"{'PrefetchQP50':>12} {'PrefetchQP95':>12} "
    f"{'OngoingP50':>11} {'OngoingP95':>11}"
)
print("-" * 110)

for x in results:
    o50, o95, omax = stat(x["occupied"])
    q50, q95, _ = stat(x["prefetch_queue_depth"])
    g50, g95, _ = stat(x["ongoing_prefetch"])

    print(
        f"{x['name']:>16} "
        f"{o50:>13} {o95:>13} {omax:>13} "
        f"{q50:>12} {q95:>12} "
        f"{g50:>11} {g95:>11}"
    )

print()
print("=" * 80)
print("QUERY SOURCE / EARLY ACTION")
print("=" * 80)

for x in results:
    print(
        f"{x['name']}: "
        f"worker={x['query_source']['worker']} "
        f"precomputed={x['query_source']['precomputed']} "
        f"early_restore={x['early_restore']} "
        f"early_recompute={x['early_recompute']}"
    )
