import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

LOG = Path(
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/l3_pressure_probe_restore_server.log"
)

def find_int(line, names):
    for name in names:
        m = re.search(rf"\b{re.escape(name)}=(-?\d+)", line)
        if m:
            return int(m.group(1))
    return None

def find_str(line, names):
    for name in names:
        m = re.search(rf"\b{re.escape(name)}=([A-Za-z0-9_]+)", line)
        if m:
            return m.group(1)
    return None

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
        if "[HiCacheDecision]" not in line:
            continue

        prefix = find_int(
            line,
            ["host_hit_length", "prefix_len", "prefix"],
        )

        running_bs = find_int(
            line,
            ["running_bs"],
        )

        action = find_str(
            line,
            ["action"],
        )

        if prefix is None or running_bs is None:
            continue

        groups[prefix].append(
            {
                "running_bs": running_bs,
                "action": action or "unknown",
            }
        )

print()
print("=" * 100)
print("HICACHE DECISION RUNTIME LOAD")
print("=" * 100)

print(
    f"{'Prefix':>8} "
    f"{'N':>5} "
    f"{'BS Mean':>10} "
    f"{'BS P50':>10} "
    f"{'BS P95':>10} "
    f"{'BS Max':>10} "
    f"{'Actions':>30}"
)

print("-" * 95)

for prefix in sorted(groups):
    rows = groups[prefix]
    bs = [x["running_bs"] for x in rows]
    actions = Counter(x["action"] for x in rows)

    action_text = ",".join(
        f"{k}:{v}"
        for k, v in sorted(actions.items())
    )

    print(
        f"{prefix:8d} "
        f"{len(rows):5d} "
        f"{mean(bs):10.2f} "
        f"{median(bs):10.2f} "
        f"{percentile(bs, 0.95):10.2f} "
        f"{max(bs):10d} "
        f"{action_text:>30}"
    )
