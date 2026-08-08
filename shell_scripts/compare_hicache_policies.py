import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
POLICIES = ["always_restore", "always_recompute", "token_threshold"]

def load(policy):
    d = defaultdict(list)
    with (BASE / f"policy_{policy}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            d[x["prompt_len"]].append(x)
    return d

data = {p: load(p) for p in POLICIES}
lengths = sorted(set.intersection(*(set(data[p]) for p in POLICIES)))

print(f"{'Len':>7} {'Restore':>11} {'Recompute':>11} {'Threshold':>11} {'Oracle':>11} {'Regret':>9} {'Action':>10}")
for n in lengths:
    vals = {}
    for p in POLICIES:
        vals[p] = statistics.median(x["revisit_latency_ms"] for x in data[p][n])
    oracle = min(vals["always_restore"], vals["always_recompute"])
    regret = vals["token_threshold"] - oracle
    actions = [x["action"] for x in data["token_threshold"][n]]
    action = max(set(actions), key=actions.count)
    print(f"{n:7d} {vals['always_restore']:11.3f} {vals['always_recompute']:11.3f} "
          f"{vals['token_threshold']:11.3f} {oracle:11.3f} {regret:9.3f} {action:>10}")
