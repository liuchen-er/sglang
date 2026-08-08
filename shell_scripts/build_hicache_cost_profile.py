import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache")
PAGE_SIZE = 64
PREFILL_BUCKET = 2048

PATTERN = re.compile(
    r"\[HiCacheDecision\] policy=(\S+) action=(\S+) rid=(\S+) prompt_len=(\d+) host_hit_length=(\d+).*?"
    r"prefill_batch_tokens=(\d+) running_bs=(\d+) qload=(\d+)"
)

def decisions(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            policy, action, rid, prompt_len, host_hit, prefill, running_bs, qload = m.groups()
            out[rid] = {
                "policy": policy, "action": action, "prompt_len": int(prompt_len),
                "host_hit": int(host_hit), "prefill": int(prefill),
                "running_bs": int(running_bs), "qload": int(qload),
            }
    return out

def results(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out[x["rid"]] = x
    return out

def bucket_up(x, size):
    return ((x + size - 1) // size) * size

restore_dec = decisions(BASE / "logs/server_load_restore.log")
recompute_dec = decisions(BASE / "logs/server_load_recompute.log")
restore_res = results(BASE / "results/load_always_restore.jsonl")
recompute_res = results(BASE / "results/load_always_recompute.jsonl")

groups = {"restore": defaultdict(list), "recompute": defaultdict(list)}

for action, decs, res in [
    ("restore", restore_dec, restore_res),
    ("recompute", recompute_dec, recompute_res),
]:
    for rid, d in decs.items():
        if rid not in res:
            continue
        key = (d["host_hit"], bucket_up(d["prefill"], PREFILL_BUCKET), d["qload"])
        groups[action][key].append((res[rid]["latency_ms"], d["running_bs"]))

profile = {
    "version": 1, "page_size": PAGE_SIZE, "prefill_bucket_size": PREFILL_BUCKET,
    "entries": {"restore": [], "recompute": []},
}

for action in ("restore", "recompute"):
    for (host_hit, prefill_bucket, qload), values in sorted(groups[action].items()):
        latencies = [x[0] for x in values]
        running_bs = [x[1] for x in values]
        profile["entries"][action].append({
            "host_hit": host_hit, "prefill_bucket": prefill_bucket, "qload": qload,
            "latency_ms": statistics.median(latencies),
            "running_bs_median": statistics.median(running_bs), "samples": len(values),
        })

out = BASE / "profiles/hicache_cost_profile.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
print(f"Profile saved to {out}")
print(f"restore entries={len(profile['entries']['restore'])}")
print(f"recompute entries={len(profile['entries']['recompute'])}")
