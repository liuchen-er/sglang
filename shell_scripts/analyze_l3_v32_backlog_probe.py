import re

LOG = (
    "/root/projects/sglang-qwen2-adaptive-prefill/"
    "hicache/logs/server_l3_v32_backlog_probe.log"
)

pattern = re.compile(
    r"\[HiCacheEarlyDecision\] policy=cost_model "
    r"action=(restore|recompute) "
    r"rid=agent_target_cost_model_c16_t0_s(\d+)_p16384 "
    r"storage_hit_length=(\d+) "
    r"io_pending_tokens=(\d+) "
    r"admission_pending_tokens=(\d+) "
    r"waiting_queue_len=(\d+) "
    r"running_bs=(\d+)"
)

rows = []

with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        rows.append(
            {
                "sid": int(m.group(2)),
                "action": m.group(1),
                "storage_hit": int(m.group(3)),
                "io_pending": int(m.group(4)),
                "admission_pending": int(m.group(5)),
                "waiting": int(m.group(6)),
                "running_bs": int(m.group(7)),
            }
        )

rows.sort(key=lambda x: x["sid"])

print(
    f"{'SID':>3} {'Action':>10} "
    f"{'IOPending':>10} {'Admission':>10} "
    f"{'Waiting':>8} {'Running':>8}"
)
print("-" * 65)

for x in rows:
    print(
        f"{x['sid']:3d} "
        f"{x['action']:>10} "
        f"{x['io_pending']:10d} "
        f"{x['admission_pending']:10d} "
        f"{x['waiting']:8d} "
        f"{x['running_bs']:8d}"
    )
