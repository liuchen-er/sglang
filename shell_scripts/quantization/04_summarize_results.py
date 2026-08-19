#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path


if len(sys.argv) >= 2:
    precision = sys.argv[1]
else:
    precision = "bf16"


RESULT_ROOT = Path("/root/autodl-tmp/qwen25_quant/results")
RESULT_DIR = RESULT_ROOT / precision


pattern = re.compile(
    rf"{re.escape(precision)}_"
    r"in(\d+)_out(\d+)_c(\d+)_n(\d+)_"
    r"(\d{8}_\d{6})\.jsonl"
)


rows = []

for path in RESULT_DIR.glob("*.jsonl"):
    match = pattern.fullmatch(path.name)

    if not match:
        continue

    input_len = int(match.group(1))
    output_len = int(match.group(2))
    concurrency = int(match.group(3))
    num_prompts = int(match.group(4))
    timestamp = match.group(5)

    with path.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        continue

    # 当前一个文件对应一轮 benchmark，取最后一行
    data = json.loads(lines[-1])

    rows.append(
        {
            "input": input_len,
            "output": output_len,
            "concurrency": concurrency,
            "prompts": num_prompts,
            "success": data.get("completed"),
            "duration": data.get("duration"),
            "ttft": data.get("mean_ttft_ms"),
            "tpot": data.get("mean_tpot_ms"),
            "itl": data.get("mean_itl_ms"),
            "input_tps": data.get("input_throughput"),
            "output_tps": data.get("output_throughput"),
            "total_tps": data.get("total_throughput"),
            "timestamp": timestamp,
            "file": path.name,
        }
    )


rows.sort(
    key=lambda x: (
        -x["input"],
        x["output"],
        x["concurrency"],
        x["timestamp"],
    )
)


print()
print(f"{precision.upper()} Benchmark Summary")
print("=" * 118)

header = (
    f"{'Input':>7} "
    f"{'Output':>7} "
    f"{'Conc':>6} "
    f"{'Succ':>6} "
    f"{'Dur(s)':>8} "
    f"{'TTFT(ms)':>10} "
    f"{'TPOT(ms)':>10} "
    f"{'ITL(ms)':>10} "
    f"{'Input tok/s':>13} "
    f"{'Output tok/s':>14} "
    f"{'Total tok/s':>13}"
)

print(header)
print("-" * 118)

for r in rows:
    print(
        f"{r['input']:>7} "
        f"{r['output']:>7} "
        f"{r['concurrency']:>6} "
        f"{r['success']:>6} "
        f"{r['duration']:>8.2f} "
        f"{r['ttft']:>10.2f} "
        f"{r['tpot']:>10.2f} "
        f"{r['itl']:>10.2f} "
        f"{r['input_tps']:>13.2f} "
        f"{r['output_tps']:>14.2f} "
        f"{r['total_tps']:>13.2f}"
    )

print("=" * 118)
print(f"Total result files: {len(rows)}")