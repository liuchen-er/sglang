import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-v2", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

input_path = Path(args.input_v2)
output_path = Path(args.output)

with input_path.open(encoding="utf-8") as f:
    profile = json.load(f)

if profile.get("version") != 2:
    raise ValueError(
        f"Expected V2 input profile, got version={profile.get('version')}"
    )

profile["version"] = 3

profile["l3"] = {
    "restore_base": [
        {"storage_hit": 256, "latency_ms": 28.935},
        {"storage_hit": 512, "latency_ms": 40.790},
        {"storage_hit": 1024, "latency_ms": 60.714},
        {"storage_hit": 2048, "latency_ms": 101.697},
        {"storage_hit": 4096, "latency_ms": 119.952},
        {"storage_hit": 8192, "latency_ms": 217.396},
        {"storage_hit": 16384, "latency_ms": 420.752}
    ],
    "recompute_base": [
        {"storage_hit": 256, "latency_ms": 17.954},
        {"storage_hit": 512, "latency_ms": 18.681},
        {"storage_hit": 1024, "latency_ms": 28.551},
        {"storage_hit": 2048, "latency_ms": 62.090},
        {"storage_hit": 4096, "latency_ms": 105.156},
        {"storage_hit": 8192, "latency_ms": 196.935},
        {"storage_hit": 16384, "latency_ms": 480.382}
    ],
    "io_backlog_ms_per_token": 0.025
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2)

print(f"Wrote V3 profile: {output_path}")
