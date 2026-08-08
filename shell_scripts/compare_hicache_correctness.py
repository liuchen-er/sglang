import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/projects/sglang-qwen2-adaptive-prefill/hicache/results")
RESTORE_FILE = BASE / "correctness_always_restore.jsonl"
RECOMPUTE_FILE = BASE / "correctness_always_recompute.jsonl"


def load(path):
    data = {}
    duplicate = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            key = (x["prompt_len"], x["trial"])
            if key in data:
                duplicate += 1
            data[key] = x
    if duplicate:
        print(
            f"[WARN] {path.name}: found {duplicate} duplicate keys, using the latest records."
        )
    return data


restore = load(RESTORE_FILE)
recompute = load(RECOMPUTE_FILE)
keys = sorted(set(restore) | set(recompute))

print("=" * 118)
print(
    f"{'Len':>6} {'Trial':>5} {'R_cached':>9} {'R_load':>9} {'C_cached':>9} {'C_load':>9} "
    f"{'Output':>8} {'R_ms':>10} {'C_ms':>10} {'R-C(ms)':>10} {'Speedup':>9} {'Path':>6}"
)
print("=" * 118)

passed = 0
failed = 0
groups = defaultdict(list)

for key in keys:
    r = restore.get(key)
    c = recompute.get(key)
    if r is None or c is None:
        print(
            f"[MISSING] len={key[0]}, trial={key[1]}, restore={r is not None}, recompute={c is not None}"
        )
        failed += 1
        continue

    r_load = r["revisit_load_back_delta"]
    c_load = c["revisit_load_back_delta"]
    r_cached = r.get("revisit_cached_tokens")
    c_cached = c.get("revisit_cached_tokens")
    r_ms = r["revisit_latency_ms"]
    c_ms = c["revisit_latency_ms"]

    output_same = r["revisit_output"] == c["revisit_output"]
    restore_path_ok = r_load > 0
    recompute_path_ok = c_load == 0
    path_ok = restore_path_ok and recompute_path_ok
    case_ok = output_same and path_ok

    delta_ms = r_ms - c_ms
    speedup = c_ms / r_ms if r_ms > 0 else float("nan")

    print(
        f"{key[0]:6d} {key[1]:5d} {str(r_cached):>9} {r_load:9.0f} {str(c_cached):>9} {c_load:9.0f} "
        f"{str(output_same):>8} {r_ms:10.3f} {c_ms:10.3f} {delta_ms:10.3f} {speedup:8.3f}x "
        f"{'PASS' if path_ok else 'FAIL':>6}"
    )

    groups[key[0]].append(
        {
            "r_ms": r_ms,
            "c_ms": c_ms,
            "output_same": output_same,
            "path_ok": path_ok,
            "r_load": r_load,
            "c_load": c_load,
        }
    )

    if case_ok:
        passed += 1
    else:
        failed += 1
        if not output_same:
            print(f"       [OUTPUT FAIL] restore={r['revisit_output']!r}")
            print(f"                     recompute={c['revisit_output']!r}")
        if not restore_path_ok:
            print(
                f"       [PATH FAIL] always_restore load_back_delta={r_load}, expected > 0"
            )
        if not recompute_path_ok:
            print(
                f"       [PATH FAIL] always_recompute load_back_delta={c_load}, expected = 0"
            )

print("\n" + "=" * 100)
print("SUMMARY BY PROMPT LENGTH")
print("=" * 100)
print(
    f"{'Len':>6} {'N':>4} {'Restore(ms)':>13} {'Recompute(ms)':>15} {'R-C(ms)':>11} "
    f"{'Speedup':>10} {'Output':>10} {'Path':>10}"
)
print("-" * 100)

for prompt_len in sorted(groups):
    rows = groups[prompt_len]
    r_med = statistics.median(x["r_ms"] for x in rows)
    c_med = statistics.median(x["c_ms"] for x in rows)
    delta = r_med - c_med
    speedup = c_med / r_med if r_med > 0 else float("nan")
    output_pass = sum(x["output_same"] for x in rows)
    path_pass = sum(x["path_ok"] for x in rows)

    print(
        f"{prompt_len:6d} {len(rows):4d} {r_med:13.3f} {c_med:15.3f} {delta:11.3f} "
        f"{speedup:9.3f}x {output_pass:3d}/{len(rows):<6d} {path_pass:3d}/{len(rows):<6d}"
    )

print("\n" + "=" * 100)
print("FINAL")
print("=" * 100)
print(f"PASS = {passed}")
print(f"FAIL = {failed}")

if failed == 0:
    print("[ALL PASS] Restore/Recompute outputs and control paths are consistent.")
else:
    print("[FAIL] Some cases have output mismatch or incorrect control path.")
