#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def max_item(rows, key):
    valid = [
        r for r in rows
        if num(r.get(key)) is not None
    ]

    if not valid:
        return None, None

    row = max(
        valid,
        key=lambda r: num(r[key]),
    )

    return num(row[key]), num(row["elapsed_s"])


def weighted_mean(rows, value_key, count_key):
    total = 0.0
    count = 0.0

    for r in rows:
        v = num(r.get(value_key))
        c = num(r.get(count_key))

        if v is None or c is None or c <= 0:
            continue

        total += v * c
        count += c

    if count == 0:
        return None

    return total / count


def total_completed(rows):
    total = 0.0

    for r in rows:
        v = num(r.get("kv_completed"))
        if v is not None:
            total += v

    return total


def aggregate_bandwidth(rows):
    total_mb = 0.0
    total_latency_ms = 0.0

    for r in rows:
        c = num(r.get("kv_completed"))
        mb = num(r.get("kv_size_mb"))
        lat = num(r.get("kv_latency_ms"))

        if (
            c is None
            or c <= 0
            or mb is None
            or lat is None
        ):
            continue

        total_mb += mb * c
        total_latency_ms += lat * c

    if total_latency_ms <= 0:
        return None

    return (
        (total_mb / 1024.0)
        / (total_latency_ms / 1000.0)
    )


def fmt(v, unit=""):
    if v is None:
        return "-"
    return f"{v:.2f}{unit}"


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python 18_summarize_pd_pipeline.py <csv>"
        )
        sys.exit(1)

    path = Path(sys.argv[1])

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Empty CSV")

    queue_metrics = [
        (
            "Prefill bootstrap queue",
            "prefill_bootstrap_queue",
        ),
        (
            "Prefill inflight queue",
            "prefill_inflight_queue",
        ),
        (
            "Prefill scheduler queue",
            "prefill_waiting",
        ),
        (
            "Decode prealloc queue",
            "decode_prealloc_queue",
        ),
        (
            "Decode transfer queue",
            "decode_transfer_queue",
        ),
        (
            "Decode scheduler queue",
            "decode_waiting",
        ),
    ]

    print()
    print("=" * 78)
    print("PD PIPELINE TIME-SERIES SUMMARY")
    print("=" * 78)

    print()
    print("Queue peak:")
    print(
        f"{'Metric':<32}"
        f"{'Peak':>10}"
        f"{'At(s)':>12}"
    )
    print("-" * 54)

    for name, key in queue_metrics:
        peak, ts = max_item(rows, key)

        print(
            f"{name:<32}"
            f"{fmt(peak):>10}"
            f"{fmt(ts):>12}"
        )

    completed = total_completed(rows)

    kv_lat = weighted_mean(
        rows,
        "kv_latency_ms",
        "kv_completed",
    )

    kv_size = weighted_mean(
        rows,
        "kv_size_mb",
        "kv_completed",
    )

    kv_reported = weighted_mean(
        rows,
        "kv_reported_gbs",
        "kv_completed",
    )

    kv_effective = aggregate_bandwidth(rows)

    pf_queue = weighted_mean(
        rows,
        "prefill_queue_ms",
        "kv_completed",
    )

    pf_forward = weighted_mean(
        rows,
        "prefill_forward_ms",
        "kv_completed",
    )

    pf_xfer = weighted_mean(
        rows,
        "prefill_transfer_stage_ms",
        "kv_completed",
    )

    d_xfer = weighted_mean(
        rows,
        "decode_transferred_ms",
        "kv_completed",
    )

    d_queue = weighted_mean(
        rows,
        "decode_queue_ms",
        "kv_completed",
    )

    print()
    print("Transfer / stage summary:")
    print("-" * 54)

    print(
        f"{'KV completions':<32}"
        f"{completed:>15.0f}"
    )

    print(
        f"{'KV latency':<32}"
        f"{fmt(kv_lat, ' ms'):>15}"
    )

    print(
        f"{'KV size':<32}"
        f"{fmt(kv_size, ' MiB'):>15}"
    )

    print(
        f"{'KV reported speed':<32}"
        f"{fmt(kv_reported, ' GB/s'):>15}"
    )

    print(
        f"{'KV effective bandwidth':<32}"
        f"{fmt(kv_effective, ' GiB/s'):>15}"
    )

    print(
        f"{'Prefill queue':<32}"
        f"{fmt(pf_queue, ' ms'):>15}"
    )

    print(
        f"{'Prefill forward':<32}"
        f"{fmt(pf_forward, ' ms'):>15}"
    )

    print(
        f"{'Prefill transfer stage':<32}"
        f"{fmt(pf_xfer, ' ms'):>15}"
    )

    print(
        f"{'Decode transferred':<32}"
        f"{fmt(d_xfer, ' ms'):>15}"
    )

    print(
        f"{'Decode queue':<32}"
        f"{fmt(d_queue, ' ms'):>15}"
    )

    print()
    print(f"Samples: {len(rows)}")
    print(f"File   : {path}")


if __name__ == "__main__":
    main()
