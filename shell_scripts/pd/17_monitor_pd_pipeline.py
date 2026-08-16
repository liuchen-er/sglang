#!/usr/bin/env python3

import argparse
import csv
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path


LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(.*)\})?\s+'
    r'([-+0-9.eE]+|NaN|Inf|-Inf)$'
)

LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def fetch_metrics(url):
    with urllib.request.urlopen(url, timeout=3) as resp:
        text = resp.read().decode("utf-8")

    result = []

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        m = LINE_RE.match(line)
        if not m:
            continue

        name = m.group(1)
        labels_str = m.group(2) or ""
        value_str = m.group(3)

        try:
            value = float(value_str)
        except ValueError:
            continue

        labels = dict(LABEL_RE.findall(labels_str))

        result.append(
            {
                "name": name,
                "labels": labels,
                "value": value,
            }
        )

    return result


def get_metric(metrics, name, stage=None):
    values = []

    for item in metrics:
        if item["name"] != name:
            continue

        labels = item["labels"]

        if stage is not None:
            if labels.get("stage") != stage:
                continue

        values.append(item["value"])

    if not values:
        return None

    return sum(values)


def safe_delta(cur, prev):
    if cur is None or prev is None:
        return None

    d = cur - prev

    # Prometheus process restart / counter reset
    if d < 0:
        return None

    return d


def safe_mean(delta_sum, delta_count, scale=1.0):
    if (
        delta_sum is None
        or delta_count is None
        or delta_count <= 0
    ):
        return None

    return delta_sum / delta_count * scale


def fmt(v):
    if v is None:
        return ""
    return v


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prefill-url",
        default="http://127.0.0.1:30000/metrics",
    )

    parser.add_argument(
        "--decode-url",
        default="http://127.0.0.1:30001/metrics",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--stop-file",
        required=True,
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stop_file = Path(args.stop_file)

    fields = [
        "timestamp",
        "elapsed_s",

        # instantaneous queue gauges
        "prefill_bootstrap_queue",
        "prefill_inflight_queue",
        "prefill_running",
        "prefill_waiting",

        "decode_prealloc_queue",
        "decode_transfer_queue",
        "decode_running",
        "decode_waiting",

        # interval KV statistics
        "kv_completed",
        "kv_latency_ms",
        "kv_size_mb",
        "kv_reported_gbs",
        "kv_effective_gbs",

        # interval stage statistics
        "prefill_queue_ms",
        "prefill_forward_ms",
        "prefill_transfer_stage_ms",
        "decode_transferred_ms",
        "decode_queue_ms",
    ]

    previous = None

    start = time.monotonic()

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
        buffering=1,
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        while not stop_file.exists():

            try:
                pm = fetch_metrics(args.prefill_url)
                dm = fetch_metrics(args.decode_url)

            except Exception as e:
                print(
                    f"[WARN] metrics fetch failed: {e}",
                    flush=True,
                )
                time.sleep(args.interval)
                continue

            # ----------------------------
            # Instantaneous gauges
            # ----------------------------

            gauges = {
                "prefill_bootstrap_queue": get_metric(
                    pm,
                    "sglang:num_prefill_bootstrap_queue_reqs",
                ),

                "prefill_inflight_queue": get_metric(
                    pm,
                    "sglang:num_prefill_inflight_queue_reqs",
                ),

                "prefill_running": get_metric(
                    pm,
                    "sglang:num_running_reqs",
                ),

                "prefill_waiting": get_metric(
                    pm,
                    "sglang:num_queue_reqs",
                ),

                "decode_prealloc_queue": get_metric(
                    dm,
                    "sglang:num_decode_prealloc_queue_reqs",
                ),

                "decode_transfer_queue": get_metric(
                    dm,
                    "sglang:num_decode_transfer_queue_reqs",
                ),

                "decode_running": get_metric(
                    dm,
                    "sglang:num_running_reqs",
                ),

                "decode_waiting": get_metric(
                    dm,
                    "sglang:num_queue_reqs",
                ),
            }

            # ----------------------------
            # Cumulative Prometheus values
            # ----------------------------

            current = {
                # KV
                "kv_lat_sum": get_metric(
                    pm,
                    "sglang:kv_transfer_latency_ms_sum",
                ),

                "kv_lat_count": get_metric(
                    pm,
                    "sglang:kv_transfer_latency_ms_count",
                ),

                "kv_mb_sum": get_metric(
                    pm,
                    "sglang:kv_transfer_total_mb_sum",
                ),

                "kv_speed_sum": get_metric(
                    pm,
                    "sglang:kv_transfer_speed_gb_s_sum",
                ),

                # Prefill scheduler queue
                "pf_q_sum": get_metric(
                    pm,
                    "sglang:queue_time_seconds_sum",
                ),

                "pf_q_count": get_metric(
                    pm,
                    "sglang:queue_time_seconds_count",
                ),

                # Decode scheduler queue
                "d_q_sum": get_metric(
                    dm,
                    "sglang:queue_time_seconds_sum",
                ),

                "d_q_count": get_metric(
                    dm,
                    "sglang:queue_time_seconds_count",
                ),

                # Prefill forward
                "pf_fwd_sum": get_metric(
                    pm,
                    "sglang:per_stage_req_latency_seconds_sum",
                    stage="prefill_forward",
                ),

                "pf_fwd_count": get_metric(
                    pm,
                    "sglang:per_stage_req_latency_seconds_count",
                    stage="prefill_forward",
                ),

                # Prefill transfer stage
                "pf_xfer_sum": get_metric(
                    pm,
                    "sglang:per_stage_req_latency_seconds_sum",
                    stage="prefill_transfer_kv_cache",
                ),

                "pf_xfer_count": get_metric(
                    pm,
                    "sglang:per_stage_req_latency_seconds_count",
                    stage="prefill_transfer_kv_cache",
                ),

                # Decode transferred
                "d_xfer_sum": get_metric(
                    dm,
                    "sglang:per_stage_req_latency_seconds_sum",
                    stage="decode_transferred",
                ),

                "d_xfer_count": get_metric(
                    dm,
                    "sglang:per_stage_req_latency_seconds_count",
                    stage="decode_transferred",
                ),
            }

            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),

                "elapsed_s": round(
                    time.monotonic() - start,
                    3,
                ),

                **gauges,
            }

            if previous is not None:

                d_kv_count = safe_delta(
                    current["kv_lat_count"],
                    previous["kv_lat_count"],
                )

                d_kv_lat_sum = safe_delta(
                    current["kv_lat_sum"],
                    previous["kv_lat_sum"],
                )

                d_kv_mb_sum = safe_delta(
                    current["kv_mb_sum"],
                    previous["kv_mb_sum"],
                )

                d_kv_speed_sum = safe_delta(
                    current["kv_speed_sum"],
                    previous["kv_speed_sum"],
                )

                row["kv_completed"] = fmt(d_kv_count)

                row["kv_latency_ms"] = fmt(
                    safe_mean(
                        d_kv_lat_sum,
                        d_kv_count,
                    )
                )

                row["kv_size_mb"] = fmt(
                    safe_mean(
                        d_kv_mb_sum,
                        d_kv_count,
                    )
                )

                row["kv_reported_gbs"] = fmt(
                    safe_mean(
                        d_kv_speed_sum,
                        d_kv_count,
                    )
                )

                if (
                    d_kv_lat_sum is not None
                    and d_kv_mb_sum is not None
                    and d_kv_lat_sum > 0
                ):
                    row["kv_effective_gbs"] = (
                        (d_kv_mb_sum / 1024.0)
                        / (d_kv_lat_sum / 1000.0)
                    )
                else:
                    row["kv_effective_gbs"] = ""

                # Prefill queue
                row["prefill_queue_ms"] = fmt(
                    safe_mean(
                        safe_delta(
                            current["pf_q_sum"],
                            previous["pf_q_sum"],
                        ),
                        safe_delta(
                            current["pf_q_count"],
                            previous["pf_q_count"],
                        ),
                        1000.0,
                    )
                )

                # Prefill forward
                row["prefill_forward_ms"] = fmt(
                    safe_mean(
                        safe_delta(
                            current["pf_fwd_sum"],
                            previous["pf_fwd_sum"],
                        ),
                        safe_delta(
                            current["pf_fwd_count"],
                            previous["pf_fwd_count"],
                        ),
                        1000.0,
                    )
                )

                # Prefill transfer stage
                row["prefill_transfer_stage_ms"] = fmt(
                    safe_mean(
                        safe_delta(
                            current["pf_xfer_sum"],
                            previous["pf_xfer_sum"],
                        ),
                        safe_delta(
                            current["pf_xfer_count"],
                            previous["pf_xfer_count"],
                        ),
                        1000.0,
                    )
                )

                # Decode transfer wait
                row["decode_transferred_ms"] = fmt(
                    safe_mean(
                        safe_delta(
                            current["d_xfer_sum"],
                            previous["d_xfer_sum"],
                        ),
                        safe_delta(
                            current["d_xfer_count"],
                            previous["d_xfer_count"],
                        ),
                        1000.0,
                    )
                )

                # Decode queue
                row["decode_queue_ms"] = fmt(
                    safe_mean(
                        safe_delta(
                            current["d_q_sum"],
                            previous["d_q_sum"],
                        ),
                        safe_delta(
                            current["d_q_count"],
                            previous["d_q_count"],
                        ),
                        1000.0,
                    )
                )

            else:
                for field in fields:
                    if field not in row:
                        row[field] = ""

            writer.writerow(row)

            previous = current

            time.sleep(args.interval)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
