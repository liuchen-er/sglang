#!/usr/bin/env python3

import csv
import re
from pathlib import Path


EXP_ROOT = Path("/root/autodl-tmp/sglang_pd_exp")

RAW_ROOT = (
    EXP_ROOT
    / "results"
    / "raw"
    / "profile"
    / "stage_breakdown"
)

SUMMARY_ROOT = (
    EXP_ROOT
    / "results"
    / "summary"
    / "profile"
)

SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CSV = SUMMARY_ROOT / "pd_stage_breakdown.csv"


CASE_META = {
    "caseA": {
        "input_len": 4096,
        "output_len": 256,
        "rate": 0.73,
    },
    "caseB": {
        "input_len": 8192,
        "output_len": 256,
        "rate": 0.30,
    },
    "caseC": {
        "input_len": 8192,
        "output_len": 256,
        "rate": 0.54,
    },
}


LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_metrics(path: Path):
    """
    Return:
        {
            (metric_name, tuple(sorted(labels.items()))): value
        }
    """
    data = {}

    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                left, value_str = line.rsplit(None, 1)
                value = float(value_str)
            except ValueError:
                continue

            if "{" in left:
                metric_name = left.split("{", 1)[0]
                label_text = left.split("{", 1)[1].rsplit("}", 1)[0]
                labels = dict(LABEL_RE.findall(label_text))
            else:
                metric_name = left
                labels = {}

            key = (
                metric_name,
                tuple(sorted(labels.items())),
            )

            data[key] = value

    return data


def find_metric(data, metric_name, stage=None):
    """
    Find one metric value, optionally matching stage label.
    """
    matches = []

    for (name, labels_tuple), value in data.items():
        if name != metric_name:
            continue

        labels = dict(labels_tuple)

        if stage is not None and labels.get("stage") != stage:
            continue

        matches.append(value)

    if not matches:
        return None

    # Current experiment is TP=1 / PP=1, so normally exactly one.
    # Sum here also keeps the script usable if duplicate label sets appear.
    return sum(matches)


def delta_value(before, after, metric_name, stage=None):
    a = find_metric(after, metric_name, stage)
    b = find_metric(before, metric_name, stage)

    if a is None:
        return None

    if b is None:
        b = 0.0

    return a - b


def mean_from_sum_count(
    before,
    after,
    base_metric,
    stage=None,
    scale=1.0,
):
    """
    base_metric:
        e.g. sglang:queue_time_seconds
    Uses:
        <base>_sum
        <base>_count

    scale:
        seconds -> ms: 1000
        already ms: 1
    """

    total = delta_value(
        before,
        after,
        base_metric + "_sum",
        stage,
    )

    count = delta_value(
        before,
        after,
        base_metric + "_count",
        stage,
    )

    if total is None or count is None or count <= 0:
        return None, 0

    return total / count * scale, int(round(count))


def load_case_pair(case_name, role):
    """
    Preferred:
        caseX/prefill_before.metrics
        caseX/prefill_after.metrics

    Legacy Case A fallback:
        caseA_prefill.metrics
        caseA_decode.metrics
    """

    case_dir = RAW_ROOT / case_name

    before_path = case_dir / f"{role}_before.metrics"
    after_path = case_dir / f"{role}_after.metrics"

    if before_path.exists() and after_path.exists():
        return (
            parse_metrics(before_path),
            parse_metrics(after_path),
            "delta",
        )

    # Old Case A format
    legacy_path = RAW_ROOT / f"{case_name}_{role}.metrics"

    if legacy_path.exists():
        return (
            {},
            parse_metrics(legacy_path),
            "snapshot",
        )

    return {}, {}, "missing"


def fmt_ms(x):
    if x is None:
        return "-"
    return f"{x:.2f}"


def fmt_num(x, digits=2):
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def collect_case(case_name):
    meta = CASE_META[case_name]

    p_before, p_after, p_mode = load_case_pair(
        case_name,
        "prefill",
    )

    d_before, d_after, d_mode = load_case_pair(
        case_name,
        "decode",
    )

    if not p_after or not d_after:
        return None

    # -------------------------------
    # Prefill stages
    # -------------------------------

    prefill_bootstrap_ms, prefill_stage_n = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:per_stage_req_latency_seconds",
        stage="prefill_bootstrap",
        scale=1000.0,
    )

    prefill_forward_ms, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:per_stage_req_latency_seconds",
        stage="prefill_forward",
        scale=1000.0,
    )

    prefill_transfer_stage_ms, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:per_stage_req_latency_seconds",
        stage="prefill_transfer_kv_cache",
        scale=1000.0,
    )

    chunked_prefill_ms, chunked_n = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:per_stage_req_latency_seconds",
        stage="chunked_prefill",
        scale=1000.0,
    )

    prefill_queue_ms, prefill_queue_n = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:queue_time_seconds",
        scale=1000.0,
    )

    # -------------------------------
    # KV transfer
    # -------------------------------

    kv_latency_ms, kv_n = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:kv_transfer_latency_ms",
        scale=1.0,
    )

    kv_total_mb, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:kv_transfer_total_mb",
        scale=1.0,
    )

    kv_speed_avg_gbs, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:kv_transfer_speed_gb_s",
        scale=1.0,
    )

    kv_bootstrap_ms, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:kv_transfer_bootstrap_ms",
        scale=1.0,
    )

    kv_alloc_ms, _ = mean_from_sum_count(
        p_before,
        p_after,
        "sglang:kv_transfer_alloc_ms",
        scale=1.0,
    )

    # Aggregate effective bandwidth:
    # total transferred bytes / total transfer latency
    kv_total_mb_sum = delta_value(
        p_before,
        p_after,
        "sglang:kv_transfer_total_mb_sum",
    )

    kv_latency_ms_sum = delta_value(
        p_before,
        p_after,
        "sglang:kv_transfer_latency_ms_sum",
    )

    if (
        kv_total_mb_sum is not None
        and kv_latency_ms_sum is not None
        and kv_latency_ms_sum > 0
    ):
        kv_effective_gbs = (
            (kv_total_mb_sum / 1024.0)
            / (kv_latency_ms_sum / 1000.0)
        )
    else:
        kv_effective_gbs = None

    # -------------------------------
    # Decode stages
    # -------------------------------

    decode_prepare_ms, decode_stage_n = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:per_stage_req_latency_seconds",
        stage="decode_prepare",
        scale=1000.0,
    )

    decode_bootstrap_ms, _ = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:per_stage_req_latency_seconds",
        stage="decode_bootstrap",
        scale=1000.0,
    )

    decode_transferred_ms, _ = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:per_stage_req_latency_seconds",
        stage="decode_transferred",
        scale=1000.0,
    )

    decode_waiting_ms, _ = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:per_stage_req_latency_seconds",
        stage="decode_waiting",
        scale=1000.0,
    )

    fake_output_ms, _ = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:per_stage_req_latency_seconds",
        stage="fake_output",
        scale=1000.0,
    )

    decode_queue_ms, decode_queue_n = mean_from_sum_count(
        d_before,
        d_after,
        "sglang:queue_time_seconds",
        scale=1000.0,
    )

    return {
        "case": case_name,
        "input_len": meta["input_len"],
        "output_len": meta["output_len"],
        "rate": meta["rate"],

        "prefill_mode": p_mode,
        "decode_mode": d_mode,

        "prefill_stage_count": prefill_stage_n,
        "prefill_queue_count": prefill_queue_n,
        "kv_count": kv_n,
        "decode_stage_count": decode_stage_n,
        "decode_queue_count": decode_queue_n,
        "chunked_count": chunked_n,

        "prefill_bootstrap_ms": prefill_bootstrap_ms,
        "prefill_queue_ms": prefill_queue_ms,
        "prefill_forward_ms": prefill_forward_ms,
        "chunked_prefill_ms": chunked_prefill_ms,
        "prefill_transfer_stage_ms": prefill_transfer_stage_ms,

        "kv_latency_ms": kv_latency_ms,
        "kv_total_mb": kv_total_mb,
        "kv_speed_avg_gbs": kv_speed_avg_gbs,
        "kv_effective_gbs": kv_effective_gbs,
        "kv_bootstrap_ms": kv_bootstrap_ms,
        "kv_alloc_ms": kv_alloc_ms,

        "decode_prepare_ms": decode_prepare_ms,
        "decode_bootstrap_ms": decode_bootstrap_ms,
        "decode_transferred_ms": decode_transferred_ms,
        "decode_waiting_ms": decode_waiting_ms,
        "decode_queue_ms": decode_queue_ms,
        "fake_output_ms": fake_output_ms,
    }


def print_main_table(rows):
    print()
    print("=" * 160)
    print("PD STAGE BREAKDOWN: CASE A / B / C")
    print("=" * 160)

    header = (
        f"{'Case':<7}"
        f"{'Input':>8}"
        f"{'Rate':>8}"
        f"{'N':>6}"
        f"{'PF Boot':>11}"
        f"{'PF Queue':>11}"
        f"{'PF Fwd':>11}"
        f"{'PF Xfer':>11}"
        f"{'KV Lat':>11}"
        f"{'KV MB':>10}"
        f"{'KV GB/s':>10}"
        f"{'D Boot':>11}"
        f"{'D Xfer':>11}"
        f"{'D Queue':>11}"
    )

    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['case']:<7}"
            f"{r['input_len']:>8}"
            f"{r['rate']:>8.2f}"
            f"{r['kv_count']:>6}"
            f"{fmt_ms(r['prefill_bootstrap_ms']):>11}"
            f"{fmt_ms(r['prefill_queue_ms']):>11}"
            f"{fmt_ms(r['prefill_forward_ms']):>11}"
            f"{fmt_ms(r['prefill_transfer_stage_ms']):>11}"
            f"{fmt_ms(r['kv_latency_ms']):>11}"
            f"{fmt_num(r['kv_total_mb'], 1):>10}"
            f"{fmt_num(r['kv_effective_gbs'], 3):>10}"
            f"{fmt_ms(r['decode_bootstrap_ms']):>11}"
            f"{fmt_ms(r['decode_transferred_ms']):>11}"
            f"{fmt_ms(r['decode_queue_ms']):>11}"
        )

    print()
    print("Time columns: ms")
    print("KV GB/s    : aggregate effective bandwidth = total MiB / total transfer time")
    print()


def pct_change(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def print_bc_comparison(rows):
    lookup = {r["case"]: r for r in rows}

    if "caseB" not in lookup or "caseC" not in lookup:
        return

    b = lookup["caseB"]
    c = lookup["caseC"]

    print("=" * 90)
    print("CASE B -> CASE C: SAME 8K INPUT, HIGHER OFFERED LOAD")
    print("=" * 90)

    metrics = [
        ("Prefill Bootstrap", "prefill_bootstrap_ms"),
        ("Prefill Queue", "prefill_queue_ms"),
        ("Prefill Forward", "prefill_forward_ms"),
        ("Prefill Transfer Stage", "prefill_transfer_stage_ms"),
        ("KV Transfer Latency", "kv_latency_ms"),
        ("Decode Transferred", "decode_transferred_ms"),
        ("Decode Queue", "decode_queue_ms"),
    ]

    print(
        f"{'Metric':<28}"
        f"{'Case B':>14}"
        f"{'Case C':>14}"
        f"{'Change':>14}"
    )

    print("-" * 70)

    for name, key in metrics:
        bv = b[key]
        cv = c[key]
        change = pct_change(cv, bv)

        change_str = "-" if change is None else f"{change:+.1f}%"

        print(
            f"{name:<28}"
            f"{fmt_ms(bv):>14}"
            f"{fmt_ms(cv):>14}"
            f"{change_str:>14}"
        )

    print()


def print_notes(rows):
    snapshot_cases = [
        r["case"]
        for r in rows
        if r["prefill_mode"] == "snapshot"
        or r["decode_mode"] == "snapshot"
    ]

    if snapshot_cases:
        print("WARNING:")
        print(
            "  The following cases use cumulative snapshot metrics instead of"
            " before/after delta:"
        )
        print("  " + ", ".join(snapshot_cases))
        print(
            "  Their counts may include warmup / health-check requests."
        )
        print(
            "  For strict A/B/C comparison, rerun them with 15_pd_stage_case.sh."
        )
        print()


def save_csv(rows):
    fields = [
        "case",
        "input_len",
        "output_len",
        "rate",
        "prefill_mode",
        "decode_mode",
        "prefill_stage_count",
        "prefill_queue_count",
        "kv_count",
        "decode_stage_count",
        "decode_queue_count",

        "prefill_bootstrap_ms",
        "prefill_queue_ms",
        "prefill_forward_ms",
        "chunked_prefill_ms",
        "prefill_transfer_stage_ms",

        "kv_latency_ms",
        "kv_total_mb",
        "kv_speed_avg_gbs",
        "kv_effective_gbs",
        "kv_bootstrap_ms",
        "kv_alloc_ms",

        "decode_prepare_ms",
        "decode_bootstrap_ms",
        "decode_transferred_ms",
        "decode_waiting_ms",
        "decode_queue_ms",
        "fake_output_ms",
    ]

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fields
                }
            )


def main():
    rows = []

    for case_name in ("caseA", "caseB", "caseC"):
        row = collect_case(case_name)

        if row is None:
            print(
                f"[SKIP] {case_name}: metrics files not found"
            )
            continue

        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No metrics found under: {RAW_ROOT}"
        )

    print_main_table(rows)
    print_bc_comparison(rows)
    print_notes(rows)

    save_csv(rows)

    print("Saved:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
