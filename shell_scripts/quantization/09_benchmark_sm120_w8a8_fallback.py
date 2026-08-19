#!/usr/bin/env python3

import argparse
import gc

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8


WARMUP = 10
REPEAT = 50

SHAPES = {
    # name: (K, N)
    # Linear input [M, K], runtime weight [K, N]
    "q_proj": (3584, 3584),
    "kv_proj": (3584, 512),
    "gate_up_proj": (3584, 18944),
    "down_proj": (18944, 3584),
}

M_VALUES = [1, 4, 16, 17, 32, 128, 512]


def cuda_benchmark(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []

    for _ in range(repeat):
        start.record()
        fn()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return sum(times_ms) / len(times_ms)


def make_inputs(M, K, N):
    # BF16 activation
    x = torch.randn(
        M,
        K,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Simulate SGLang runtime INT8 weight:
    #
    # checkpoint:
    #   [N, K], row-major
    #
    # process_weights_after_loading:
    #   .t()
    #
    # runtime:
    #   [K, N], column-major
    weight_q = torch.randint(
        -127,
        128,
        (N, K),
        device="cuda",
        dtype=torch.int8,
    ).t()

    weight_scale = torch.rand(
        N,
        1,
        device="cuda",
        dtype=torch.float32,
    ) * 0.02 + 1e-6

    bias = torch.randn(
        N,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.01

    # BF16 baseline weight
    weight_bf16 = torch.randn(
        N,
        K,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.02

    return x, weight_q, weight_scale, bias, weight_bf16


def benchmark_one(M, K, N):
    x, weight_q, weight_scale, bias, weight_bf16 = make_inputs(
        M, K, N
    )

    # ------------------------------------------------------------
    # Prepare reusable intermediates for isolated benchmarks.
    # These are outside individual timing regions intentionally.
    # ------------------------------------------------------------

    x_q, x_scale = per_token_quant_int8(x)

    if M <= 16:
        x_work = F.pad(
            x_q,
            (0, 0, 0, 17 - M),
            mode="constant",
            value=0,
        )
    else:
        x_work = x_q

    acc_work = torch._int_mm(x_work, weight_q)
    acc = acc_work[:M]

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # 1. Activation quantization
    # ------------------------------------------------------------

    quant_ms = cuda_benchmark(
        lambda: per_token_quant_int8(x)
    )

    # ------------------------------------------------------------
    # 2. Padding
    # ------------------------------------------------------------

    if M <= 16:
        pad_ms = cuda_benchmark(
            lambda: F.pad(
                x_q,
                (0, 0, 0, 17 - M),
                mode="constant",
                value=0,
            )
        )
    else:
        pad_ms = 0.0

    # ------------------------------------------------------------
    # 3. INT8 GEMM
    #
    # x_work is already prepared, so this isolates _int_mm.
    # ------------------------------------------------------------

    int_mm_ms = cuda_benchmark(
        lambda: torch._int_mm(
            x_work,
            weight_q,
        )
    )

    # ------------------------------------------------------------
    # 4. Epilogue
    #
    # INT32
    #   ↓
    # FP32
    #   ↓
    # activation scale
    #   ↓
    # weight scale
    #   ↓
    # bias
    #   ↓
    # BF16
    # ------------------------------------------------------------

    def epilogue():
        out = (
            acc.float()
            * x_scale.reshape(M, 1).float()
            * weight_scale.reshape(1, N).float()
        )

        out = out + bias.reshape(1, N).float()

        return out.to(torch.bfloat16)

    epilogue_ms = cuda_benchmark(epilogue)

    # ------------------------------------------------------------
    # 5. Full current SM120 fallback
    # ------------------------------------------------------------

    def full_fallback():
        local_x_q, local_x_scale = per_token_quant_int8(x)

        if M <= 16:
            local_x_work = F.pad(
                local_x_q,
                (0, 0, 0, 17 - M),
                mode="constant",
                value=0,
            )
        else:
            local_x_work = local_x_q

        local_acc = torch._int_mm(
            local_x_work,
            weight_q,
        )

        local_acc = local_acc[:M]

        out = (
            local_acc.float()
            * local_x_scale.reshape(M, 1).float()
            * weight_scale.reshape(1, N).float()
        )

        out = out + bias.reshape(1, N).float()

        return out.to(torch.bfloat16)

    total_ms = cuda_benchmark(full_fallback)

    # ------------------------------------------------------------
    # 6. BF16 Linear baseline
    # ------------------------------------------------------------

    bf16_ms = cuda_benchmark(
        lambda: F.linear(
            x,
            weight_bf16,
            bias,
        )
    )

    component_sum = (
        quant_ms
        + pad_ms
        + int_mm_ms
        + epilogue_ms
    )

    padding_ratio = (
        17.0 / M if M <= 16 else 1.0
    )

    return {
        "M": M,
        "Quant": quant_ms,
        "Pad": pad_ms,
        "INT8_GEMM": int_mm_ms,
        "Epilogue": epilogue_ms,
        "PartsSum": component_sum,
        "Total": total_ms,
        "BF16": bf16_ms,
        "W8A8/BF16": total_ms / bf16_ms,
        "PadRatio": padding_ratio,
    }


def print_header(shape_name, K, N):
    print()
    print("=" * 124)
    print(
        f"Shape: {shape_name} | "
        f"[M,{K}] x [{K},{N}]"
    )
    print("=" * 124)

    print(
        f"{'M':>6} "
        f"{'Quant(ms)':>12} "
        f"{'Pad(ms)':>12} "
        f"{'INT8 GEMM':>12} "
        f"{'Epilogue':>12} "
        f"{'Parts Sum':>12} "
        f"{'Total':>12} "
        f"{'BF16':>12} "
        f"{'W8/BF16':>10} "
        f"{'Pad x':>8}"
    )

    print("-" * 124)


def print_row(r):
    print(
        f"{r['M']:6d} "
        f"{r['Quant']:12.4f} "
        f"{r['Pad']:12.4f} "
        f"{r['INT8_GEMM']:12.4f} "
        f"{r['Epilogue']:12.4f} "
        f"{r['PartsSum']:12.4f} "
        f"{r['Total']:12.4f} "
        f"{r['BF16']:12.4f} "
        f"{r['W8A8/BF16']:10.2f} "
        f"{r['PadRatio']:8.2f}"
    )


def run_shape(shape_name):
    K, N = SHAPES[shape_name]

    print_header(shape_name, K, N)

    for M in M_VALUES:
        result = benchmark_one(
            M=M,
            K=K,
            N=N,
        )

        print_row(result)

        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shape",
        choices=list(SHAPES.keys()) + ["all"],
        default="q_proj",
    )

    args = parser.parse_args()

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Warmup:", WARMUP)
    print("Repeat:", REPEAT)

    if args.shape == "all":
        names = SHAPES.keys()
    else:
        names = [args.shape]

    for name in names:
        run_shape(name)


if __name__ == "__main__":
    main()