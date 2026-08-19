#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8


def sm120_int8_scaled_mm_fallback(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
):
    """
    Functional fallback for SGLang W8A8 on SM120.

    x_q:
        [M, K], INT8, row-major

    weight:
        [K, N], INT8, column-major
        This matches SGLang runtime weight layout.

    x_scale:
        [M, 1], FP32, per-token

    weight_scale:
        [N, 1] or [N], per-output-channel
    """

    assert x_q.dtype == torch.int8
    assert weight.dtype == torch.int8

    M, K = x_q.shape
    K2, N = weight.shape

    assert K == K2

    # torch._int_mm requires M > 16.
    if M <= 16:
        padded_m = 17

        x_work = F.pad(
            x_q,
            (0, 0, 0, padded_m - M),
            mode="constant",
            value=0,
        )
    else:
        x_work = x_q

    # INT8 x INT8 -> INT32
    acc = torch._int_mm(
        x_work,
        weight,
    )

    # Remove padded rows.
    acc = acc[:M]

    # Restore quantization scales.
    #
    # x_scale      [M, 1]
    # weight_scale [N, 1] -> [1, N]
    out = (
        acc.float()
        * x_scale.reshape(M, 1).float()
        * weight_scale.reshape(1, N).float()
    )

    if bias is not None:
        out = out + bias.reshape(1, N).float()

    return out.to(out_dtype)


def quantize_weight_per_channel(weight_fp):
    """
    Simulate the W8A8 checkpoint format.

    Original weight:
        [N, K]

    Quantized checkpoint:
        weight INT8      [N, K]
        weight_scale     [N, 1]

    SGLang runtime:
        weight.t()       [K, N], column-major
    """

    absmax = (
        weight_fp.float()
        .abs()
        .amax(dim=1, keepdim=True)
        .clamp_min(1e-10)
    )

    scale = absmax / 127.0

    weight_q = torch.round(
        weight_fp.float() / scale
    ).clamp(
        -127, 127
    ).to(torch.int8)

    return weight_q, scale


def run_test(M, K, N):
    print()
    print("=" * 70)
    print(f"M={M}, K={K}, N={N}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Simulate BF16 Linear
    # ------------------------------------------------------------

    x = torch.randn(
        M,
        K,
        device="cuda",
        dtype=torch.bfloat16,
    )

    weight_fp = (
        torch.randn(
            N,
            K,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )

    bias = (
        torch.randn(
            N,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.01
    )

    # ------------------------------------------------------------
    # Weight quantization: static per-channel INT8
    # ------------------------------------------------------------

    weight_q_checkpoint, weight_scale = (
        quantize_weight_per_channel(weight_fp)
    )

    # Mimic SGLang process_weights_after_loading():
    #
    # checkpoint:
    #   [N, K], row-major
    #
    # runtime:
    #   [K, N], column-major
    weight_runtime = weight_q_checkpoint.t()

    # ------------------------------------------------------------
    # Activation quantization: dynamic per-token INT8
    # ------------------------------------------------------------

    x_q, x_scale = per_token_quant_int8(x)

    print("x:")
    print("  dtype :", x.dtype)
    print("  shape :", tuple(x.shape))

    print("x_q:")
    print("  dtype :", x_q.dtype)
    print("  shape :", tuple(x_q.shape))
    print("  stride:", x_q.stride())

    print("x_scale:")
    print("  dtype :", x_scale.dtype)
    print("  shape :", tuple(x_scale.shape))

    print("weight:")
    print("  dtype :", weight_runtime.dtype)
    print("  shape :", tuple(weight_runtime.shape))
    print("  stride:", weight_runtime.stride())

    print("weight_scale:")
    print("  dtype :", weight_scale.dtype)
    print("  shape :", tuple(weight_scale.shape))

    # ------------------------------------------------------------
    # SM120 fallback
    # ------------------------------------------------------------

    y_fallback = sm120_int8_scaled_mm_fallback(
        x_q=x_q,
        weight=weight_runtime,
        x_scale=x_scale,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=bias,
    )

    # ------------------------------------------------------------
    # Quantized mathematical reference
    # ------------------------------------------------------------

    acc_ref = (
        x_q.float()
        @ weight_runtime.float()
    )

    y_quant_ref = (
        acc_ref
        * x_scale.float()
        * weight_scale.reshape(1, N).float()
        + bias.reshape(1, N).float()
    )

    y_quant_ref_bf16 = y_quant_ref.to(
        torch.bfloat16
    )

    # ------------------------------------------------------------
    # Original BF16 Linear
    # ------------------------------------------------------------

    y_bf16 = F.linear(
        x,
        weight_fp,
        bias,
    )

    # ------------------------------------------------------------
    # Error 1:
    # fallback vs exact quantized computation
    # ------------------------------------------------------------

    fallback_error = (
        y_fallback.float()
        - y_quant_ref_bf16.float()
    ).abs()

    # ------------------------------------------------------------
    # Error 2:
    # W8A8 vs original BF16
    # ------------------------------------------------------------

    quant_error = (
        y_fallback.float()
        - y_bf16.float()
    ).abs()

    print()
    print("Fallback vs quantized reference:")
    print(
        "  max abs error :",
        fallback_error.max().item(),
    )
    print(
        "  mean abs error:",
        fallback_error.mean().item(),
    )

    print()
    print("W8A8 vs BF16:")
    print(
        "  max abs error :",
        quant_error.max().item(),
    )
    print(
        "  mean abs error:",
        quant_error.mean().item(),
    )

    if fallback_error.max().item() == 0:
        print()
        print("PASS")
    else:
        print()
        print("CHECK REQUIRED")


def main():
    torch.manual_seed(42)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )

    # Qwen2.5-7B q_proj-like shape
    K = 3584
    N = 3584

    for M in [
        1,
        4,
        16,
        17,
        32,
        128,
    ]:
        run_test(
            M=M,
            K=K,
            N=N,
        )


if __name__ == "__main__":
    main()