"""Per-call latency: bposit-W8A8 BPositLinear vs native bf16 nn.Linear.

Reproduces the per-call shape table in the iter-13 research doc
(``docs/research/bposit_decode_gap_2026-05-08.md``) and the
``tab:e2e_progression`` building block of whitepaper §4.5.

Three representative shapes:

  - **FFN-gate**     ``M=128, K=4096, N=11008``  Llama-class MLP gate-proj
  - **Decode**       ``M=1,   K=4096, N=4096``   single-token autoreg step
  - **Small-square** ``M=32,  K=2048, N=2048``   small batch + small dims

Timing uses ``torch.cuda.Event`` and reports the median over many iterations
(default 200 reps after a 20-iter warmup). Both paths take a bf16 input by
default — that's the realistic deployment baseline. ``--dtype fp32`` exercises
the fp32-input fallback through ``BPositLinear`` for comparison.
"""
from __future__ import annotations
import argparse

import torch

from mosyne_bposit.torch_compat import BPositLinear


SHAPES = [
    ("FFN-gate",     128, 4096, 11008),
    ("Decode",         1, 4096,  4096),
    ("Small-square",  32, 2048,  2048),
]


def time_op(fn, n_warmup, n_iter):
    """Median elapsed time, in microseconds, of ``fn`` over ``n_iter`` reps."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2] * 1000.0


def bench_shape(name, M, K, N, dtype, device, n_warmup, n_iter, dtype_label):
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    bf = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
    bf.weight.data.copy_(w)
    bp = BPositLinear(weight=w)

    base_us = time_op(lambda: bf(x), n_warmup, n_iter)
    bp_us = time_op(lambda: bp(x), n_warmup, n_iter)

    print(
        f"  {name:14s} | {dtype_label} nn.Linear={base_us:7.1f} µs | "
        f"bposit={bp_us:7.1f} µs | gap={bp_us - base_us:+7.1f} µs | "
        f"bp/{dtype_label}={bp_us / base_us:5.2f}×"
    )
    return base_us, bp_us


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="activation/weight dtype for both paths (default: bf16)")
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=200,
                    help="reps used for the median timing")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    print(f"GPU:    {torch.cuda.get_device_name(device.index or 0)} "
          f"(cap={torch.cuda.get_device_capability(device.index or 0)})")
    print(f"dtype:  {args.dtype}  reps:   {args.n_iter}  warmup: {args.n_warmup}")
    print()

    for name, M, K, N in SHAPES:
        bench_shape(name, M, K, N, dtype, device, args.n_warmup, args.n_iter, args.dtype)


if __name__ == "__main__":
    main()
