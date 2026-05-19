"""Per-shape latency sweep across transformer model classes.

Extends examples/per_shape_bench.py (3 fixed shapes) into a full
sweep spanning small-decode through Llama-70B-class FFN. Designed
to surface where bposit-W8A8 wins vs bf16 and where it loses, so
deployment teams can match the bposit path to the shapes they
actually run.

Usage:

    # Quick run (~30s, default 200 reps / 40-iter warmup)
    python packages/mosyne_bposit/examples/per_shape_sweep.py

    # Custom workspace size — Llama-70B-class shapes may select
    # different cuBLASLt algorithms with a larger workspace.
    MOSYNE_BPOSIT_WS_MB=256 \
        python packages/mosyne_bposit/examples/per_shape_sweep.py

    # Subset of shapes for fast iteration
    python packages/mosyne_bposit/examples/per_shape_sweep.py \
        --shapes decode_qkv,ffn_gate_7b,ffn_down_70b

Methodology:

  - cuBLASLt algorithm picker primed during the 40-iter warmup
    before any measurement (~1ms of overhead amortised away).
    Skipping warmup gives misleading first-batch numbers — we
    learned this the hard way; the warmup is load-bearing.
  - Median (p50) over `reps` measurements per shape — robust to
    occasional GPU-scheduling-induced spikes.
  - Synchronises before AND after each timed call to avoid
    overlap with the previous iter's tail latency.

What this script does NOT do:

  - Compare against fp16/fp32 — the realistic deployment baseline
    is bf16. fp16 numbers are in examples/per_shape_bench.py.
  - Sweep batch sizes — only the M dim listed per shape. Real
    deployments use specific batch sizes per workload type;
    this sweep targets the per-call latency, not throughput
    over a full inference loop.
  - End-to-end model timing — see qwen_generate_bench.py for that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


# Production shape classes. M is the batch dim — separated by
# decode (M=1), prefill (M=2048), and a middle "FFN" batch
# representative of the layer-batched throughput shape Llama
# inference produces internally.
SHAPES = [
    # (label, M, K, N)
    ("decode_tiny",      1,    2048,  2048),
    ("decode_qkv",       1,    4096, 12288),
    ("decode_ffn_30b",   1,    6144, 16384),
    ("ffn_gate_7b",      128,  4096, 11008),
    ("ffn_down_7b",      128, 11008,  4096),
    ("ffn_gate_30b",     128,  6144, 16384),
    ("ffn_down_30b",     128, 16384,  6144),
    ("ffn_gate_70b",     128,  8192, 28672),
    ("ffn_down_70b",     128, 28672,  8192),
    ("prefill_qkv_7b",  2048,  4096, 12288),
    ("prefill_qkv_30b", 2048,  6144, 18432),
]


def measure(fn, x, reps: int = 200, warmup: int = 40) -> dict:
    """Return latency stats (min/p50/p90/max in µs) for `fn(x)`."""
    import torch
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return {
        "min_us": round(times[0], 2),
        "p50_us": round(times[len(times) // 2], 2),
        "p90_us": round(times[int(len(times) * 0.9)], 2),
        "max_us": round(times[-1], 2),
        "n": len(times),
    }


def run_shape(M: int, K: int, N: int, reps: int, warmup: int) -> dict:
    """Bench bposit-W8A8 vs bf16 nn.Linear at the given shape."""
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    torch.manual_seed(0)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16,
                         device="cuda") * (1.0 / K ** 0.5)
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5

    bf16 = torch.nn.Linear(K, N, bias=False,
                           dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        bf16.weight.copy_(w_bf16)

    # BPositLinear takes the float32 weight; quantises internally.
    bposit = BPositLinear(weight=w_bf16.to(torch.float32))

    bf16_stats = measure(bf16, x_bf16, reps=reps, warmup=warmup)
    bp_stats = measure(bposit, x_bf16, reps=reps, warmup=warmup)
    return {
        "bf16": bf16_stats,
        "bposit": bp_stats,
        "ratio_p50": round(bp_stats["p50_us"] / bf16_stats["p50_us"], 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-shape latency sweep — bposit-W8A8 vs bf16.")
    ap.add_argument("--reps", type=int, default=200,
                    help="Measured iterations per shape per dtype (default 200)")
    ap.add_argument("--warmup", type=int, default=40,
                    help="Warmup iterations per shape (default 40 — "
                         "load-bearing for cuBLASLt algo cache; do "
                         "not go below 20)")
    ap.add_argument("--shapes", default=None,
                    help="Comma-separated shape labels (default: all)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON on stdout "
                         "(human narration on stderr)")
    args = ap.parse_args(argv)

    # Reuse the centralised compat check so Blackwell-on-old-PyTorch
    # gives an actionable hint, not a cryptic CUDA crash.
    from mosyne_bposit._gpu_compat import check_gpu_compat
    rc = check_gpu_compat()
    if rc != 0:
        return rc
    import torch

    if args.shapes:
        wanted = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[0] in wanted]
    else:
        shapes = SHAPES

    out = sys.stderr if args.json else sys.stdout
    print("Per-shape sweep — bposit-W8A8 vs bf16", file=out)
    print(f"GPU:        {torch.cuda.get_device_name(0)}", file=out)
    print(f"PyTorch:    {torch.__version__}", file=out)
    print(f"Workspace:  {os.environ.get('MOSYNE_BPOSIT_WS_MB', '64')} MB", file=out)
    print(f"Reps:       {args.reps} (warmup={args.warmup})", file=out)
    print(file=out)
    print(f"  {'shape':<18s}  {'M':>5s} {'K':>6s} {'N':>6s}   "
          f"{'bf16 p50':>9s}   {'bp p50':>9s}   {'ratio':>6s}", file=out)
    print(f"  {'-' * 18}  {'-' * 5} {'-' * 6} {'-' * 6}   "
          f"{'-' * 9}   {'-' * 9}   {'-' * 6}", file=out)

    results = []
    for label, M, K, N in shapes:
        try:
            r = run_shape(M, K, N, reps=args.reps, warmup=args.warmup)
            ratio = r["ratio_p50"]
            ratio_str = (f"{ratio:>5.3f}×  WIN" if ratio < 1.0
                         else f"{ratio:>5.3f}×")
            print(f"  {label:<18s}  {M:>5d} {K:>6d} {N:>6d}   "
                  f"{r['bf16']['p50_us']:>7.1f}us   "
                  f"{r['bposit']['p50_us']:>7.1f}us   "
                  f"{ratio_str}", file=out)
            results.append({
                "shape": label, "M": M, "K": K, "N": N, **r,
            })
        except RuntimeError as exc:
            print(f"  {label:<18s}  {M:>5d} {K:>6d} {N:>6d}   "
                  f"FAILED: {exc}", file=out)
            results.append({
                "shape": label, "M": M, "K": K, "N": N,
                "error": str(exc),
            })
            torch.cuda.empty_cache()

    if args.json:
        print(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "ws_mb": int(os.environ.get("MOSYNE_BPOSIT_WS_MB", "64")),
            "reps": args.reps, "warmup": args.warmup,
            "results": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
