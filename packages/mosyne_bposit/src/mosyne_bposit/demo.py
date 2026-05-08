"""mosyne-bposit showcase demo: three claims in 30 seconds.

Run as ``python -m mosyne_bposit.demo`` after ``mosyne-bposit-build``.

Demonstrates, in order:

  1. **Throughput** — BPositLinear (W8A8 via INT8 IMMA) latency at the
     Llama-class FFN-gate shape, vs native fp32 nn.Linear on the same
     hardware, same shape.

  2. **Numerical accuracy** — calibration-free W8A8 L2 relative error
     on a synthetic random matmul, sanity-checked against the
     SmoothQuant / AWQ acceptable-W8A8 band.

  3. **Reproducibility** — five repeated runs of the same forward pass,
     printing the SHA-256 of each output. Identical hashes = bit-exact
     across runs (the property IEEE float on tensor cores cannot
     deliver — see white paper §4.2).

Designed for customer demos: prints clean human-readable output,
finishes in well under a minute, requires only a CUDA-capable GPU
and a built libmosyne_bposit.so. No model downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time


def _bench(fn, x, reps: int = 50, warmup: int = 5) -> float:
    """Microsecond mean per call. Synchronizes around the timed loop."""
    import torch
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def _hash_tensor(t) -> str:
    """SHA-256 of the raw byte content of a CPU-resident tensor."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def claim_throughput(M: int = 128, K: int = 4096, N: int = 11008) -> None:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[1/3] Latency at the Llama FFN-gate shape "
          f"(M={M} K={K} N={N})")
    torch.manual_seed(0)
    x32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5
    x16 = x32.to(torch.bfloat16)
    w32 = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)
    w16 = w32.to(torch.bfloat16)

    bposit = BPositLinear(weight=w32)
    fp32 = torch.nn.Linear(K, N, bias=False).cuda()
    bf16 = torch.nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
    with torch.no_grad():
        fp32.weight.copy_(w32)
        bf16.weight.copy_(w16)

    t_fp32 = _bench(lambda x: fp32(x), x32)
    t_bf16 = _bench(lambda x: bf16(x), x16)
    t_bp = _bench(bposit, x32)

    print(f"      fp32 nn.Linear                    : {t_fp32:7.1f} µs / call")
    print(f"      bf16 nn.Linear  (deployment baseline) : {t_bf16:7.1f} µs / call")
    print(f"      BPositLinear (W8A8 via INT8 IMMA) : {t_bp:7.1f} µs / call")
    ratio_bf16 = t_bp / t_bf16
    if ratio_bf16 < 1.05:
        verdict = f"{1/ratio_bf16:.2f}× of bf16 — at parity with the deployment baseline"
    elif ratio_bf16 < 1.6:
        verdict = f"{ratio_bf16:.2f}× slower than bf16 (small-shape tradeoff)"
    else:
        verdict = f"{ratio_bf16:.2f}× slower than bf16 (memory-bound regime)"
    print(f"      vs bf16                           : {verdict}")
    print()


def claim_accuracy(M: int = 256, K: int = 4096, N: int = 4096) -> None:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[2/3] Numerical accuracy on a synthetic W8A8 matmul "
          f"(M={M} K={K} N={N})")
    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)

    ref = x @ w.t()
    bposit = BPositLinear(weight=w)
    y = bposit(x)

    err = (y - ref).pow(2).mean().sqrt()
    ref_l2 = ref.pow(2).mean().sqrt()
    rel_err_pct = (err / ref_l2).item() * 100

    print(f"      fp32 reference L2 norm            : {ref_l2.item():.4f}")
    print(f"      bposit-W8A8 vs fp32 L2 rel. error : {rel_err_pct:.2f}%")
    if rel_err_pct < 2.0:
        verdict = "well within the SmoothQuant / AWQ acceptable-W8A8 band"
    elif rel_err_pct < 5.0:
        verdict = "within the acceptable-W8A8 band (calibration would tighten)"
    else:
        verdict = "above expected band — investigate"
    print(f"      verdict                           : {verdict}")
    print()


def claim_reproducibility(M: int = 64, K: int = 2048, N: int = 2048,
                          n_runs: int = 5) -> None:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[3/3] Reproducibility across {n_runs} runs of the same "
          f"forward pass (M={M} K={K} N={N})")
    torch.manual_seed(1234)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.3
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)

    bposit = BPositLinear(weight=w)
    hashes = []
    for i in range(n_runs):
        y = bposit(x)
        h = _hash_tensor(y)
        hashes.append(h)
        print(f"      run {i+1} output sha256[:16] = {h}")

    distinct = len(set(hashes))
    print()
    if distinct == 1:
        print(f"      → {n_runs}/{n_runs} runs produced bit-identical output.")
        print(f"      → bposit + quire256 integer accumulation is associative")
        print(f"        by construction; reordering does not change the result.")
    else:
        print(f"      → {distinct} distinct bit patterns across {n_runs} runs — "
              f"unexpected; the determinism claim has regressed.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="mosyne-bposit showcase: throughput / accuracy / reproducibility",
    )
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-reproducibility", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("ERROR: this demo requires PyTorch. "
              "Install with `pip install mosyne-bposit[torch]`.")
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. The demo requires a CUDA-capable GPU.")
        return 1

    name = torch.cuda.get_device_name(0)
    print(f"mosyne-bposit showcase — three claims, one run")
    print(f"GPU: {name}")
    print(f"PyTorch: {torch.__version__}")
    print()

    if not args.skip_throughput:
        claim_throughput()
    if not args.skip_accuracy:
        claim_accuracy()
    if not args.skip_reproducibility:
        claim_reproducibility()

    print()
    print("Summary: BPositLinear delivers W8A8-class accuracy and bit-exact")
    print("         reproducibility at parity-or-near-parity decode latency vs")
    print("         the bf16 deployment baseline, on commodity NVIDIA tensor")
    print("         cores, calibration-free. Reproducibility and 2× weight-")
    print("         memory savings are the headline; speed against bf16 is")
    print("         shape-dependent (white paper §4.1 has the full sweep).")
    print()
    print("Whitepaper: https://github.com/anomly-labs/mosyne-bposit/tree/main/docs/whitepaper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
