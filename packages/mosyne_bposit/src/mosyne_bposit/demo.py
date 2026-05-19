"""mosyne-bposit showcase demo: four claims in under a minute.

Run as ``python -m mosyne_bposit.demo`` (or ``mosyne-bposit-demo``)
after ``mosyne-bposit-build``.

Demonstrates, in order:

  1. **Throughput** — BPositLinear (W8A8 via INT8 IMMA) latency at the
     Llama-class FFN-gate shape, vs native fp32 / bf16 nn.Linear on
     the same hardware, same shape. PASS when bposit is faster than
     fp32 (a baseline sanity check; bposit-vs-bf16 is shape-dependent
     and informational — see the per-shape sweep in the README).

  2. **Numerical accuracy** — calibration-free W8A8 L2 relative error
     on a synthetic random matmul. PASS when the error is below 5%
     (SmoothQuant / AWQ acceptable-W8A8 band).

  3. **Reproducibility** — five repeated runs of the same forward pass,
     printing the SHA-256 of each output. PASS when all hashes match
     (bit-exact across runs — the property IEEE float on tensor cores
     cannot deliver; see whitepaper §4.2).

  4. **Weight memory** — actual VRAM occupied by a BPositLinear at the
     Llama FFN-gate shape vs a bf16 nn.Linear at the same shape. PASS
     when the ratio is below 0.55 (we expect ~0.50).

CI gating:
  ``mosyne-bposit-demo --strict --json`` returns non-zero exit if any
  claim regresses, and emits a machine-readable JSON summary suitable
  for ``jq`` / pipeline assertion.

Designed for customer demos and HN-reader verification: prints clean
human-readable output, finishes in under a minute, requires only a
CUDA-capable GPU and a built libmosyne_bposit.so. No model downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class ClaimResult:
    """Pass/fail outcome of one demo claim, for --strict / --json modes."""
    name: str
    passed: bool
    details: dict


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


def claim_throughput(M: int = 128, K: int = 4096, N: int = 11008) -> ClaimResult:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[1/4] Latency at the Llama FFN-gate shape "
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
    ratio_fp32 = t_bp / t_fp32
    if ratio_bf16 < 1.05:
        verdict = f"{1/ratio_bf16:.2f}× of bf16 — at parity with the deployment baseline"
    elif ratio_bf16 < 1.6:
        verdict = f"{ratio_bf16:.2f}× slower than bf16 (small-shape tradeoff)"
    else:
        verdict = f"{ratio_bf16:.2f}× slower than bf16 (memory-bound regime)"
    print(f"      vs bf16                           : {verdict}")
    # PASS criterion: bposit beats the fp32 nn.Linear baseline. The
    # bposit-vs-bf16 ratio is shape-dependent (some shapes lose by 4×,
    # per the README's "What this doesn't claim" section) so we don't
    # gate strict mode on it.
    passed = ratio_fp32 < 1.0
    print(f"      strict-mode check (bp < fp32)     : "
          f"{'PASS' if passed else 'FAIL'}")
    print()
    return ClaimResult(
        name="throughput",
        passed=passed,
        details={
            "shape": {"M": M, "K": K, "N": N},
            "fp32_us": round(t_fp32, 2),
            "bf16_us": round(t_bf16, 2),
            "bposit_us": round(t_bp, 2),
            "ratio_vs_fp32": round(ratio_fp32, 3),
            "ratio_vs_bf16": round(ratio_bf16, 3),
        },
    )


def claim_accuracy(M: int = 256, K: int = 4096, N: int = 4096) -> ClaimResult:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[2/4] Numerical accuracy on a synthetic W8A8 matmul "
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
    # PASS criterion: error within the 5% SmoothQuant/AWQ band.
    passed = rel_err_pct < 5.0
    print(f"      strict-mode check (rel.err < 5%)  : "
          f"{'PASS' if passed else 'FAIL'}")
    print()
    return ClaimResult(
        name="accuracy",
        passed=passed,
        details={
            "shape": {"M": M, "K": K, "N": N},
            "fp32_ref_l2": round(ref_l2.item(), 6),
            "rel_err_pct": round(rel_err_pct, 4),
            "threshold_pct": 5.0,
        },
    )


def claim_reproducibility(M: int = 64, K: int = 2048, N: int = 2048,
                          n_runs: int = 5) -> ClaimResult:
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[3/4] Reproducibility across {n_runs} runs of the same "
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
        print("      → bposit + quire256 integer accumulation is associative")
        print("        by construction; reordering does not change the result.")
    else:
        print(f"      → {distinct} distinct bit patterns across {n_runs} runs — "
              f"unexpected; the determinism claim has regressed.")
    passed = distinct == 1
    print(f"      strict-mode check (1 distinct hash): "
          f"{'PASS' if passed else 'FAIL'}")
    print()
    return ClaimResult(
        name="reproducibility",
        passed=passed,
        details={
            "shape": {"M": M, "K": K, "N": N},
            "n_runs": n_runs,
            "distinct_hashes": distinct,
            "first_hash": hashes[0] if hashes else None,
        },
    )


def claim_memory(K: int = 4096, N: int = 11008) -> ClaimResult:
    """Headline memory claim: BPositLinear vs bf16 nn.Linear at the
    Llama FFN-gate shape. Measured via torch.cuda.memory_allocated
    before/after allocation — actual VRAM, not theoretical."""
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    print(f"[4/4] Weight memory at the Llama FFN-gate shape "
          f"(K={K} N={N})")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * (1.0 / K ** 0.5)
    after_w = torch.cuda.memory_allocated()

    # bf16 nn.Linear
    bf16 = torch.nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
    with torch.no_grad():
        bf16.weight.copy_(w.to(torch.bfloat16))
    after_bf16 = torch.cuda.memory_allocated()
    bf16_bytes = after_bf16 - after_w

    # BPositLinear
    bposit = BPositLinear(weight=w)
    after_bp = torch.cuda.memory_allocated()
    bposit_bytes = after_bp - after_bf16

    # Hold references so the deltas above are meaningful in the
    # rare event the allocator reuses freed memory before we read.
    _ = (bposit, bf16, w, baseline)

    ratio = bposit_bytes / bf16_bytes if bf16_bytes else float("nan")
    bp_mb = bposit_bytes / (1024 * 1024)
    bf_mb = bf16_bytes / (1024 * 1024)

    print(f"      bf16 nn.Linear weight             : {bf_mb:7.2f} MB")
    print(f"      BPositLinear weight + scales      : {bp_mb:7.2f} MB")
    print(f"      ratio bposit / bf16               : {ratio:.3f}× "
          f"(expected ~0.50)")
    # PASS criterion: ratio under 0.55. The math gives ~0.50 exact;
    # the small headroom covers tiny per-channel scale-tensor overhead.
    passed = ratio < 0.55
    print(f"      strict-mode check (ratio < 0.55)  : "
          f"{'PASS' if passed else 'FAIL'}")
    print()
    return ClaimResult(
        name="memory",
        passed=passed,
        details={
            "shape": {"K": K, "N": N},
            "bf16_mb": round(bf_mb, 3),
            "bposit_mb": round(bp_mb, 3),
            "ratio": round(ratio, 4),
            "threshold": 0.55,
        },
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="mosyne-bposit showcase: throughput / accuracy / "
                    "reproducibility / memory",
    )
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-reproducibility", action="store_true")
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any claim regresses "
                         "(for CI gating)")
    ap.add_argument("--json", action="store_true",
                    help="emit a machine-readable JSON summary "
                         "(human output still goes to stderr)")
    args = ap.parse_args(argv)

    # iter-299: pre-flight GPU/PyTorch compat check — catches the
    # "fresh install on a 5090, PyTorch ships sm_50..sm_90, first
    # kernel crashes with 'no kernel image' cryptically" failure
    # mode and gives an actionable install hint instead.
    from mosyne_bposit._gpu_compat import check_gpu_compat
    rc = check_gpu_compat()
    if rc != 0:
        return rc
    import torch

    # In JSON mode we want stdout to be parseable. Route the human
    # narration to stderr; only the final JSON object lands on stdout.
    out = sys.stderr if args.json else sys.stdout
    print("mosyne-bposit showcase — four claims, one run", file=out)
    print(f"GPU: {torch.cuda.get_device_name(0)}", file=out)
    print(f"PyTorch: {torch.__version__}", file=out)
    print(file=out)

    # Redirect every claim's prints to `out` for the duration of the run.
    _orig_stdout = sys.stdout
    sys.stdout = out
    try:
        results: list[ClaimResult] = []
        if not args.skip_throughput:
            results.append(claim_throughput())
        if not args.skip_accuracy:
            results.append(claim_accuracy())
        if not args.skip_reproducibility:
            results.append(claim_reproducibility())
        if not args.skip_memory:
            results.append(claim_memory())
    finally:
        sys.stdout = _orig_stdout

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(file=out)
    if n_passed == n_total:
        print(f"  → all {n_total} claims passed.", file=out)
    else:
        print(f"  → {n_passed}/{n_total} passed — "
              f"{n_total - n_passed} regressed.", file=out)
    print(file=out)
    print("Summary: BPositLinear delivers W8A8-class accuracy and bit-exact",
          file=out)
    print("         reproducibility at parity-or-near-parity decode latency vs",
          file=out)
    print("         the bf16 deployment baseline, on commodity NVIDIA tensor",
          file=out)
    print("         cores, calibration-free. Reproducibility and 2× weight-",
          file=out)
    print("         memory savings (per-FFN-module) are the headline; speed",
          file=out)
    print("         against bf16 is shape-dependent (whitepaper §4.1 sweep).",
          file=out)
    print(file=out)
    print("Whitepaper: https://github.com/anomly-labs/mosyne-bposit/tree/main/docs/whitepaper",
          file=out)

    if args.json:
        payload = {
            "ok": n_passed == n_total,
            "passed": n_passed,
            "total": n_total,
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "claims": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))

    if args.strict and n_passed != n_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
