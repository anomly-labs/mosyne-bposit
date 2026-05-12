"""Diagnostic for the bposit attention regression observed in §4.5.

Whitepaper §4.5 documents that bposit-W8A8 reaches FFN-only parity
(-0.4%) but attention regresses -13 to -16% on the same model. The
two surfaces share matmul shape and dtype — Q/K/V/O projections are
nn.Linear too — so the regression is not from the kernel itself. The
remaining hypotheses are:

  H1. **Per-shape latency**: attention shapes (5120 -> 1024 GQA
      K/V proj) hit a slower bposit path than FFN shapes
      (5120 -> 14336). Same kernel, different work-distribution.
  H2. **Quantization quality**: pre-RMSNorm hidden states have
      heavier tails than post-SiLU activations, so per-token int8
      quantization is more lossy on the attention input distribution.
  H3. **Both, with quality dominating** (most likely given that the
      iter-13 e2e bench showed FFN parity at the same kernel).

This script measures both surfaces side-by-side at Qwen3-Coder-class
shapes and two input distributions, producing a table that lets us
attribute the regression. Running on 3090 (sm_86) since PyTorch is
not yet built with sm_120 kernels for the 5090.

Usage:
    CUDA_VISIBLE_DEVICES=1 python attn_vs_ffn_diagnostic.py
"""
from __future__ import annotations
import argparse

import torch

from mosyne_bposit.torch_compat import BPositLinear


# Qwen3-Coder-30B-A3B class shapes. Hidden=4096, head_dim=128.
# Q-proj keeps full hidden; K/V use GQA reduction to 1024.
SHAPES = [
    # (label, surface, M=tokens, K, N)
    ("attn-q",     "ATTN", 512, 4096, 4096),
    ("attn-kv",    "ATTN", 512, 4096, 1024),
    ("attn-o",     "ATTN", 512, 4096, 4096),
    ("ffn-gate",   "FFN",  512, 4096, 14336),
    ("ffn-up",     "FFN",  512, 4096, 14336),
    ("ffn-down",   "FFN",  512, 14336, 4096),
    # Decode-step (single-token autoreg).
    ("attn-q-dec",  "ATTN",  1, 4096, 4096),
    ("attn-kv-dec", "ATTN",  1, 4096, 1024),
    ("ffn-gate-dec","FFN",   1, 4096, 14336),
    ("ffn-down-dec","FFN",   1, 14336, 4096),
]


def gen_norm_shaped(M, K, dtype, device):
    """Approximate post-RMSNorm hidden-state distribution: zero-mean
    unit-variance Gaussian, with the heavy-tailed outliers RMSNorm
    fails to fully suppress. SmoothQuant-style 1% outlier rows."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    if M > 1 and K > 1:
        # Inject 1% of rows with 5x amplitude — the well-known LLM
        # outlier-channel pathology that makes naive int8 hurt.
        n_outlier_rows = max(1, M // 100)
        idx = torch.randperm(M)[:n_outlier_rows]
        x[idx] *= 5.0
    return x


def gen_silu_shaped(M, K, dtype, device):
    """Approximate post-SiLU activation: x * sigmoid(g) for two
    independent gaussians. The product has tighter tails than a raw
    gaussian — empirically friendlier to per-token int8 quantization."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    g = torch.randn(M, K, dtype=dtype, device=device)
    return (x * torch.sigmoid(g)).contiguous()


INPUT_GENS = {
    "norm-like (attn input)": gen_norm_shaped,
    "silu-like (ffn-down input)": gen_silu_shaped,
}


def time_op(fn, n_warmup, n_iter):
    """Median elapsed in microseconds."""
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
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2] * 1000.0


def relative_l2(ref, got):
    """||ref - got||_2 / ||ref||_2 in fp32 to avoid measurement bias."""
    ref32 = ref.float()
    got32 = got.float()
    num = (ref32 - got32).norm()
    den = ref32.norm() + 1e-12
    return (num / den).item()


def bench_one(name, surface, M, K, N, dtype, device, args, gen_fn, gen_label):
    x = gen_fn(M, K, dtype, device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    bf = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
    bf.weight.data.copy_(w)
    bp = BPositLinear(weight=w)

    # Reference output in fp32 to score both paths consistently.
    with torch.no_grad():
        ref_fp32 = (x.float() @ w.float().t())
        bf_out = bf(x)
        bp_out = bp(x)

    bf_err = relative_l2(ref_fp32, bf_out)
    bp_err = relative_l2(ref_fp32, bp_out)

    bf_us = time_op(lambda: bf(x), args.n_warmup, args.n_iter)
    bp_us = time_op(lambda: bp(x), args.n_warmup, args.n_iter)

    return {
        "name": name, "surface": surface, "shape": f"{M}x{K}x{N}",
        "input": gen_label,
        "bf_us": bf_us, "bp_us": bp_us,
        "ratio": bp_us / bf_us if bf_us > 0 else float("nan"),
        "bf_err": bf_err, "bp_err": bp_err,
        "err_ratio": bp_err / bf_err if bf_err > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)

    print(f"GPU:    {torch.cuda.get_device_name(device.index or 0)} "
          f"(cap={torch.cuda.get_device_capability(device.index or 0)})")
    print(f"dtype:  {args.dtype}  reps={args.n_iter}  warmup={args.n_warmup}")
    print()

    rows = []
    for gen_label, gen_fn in INPUT_GENS.items():
        print(f"=== input distribution: {gen_label} ===")
        print(f"  {'shape':18s}  {'M×K×N':18s}  "
              f"{'bf_us':>8s}  {'bp_us':>8s}  {'bp/bf':>6s}  "
              f"{'bf_err':>9s}  {'bp_err':>9s}  {'err_ratio':>9s}")
        for shape in SHAPES:
            name, surface, M, K, N = shape
            r = bench_one(name, surface, M, K, N, dtype, device,
                          args, gen_fn, gen_label)
            print(f"  {r['surface']+' '+r['name']:18s}  {r['shape']:18s}  "
                  f"{r['bf_us']:8.1f}  {r['bp_us']:8.1f}  {r['ratio']:6.2f}  "
                  f"{r['bf_err']:9.4f}  {r['bp_err']:9.4f}  "
                  f"{r['err_ratio']:9.2f}")
            rows.append(r)
        print()

    # Aggregate: ATTN vs FFN, per input.
    print("=== summary: mean bposit/bf16 ratios by surface×input ===")
    print(f"  {'input':32s}  {'surface':8s}  {'lat_ratio':>10s}  {'err_ratio':>10s}")
    for gen_label in INPUT_GENS:
        for surface in ("ATTN", "FFN"):
            sub = [r for r in rows if r["input"] == gen_label
                   and r["surface"] == surface]
            if not sub:
                continue
            lat = sum(r["ratio"] for r in sub) / len(sub)
            err = sum(r["err_ratio"] for r in sub) / len(sub)
            print(f"  {gen_label:32s}  {surface:8s}  {lat:10.2f}  {err:10.2f}")


if __name__ == "__main__":
    main()
