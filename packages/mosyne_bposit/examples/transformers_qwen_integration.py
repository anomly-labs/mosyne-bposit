#!/usr/bin/env python3
"""Replace selected nn.Linear layers in a Hugging Face Qwen2.5 model with
mosyne-bposit W8A8 versions, then run inference and compare output to
the unmodified model.

Usage::

    pip install mosyne-bposit[torch] transformers
    mosyne-bposit-build               # one-time, requires nvcc on PATH
    python examples/transformers_qwen_integration.py \\
            --model Qwen/Qwen2.5-Coder-3B-Instruct \\
            --layers 10,11,12 \\
            --prompt "def quicksort(arr):"

The script loads the model in BF16, materialises a baseline output for a
short prompt, then replaces the gate / up / down projections in the
specified transformer layers with BPositLinear modules and runs the
same prompt through the modified model.  It reports per-token logit L2
relative error so you can see how W8A8 quantisation affects the layer
outputs in a real model.

Caveat: the current host-side ctypes path round-trips activations through
CPU, so end-to-end latency will be much higher than a pure-GPU
implementation.  The accuracy numbers (logit L2 error) are still
representative.
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"missing dependency: {e}\n"
          f"install with: pip install transformers torch", file=sys.stderr)
    raise SystemExit(1)

from mosyne_bposit.torch_compat import replace_linear_modules


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct",
                    help="HF model id or local path")
    ap.add_argument("--layers", default="10",
                    help="Comma-separated transformer layer indices to quantise (e.g. '10,11')")
    ap.add_argument("--prompt", default="def quicksort(arr):",
                    help="Prompt string for the comparison forward pass")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    target_layers = {int(s) for s in args.layers.split(",") if s.strip()}
    print(f"loading {args.model} on {args.device} (BF16) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    inputs = tok(args.prompt, return_tensors="pt").to(args.device)

    # Baseline forward
    with torch.no_grad():
        t0 = time.perf_counter()
        out_baseline = model(**inputs)
        t_base = time.perf_counter() - t0
    logits_baseline = out_baseline.logits.detach()

    # Save baseline state, swap in BPositLinear in selected MLP modules
    print(f"replacing nn.Linear in MLP modules of layers {sorted(target_layers)} with BPositLinear ...")

    def keep(name: str, _mod):
        if ".mlp." not in name:
            return False
        try:
            li = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return False
        return li in target_layers

    n_replaced = 0
    for name, _mod in model.named_modules():
        if keep(name, _mod):
            n_replaced += 1
    replace_linear_modules(model, predicate=keep)

    # Quantised forward
    with torch.no_grad():
        t0 = time.perf_counter()
        out_q = model(**inputs)
        t_q = time.perf_counter() - t0
    logits_q = out_q.logits.detach()

    diff = (logits_q.float() - logits_baseline.float()).pow(2).sum().sqrt()
    ref  = logits_baseline.float().pow(2).sum().sqrt()
    relerr = (diff / ref).item()

    print()
    print(f"replaced {n_replaced} nn.Linear modules across MLP blocks of layers {sorted(target_layers)}")
    print(f"baseline forward:           {t_base*1000:7.1f} ms")
    print(f"bposit-IMMA forward (host): {t_q*1000:7.1f} ms")
    print(f"logit L2 relative error:    {relerr:.4e}")
    print()
    if relerr < 0.05:
        print("✓ within 5% L2 — quantised model is functional for this layer subset.")
    else:
        print("⚠  L2 error above 5% — investigate per-channel calibration on this model.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
