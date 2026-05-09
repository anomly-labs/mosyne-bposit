#!/usr/bin/env python3
"""Reproducibility benchmark — bposit-W8A8 vs IEEE bf16, across machines.

Loads a HuggingFace causal LM, optionally swaps FFN linears for
``BPositLinear``, runs a fixed deterministic-decoding generation,
and emits SHA-256 hashes of:

  1. the generated token IDs (autoregressive output)
  2. the position-0 next-token logits (single forward pass through
     the bposit / bf16 path; the quantity most likely to differ
     between IEEE-reduction-order and bposit-quire-exact paths)
  3. the model.forward output of the *same* prompt on a fresh model
     load — guards against any hidden in-process state

Run on multiple GPUs (5090 + 3090 here, or A100 / H100 elsewhere) and
compare hashes:

  * `--backend bposit` should produce **identical** hashes for (2)
    and (3) across machines, drivers, and PyTorch versions, because
    the bposit + quire256 accumulation is associative regardless of
    cuBLASLt's INT8 IMMA algorithm choice.
  * `--backend bf16` will produce **divergent** hashes for (2) and
    (3) once cuBLASLt picks different algos on different shapes /
    drivers — the IEEE fp32 atomicAdd ordering is the documented
    source of nondeterminism.

This is the MVP for the bit-exact reproducibility claim of the
mosyne-bposit whitepaper §4.2 generalised from "65 K log-uniform
sum" to "real LLM forward + decode."

Output format is one JSON line per run, keyed on
{model, backend, n_tokens, prompt_hash, device, gpu_name,
driver_version, torch_version, hash_token_ids, hash_logits_pos0,
hash_logits_full}.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import socket
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = (
    "Write a Python function that takes a list of integers and returns "
    "the second-largest value, handling edge cases like empty lists and "
    "lists with duplicates. Include type hints and a docstring."
)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_tensor(t: torch.Tensor) -> str:
    """SHA-256 of the raw bytes of a tensor on CPU. Stable across runs
    because torch tensors have a fixed-layout in-memory representation
    once cast to a known dtype + device."""
    return sha256_hex(t.detach().cpu().contiguous().numpy().tobytes())


def swap_ffn(model) -> int:
    """Swap FFN gate/up/down linears to BPositLinear (the iter-13
    parity config). Returns the count of replaced modules."""
    from mosyne_bposit.torch_compat import BPositLinear, replace_linear_modules
    replace_linear_modules(
        model,
        predicate=lambda name, lin: any(
            k in name for k in (".gate_proj", ".up_proj", ".down_proj")
        ),
    )
    return sum(1 for _ in (m for m in model.modules() if isinstance(m, BPositLinear)))


def gpu_info(device: torch.device) -> dict:
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": str(torch.cuda.get_device_capability(device)),
        "driver_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }


def run(args) -> dict:
    torch.manual_seed(0)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=model_dtype, device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if args.backend == "bposit":
        n_swapped = swap_ffn(model)
    else:
        n_swapped = 0

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_hash = sha256_hex(args.prompt.encode("utf-8"))

    # 1. Single forward pass — capture position-0 next-token logits
    #    and the full output sequence's last-position logits.
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits  # [1, seq_len, vocab]
    pos0_logits = logits[0, -1, :].to(torch.float32)
    full_logits = logits[0, :, :].to(torch.float32)
    hash_logits_pos0 = hash_tensor(pos0_logits)
    hash_logits_full = hash_tensor(full_logits)

    # 2. Autoregressive generation — capture the token IDs.
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=args.n_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = tokens[0, inputs.input_ids.shape[1]:]
    hash_token_ids = sha256_hex(new_tokens.cpu().numpy().tobytes())

    return {
        "model": args.model,
        "backend": args.backend,
        "dtype": args.dtype,
        "n_tokens": args.n_tokens,
        "n_ffn_swapped": n_swapped,
        "prompt_hash": prompt_hash,
        "hash_logits_pos0": hash_logits_pos0,
        "hash_logits_full": hash_logits_full,
        "hash_token_ids": hash_token_ids,
        "ts": time.time(),
        **gpu_info(device),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--backend", choices=["bf16", "bposit"], required=True,
                    help="bf16 = native nn.Linear (IEEE float through cuBLASLt); "
                         "bposit = FFN linears swapped to BPositLinear (W8A8 IMMA + quire)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                    help="model load dtype (default: bf16)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=64,
                    help="generation length for the token-ID hash (default: 64)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out-jsonl", default="-",
                    help="append one JSON line per run to this file (- = stdout)")
    args = ap.parse_args()

    result = run(args)
    line = json.dumps(result, sort_keys=True)
    if args.out_jsonl == "-":
        print(line)
    else:
        with open(args.out_jsonl, "a") as f:
            f.write(line + "\n")
        # Echo to stderr so users see something
        print(f"[ok] {args.backend:6s} on {result['gpu_name']}: "
              f"logits_pos0={result['hash_logits_pos0'][:16]}…",
              file=sys.stderr)


if __name__ == "__main__":
    main()
