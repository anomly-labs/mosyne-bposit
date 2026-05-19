"""mosyne-bposit cross-GPU bit-exactness probe.

The whitepaper claims bit-exact reproducibility of bposit-W8A8 output
across NVIDIA GPU architectures. We have empirically verified this on
the RTX 3090 (Ampere sm_86) and RTX 5090 (Blackwell sm_120). The
structural argument — integer addition is associative regardless of
reduction order — applies to any NVIDIA GPU running the same INT8
IMMA path, but the readme correctly flags Hopper / Ada / Turing /
Volta as "structurally expected, empirically still to confirm".

This script closes that gap on demand. Run on each new architecture
(typically via a Lambda Labs / RunPod cloud rental, ~$2-5/hr,
~5 minutes wall-clock) and it produces a JSON record with:

  * the GPU name + compute capability + driver / CUDA / torch versions
  * a SHA-256 hash for each of 3 canonical matmul shapes (small /
    Llama-FFN-gate / square)

To verify cross-architecture bit-exactness:

  # Once, on each "canonical" GPU (3090, 5090, etc.):
  $ mosyne-bposit-probe --json > probe_3090.json
  $ mosyne-bposit-probe --json > probe_5090.json

  # On a new architecture (H100, A100, 4090, ...):
  $ mosyne-bposit-probe --json --compare-with probe_3090.json

  exits 0 with "PASS: all 3 shapes bit-identical to probe_3090.json"
  exits 1 with a per-shape diff if any hash differs

Designed to need no model downloads, no calibration set, no
network. Only inputs: a CUDA-capable GPU and a built
libmosyne_bposit.so (see mosyne-bposit-build).

The inputs are generated on CPU with an explicit Generator(device=
'cpu') so the random tensors themselves are bit-identical across
architectures — only the bposit math is what we are testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass


# Canonical shapes the probe runs against. Spans the regimes that
# matter: small (sanity), Llama FFN-gate (the headline production
# shape), and a square that exercises the symmetric-K case.
# Keys are short, stable strings used as dict keys in the JSON output.
CANONICAL_SHAPES: dict[str, tuple[int, int, int]] = {
    "small":     (64, 2048, 2048),
    "llama_ffn": (128, 4096, 11008),
    "square":    (256, 4096, 4096),
}
CANONICAL_SEED = 1234


@dataclass
class ProbeRecord:
    """One probe run: GPU + environment + the hashes."""
    gpu_name: str
    compute_capability: str
    cuda_version: str | None
    pytorch_version: str
    seed: int
    hashes: dict[str, str]   # shape_key -> sha256[:16]


def _hash_tensor(t) -> str:
    """SHA-256 of the raw byte content of the tensor, on CPU."""
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def probe(seed: int = CANONICAL_SEED,
          shapes: dict[str, tuple[int, int, int]] | None = None) -> ProbeRecord:
    """Run the canonical probe on the current GPU and return a record.

    Inputs are generated on CPU with a fixed seed so the tensors
    themselves are bit-identical across architectures — only the
    BPositLinear forward is being measured."""
    import torch
    from mosyne_bposit.torch_compat import BPositLinear

    if shapes is None:
        shapes = CANONICAL_SHAPES

    hashes: dict[str, str] = {}
    for key, (M, K, N) in shapes.items():
        # Generate on CPU with an explicit generator — guarantees the
        # source tensors are byte-identical across machines. CUDA RNG
        # is NOT guaranteed cross-architecture deterministic.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_cpu = torch.randn(M, K, generator=gen, dtype=torch.float32) * 0.3
        w_cpu = torch.randn(N, K, generator=gen, dtype=torch.float32) * (1.0 / K ** 0.5)
        x = x_cpu.cuda()
        w = w_cpu.cuda()
        bposit = BPositLinear(weight=w)
        y = bposit(x)
        hashes[key] = _hash_tensor(y)

    major, minor = torch.cuda.get_device_capability()
    return ProbeRecord(
        gpu_name=torch.cuda.get_device_name(0),
        compute_capability=f"sm_{major}{minor}",
        cuda_version=getattr(torch.version, "cuda", None),
        pytorch_version=torch.__version__,
        seed=seed,
        hashes=hashes,
    )


def compare(current: ProbeRecord, reference: dict) -> tuple[bool, list[str]]:
    """Compare a fresh probe run against a saved reference JSON.

    Returns (all_pass, lines) — `lines` is human-readable per-shape
    diff output suitable for printing whether or not all passed.
    The reference is the dict loaded from a prior probe's --json
    output (or any dict with a 'hashes' sub-dict)."""
    ref_hashes: dict[str, str] = reference.get("hashes", {})
    lines: list[str] = []
    all_pass = True
    keys = sorted(set(current.hashes) | set(ref_hashes))
    for k in keys:
        cur = current.hashes.get(k)
        ref = ref_hashes.get(k)
        if cur is None:
            lines.append(f"  [{k:<10s}]  MISSING in current run "
                         f"(reference: {ref[:16]}...)")
            all_pass = False
        elif ref is None:
            lines.append(f"  [{k:<10s}]  MISSING in reference "
                         f"(current: {cur[:16]}...)")
            all_pass = False
        elif cur == ref:
            lines.append(f"  [{k:<10s}]  PASS   {cur[:16]}")
        else:
            lines.append(f"  [{k:<10s}]  FAIL   "
                         f"current={cur[:16]}  reference={ref[:16]}")
            all_pass = False
    return all_pass, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mosyne-bposit-probe",
        description="Cross-GPU bit-exactness probe for bposit-W8A8. "
                    "Run on each new GPU architecture; compare hashes "
                    "to verify the cross-arch reproducibility claim "
                    "empirically.",
    )
    ap.add_argument("--json", action="store_true",
                    help="emit probe record as JSON on stdout "
                         "(human narration on stderr)")
    ap.add_argument("--compare-with", type=str, default=None,
                    help="path to a saved probe JSON to compare "
                         "against. Exits 0 on full bit-match, 1 on "
                         "any per-shape mismatch.")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED,
                    help="RNG seed for the input tensors "
                         f"(default: {CANONICAL_SEED}; only change "
                         "if you have a documented reason)")
    args = ap.parse_args(argv)

    # iter-299: pre-flight GPU/PyTorch compat check — gives an
    # actionable install hint when PyTorch is too old for the
    # active GPU (e.g. PyTorch 2.5 on a Blackwell 5090).
    from mosyne_bposit._gpu_compat import check_gpu_compat
    rc = check_gpu_compat()
    if rc != 0:
        return rc
    import torch

    out = sys.stderr if args.json else sys.stdout
    print("mosyne-bposit cross-GPU probe", file=out)
    print(f"GPU:        {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_capability()})", file=out)
    print(f"PyTorch:    {torch.__version__}", file=out)
    print(f"CUDA:       {getattr(torch.version, 'cuda', '?')}", file=out)
    print(f"Seed:       {args.seed}", file=out)
    print(f"Shapes:     {len(CANONICAL_SHAPES)} canonical "
          f"({', '.join(CANONICAL_SHAPES)})", file=out)
    print(file=out)

    record = probe(seed=args.seed)

    for key, h in record.hashes.items():
        M, K, N = CANONICAL_SHAPES[key]
        print(f"  {key:<10s} ({M}x{K}x{N})  sha256 = {h[:16]}...", file=out)
    print(file=out)

    rc = 0
    if args.compare_with:
        try:
            with open(args.compare_with) as f:
                reference = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not read --compare-with file "
                  f"{args.compare_with}: {exc}", file=sys.stderr)
            return 1
        all_pass, lines = compare(record, reference)
        ref_gpu = reference.get("gpu_name", "<unknown>")
        print(f"Comparing this GPU against reference probe from "
              f"{ref_gpu}:", file=out)
        for line in lines:
            print(line, file=out)
        print(file=out)
        if all_pass:
            print(f"  → PASS: all {len(record.hashes)} shapes "
                  f"bit-identical to {args.compare_with}", file=out)
        else:
            print("  → FAIL: cross-GPU bit-exactness claim does "
                  "not hold for this hardware against the reference. "
                  "Investigate before publishing.", file=out)
            rc = 1

    if args.json:
        print(json.dumps(asdict(record), indent=2))

    return rc


if __name__ == "__main__":
    sys.exit(main())
