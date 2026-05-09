# repro-bench-min — 60-second self-contained demo

This is the smallest possible artifact that demonstrates the
mosyne-bposit reproducibility property: **bposit + quire256
sum-reduction produces byte-identical output across runs and across
NVIDIA GPU generations; IEEE FP32 + atomicAdd does not.**

No PyTorch, no model load, no Python at all once built. Just the
synthetic 65 K-element log-uniform sum from whitepaper §4.2,
compiled to native sm_86 + sm_120 and run on every NVIDIA GPU on the
host.

## Usage

```bash
# clone the repo, then:
cd scripts/repro_bench/min_demo
make demo
```

Requires `nvcc 12.x` on PATH and at least one NVIDIA GPU.

## Sample output (this box, RTX 3090 + RTX 5090)

```
GPU  Name                       bposit (5 runs)         FP32 atomicAdd (5 runs, unique)
---  ----                       ----------------        -------------------------------
0    NVIDIA GeForce RTX 5090    bposit16=0x7627         bits=0x4a93ec2e … 0x4a93ec70  (5 unique)
1    NVIDIA GeForce RTX 3090    bposit16=0x7627         bits=0x4a93ec40 … 0x4a93ec4a  (5 unique)

✓ bposit produced the SAME hash on all 2 GPU(s) —
  bit-exact reproducibility across hardware generations.
  IEEE FP32 atomicAdd produced 9 distinct hashes across
  10 runs total (5 per GPU × 2 GPUs).
```

## What the demo proves

The demo script (`demo.sh`) builds and runs
`deploy/5dst_cuda/test_determinism_headline.cu` on every NVIDIA GPU
on the host, captures the bposit16 hash and the 5 FP32-atomicAdd
hashes from each, then prints a comparison table.

The kernel under test sums **65,536 log-uniform values across six
decades** (1e-3 to 1e3). Wide scale spread is the standard trigger
for IEEE non-associativity — small values get absorbed by larger
running totals depending on the order they're added. CUDA's
`atomicAdd` schedules concurrent adds in non-deterministic order, so
the final FP32 bit pattern depends on scheduler timing.

The bposit path converts each value to a 256-bit quire summand and
accumulates via warp-shuffle reduction (associative integer add) +
shared-memory partials. Final encode quire→bposit16 is a deterministic
function. Same input bits → same output bits, every run, every GPU.

## What it does NOT prove (yet)

This demo proves cross-GPU and within-process determinism for a
reduction kernel. It does not yet prove the same property at the
LLM forward pass level. `repro_bench.py` in the parent directory
hashes LLM logits + token IDs but won't surface the same differential
on a single GPU because cuBLASLt's INT8 IMMA path picks the same
algorithm on each call for the same shape — local determinism comes
"for free" there.

The differential will show up in `repro_bench.py` when:
- Run cross-GPU (different SM count, different reduction tree)
- Run cross-driver / cross-CUDA-version
- Run in workloads that hit cuBLAS's split-K reduction or attention
  layers with parallel softmax reductions

Future work: extend `repro_bench.py` to drive a parallel-reduction
layer specifically, so the LLM-level bench surfaces the same
differential the synthetic sum already does.

## Why this matters

For any production AI deployment that needs to answer "did this model
produce that output for that input on that date" as a hash check —
eval reproducibility, mechanistic interpretability baselines,
constitutional-AI training reward signals, regulatory audit, fleet-wide
A/B tests on heterogeneous hardware — IEEE float on tensor cores
cannot deliver this. bposit + quire256 can. This demo is the proof.

The full whitepaper at
`docs/whitepaper/mosyne_bposit_whitepaper.pdf` (§4.2 specifically)
backs the same claim with mean ± stddev across 8 trials. The demo
here reproduces it interactively in 60 seconds with a public-readable
output table.
