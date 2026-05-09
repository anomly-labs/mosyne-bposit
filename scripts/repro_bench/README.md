# repro_bench — bit-exact reproducibility benchmark

This is the MVP for the AI-deployment-side claim of the
`mosyne-bposit` whitepaper §4.2: that a bposit + quire256 inference
path produces **byte-identical** outputs across runs / machines /
drivers / cuBLAS algorithm choices, in a way IEEE-float-on-tensor-cores
fundamentally cannot.

## What the script does

`repro_bench.py` loads a HuggingFace causal LM, optionally swaps FFN
linears for `BPositLinear`, runs a deterministic-decoding generation,
and emits SHA-256 hashes of three quantities per run:

  1. **`hash_logits_pos0`** — next-token logits at the prompt's last
     position (pure forward pass; the quantity most likely to surface
     IEEE reduction-order divergence).
  2. **`hash_logits_full`** — full prompt logits tensor.
  3. **`hash_token_ids`** — autoregressive output tokens.

Plus device / driver / GPU metadata so the JSONL output is
self-describing.

## Usage

```bash
# Single run, hashes to stdout
python repro_bench.py --backend bposit
python repro_bench.py --backend bf16

# Append to a JSONL log
python repro_bench.py --backend bposit --out-jsonl results.jsonl

# Compare across GPUs
CUDA_VISIBLE_DEVICES=0 python repro_bench.py --backend bposit --out-jsonl all.jsonl
CUDA_VISIBLE_DEVICES=1 python repro_bench.py --backend bposit --out-jsonl all.jsonl
# Then check: unique hash_logits_pos0 values across the two device runs.
# bposit should be 1 (identical); bf16 may be 2+ depending on cuBLASLt
# heuristic divergence between the two devices.
```

## What's confirmed (2026-05-09 on this box)

3-trial within-process determinism, same GPU, same prompt:

| Backend | RTX 3090, hash_logits_pos0   | unique across 3 runs |
|---------|------------------------------|----------------------|
| bposit  | `898277f3…`                  | 1 (deterministic)    |
| bf16    | `5c5fdb4a…`                  | 1 (deterministic)    |

Both backends are deterministic within a single process on a single
GPU — that's expected, and **not** the bposit differentiator. cuBLASLt
picks the same algo on each call for the same shape, so the IEEE
non-associativity doesn't surface in this regime.

## What's pending (the actual demo)

The bposit-vs-IEEE differential surfaces when:

  1. **Different GPUs** (different cuBLASLt heuristic algo choices,
     different SM count → different reduction tree) — bposit's
     quire256 accumulation is order-invariant; IEEE float isn't.
  2. **Different cuBLAS / driver versions**.
  3. **Multi-block parallel reductions with `atomicAdd`** (the
     setting whitepaper §4.2's existing 65 K-element-sum experiment
     already isolates).

Case (3) — and case (1) in passing — is **already empirically
demonstrated** by `deploy/5dst_cuda/test_determinism_headline.cu`,
which runs a parallel sum-reduction with both an atomicAdd-FP32 path
and a bposit+quire path. Iter-42 ran it on both GPUs in this box; the
bposit hash matched bit-exactly across the 3090 and 5090 across all
10 runs (5 on each), while the FP32-atomicAdd path produced 9 distinct
hashes across the same 10 runs (one accidental collision; the sets
produced on each GPU are otherwise disjoint).

Full results:
[`docs/research/repro_bench_cross_gpu_2026-05-09.md`](../../docs/research/repro_bench_cross_gpu_2026-05-09.md).

`repro_bench.py` here in this directory is the **LLM-level** version:
hashes logits and token IDs through `BPositLinear`, useful for
end-to-end eval/audit story but doesn't surface the differential on a
single GPU because cuBLASLt picks the same algorithm on each call for
the same shape. The differential will show up here too once the bench
is extended to a parallel-reduction layer or run cross-machine on
hardware that picks different algos.

The scripted output format is already designed to support
post-hoc aggregation: append all runs to a single JSONL, then
`pandas.read_json(lines=True)` and `.groupby('backend')['hash_logits_pos0'].nunique()`.

## Why this matters (the pitch)

Existing AI deployments give "almost-deterministic" outputs — within
a single process, on a single GPU, with a fixed driver. As soon as a
production fleet rolls out a driver update, swaps a GPU SKU, or
serves a model across a heterogeneous cluster, IEEE-float-on-tensor-cores
output bytes start drifting. For most workloads this is fine. For
**eval reproducibility, regulatory audit, mechanistic-interpretability
baselines, and constitutional-AI training reward signals**, drifting
bytes mean answers like "did this model produce that output for this
input on this date" become unanswerable.

`mosyne-bposit` makes the answer a SHA-256 comparison. That's the
deployment-side novelty; this benchmark exists to prove it.
