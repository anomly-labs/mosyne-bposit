# BPositLinear vs bf16 nn.Linear — where the latency gap actually goes

Date: 2026-05-08
Hardware: RTX 3090 (sm_86), CUDA 12.8, PyTorch 2.5.1+cu124

## Motivation

Iteration 9 measured end-to-end Qwen2.5-Coder-1.5B autoregressive
generation: bposit-W8A8 = 100 tok/s, bf16 baseline = 133 tok/s. bposit
is 25% slower than bf16 in the metric customers actually deploy on. The
question: where does the gap go, and which fix has the highest leverage?

## Method

`torch.cuda.Event` per-segment timing of the BPositLinear forward
pipeline at three representative shapes:

| Shape          | M   | K    | N     | Workload                  |
|----------------|-----|------|-------|---------------------------|
| FFN-gate       | 128 | 4096 | 11008 | Llama-class MLP gate-proj |
| Decode-class   | 1   | 4096 | 4096  | Single-token autoreg step |
| Small square   | 32  | 2048 | 2048  | Small batch + small dims  |

Bencheded against PyTorch's native bf16 nn.Linear (the realistic
deployment baseline) on the same shapes.

## Per-segment breakdown

| Stage                   | FFN-gate  | Decode    | Small sq  |
|-------------------------|-----------|-----------|-----------|
| bf16 nn.Linear (total)  | 242 µs    | 51 µs     | 30 µs     |
| bposit input prep       | 18 µs     | 7 µs      | 10 µs     |
| bposit library call     | 333 µs    | 47 µs     | 50 µs     |
| bposit output prep      | 38 µs     | 7 µs      | 10 µs     |
| bposit total            | 389 µs    | 61 µs     | 70 µs     |
| **gap (bposit − bf16)** | **+148**  | **+10**   | **+41**   |
| cast/transpose share    | 56 µs     | 14 µs     | 20 µs     |

**Key finding:** the dominant cost in bposit is the library call (333 µs
at FFN-gate, ~50 µs at small shapes), not the cast/transpose. Eliminating
all cast/transpose overhead would close at most 14% of the FFN-gate gap.

## What's inside the library call

`mosyne_bposit_linear_w8a8` does, in order:
1. Per-token max-abs reduction + INT8 quantize (one fused kernel since
   iteration 5)
2. cuBLASLt INT8 IMMA matmul → INT32 accumulator
3. Per-token × per-channel dequantize back to fp32 output
4. `cudaDeviceSynchronize()` before returning to Python

For FFN-gate shape on a 3090:
- Theoretical bf16 HMMA time: 5.77 GMACs / 71 TFLOPS ≈ 162 µs
- Theoretical INT8 IMMA time: 5.77 GMACs / 142 TOPS ≈ 81 µs
- Measured library call: 333 µs

So the IMMA matmul itself accounts for ~80–150 µs of the 333 µs library
budget. The remaining ~180 µs is the per-token quantize + dequantize
kernels + sync overhead. **That's the optimization target.**

## Targeted Python-side optimization (small win)

Replaced `tensor.to(dtype).t().contiguous()` (2 kernels: cast + memcpy)
with `empty(target_shape, dtype) + copy_(t())` (1 kernel: fused
transpose + cast) on both the input and output prep paths.

| Shape    | Old prep | New prep | Δ      |
|----------|----------|----------|--------|
| FFN-gate | 56 µs    | 40 µs    | -16 µs |
| Small    | 20 µs    | 8 µs     | -12 µs |
| Decode   | 14 µs    | 14 µs    |  ~0    |

Modest win at large shapes; decode is unchanged (the old pattern's
overhead is already at irreducible launch-overhead floor for tiny
tensors). Shipped — not because the savings are dramatic, but because
the new pattern is also clearer in code.

## What would actually close the gap

Ranked by expected impact:

1. **Eliminate `cudaDeviceSynchronize()` inside the library call.**
   Currently the library forces every forward to drain the GPU queue
   before returning. In a transformer with 84 FFN linears per token,
   that's 84 drains per generated token, killing pipelining. PyTorch's
   own ops are async-by-default — bposit should be too. Fix: remove the
   sync, let the caller decide when to synchronize. Risk: requires
   careful audit of all callers (the host-side wrapper relies on the
   subsequent `cudaMemcpy` to drain).

2. **Fuse dequantize into the matmul epilogue.** cuBLASLt's
   `CUBLASLT_MATMUL_DESC_EPILOGUE_*` flags support some fused epilogues
   (bias add, ReLU, etc.) but not the per-token × per-channel dequant
   we need. Options: write a custom CUTLASS template that handles dequant
   in the epilogue, or use `cuBLASLtMatmul`'s `D` output and a separate
   dequant kernel that overlaps with the next layer's quant kernel via
   stream ordering. Estimated saving: ~30–50 µs at FFN-gate.

3. **Convert at library boundary in fp16/bf16, not fp32.** The library
   currently expects fp32 input. The torch wrapper does a bf16→fp32 cast.
   If the library accepted bf16 input directly (small kernel change in
   `k_per_token_quantize_fused`), we save the cast entirely — ~5 µs at
   FFN-gate, ~1 µs at decode.

4. **Persistent cuBLASLt streams.** Currently the library uses the
   default stream (stream 0). For better pipelining with PyTorch, take a
   stream parameter from the caller. Minimal latency change but unblocks
   stream-based concurrency.

Items 1 + 2 together would likely bring bposit to within 10–15% of bf16
at FFN-gate, possibly parity at small shapes. Items 3 + 4 are mop-up.

## Why this matters

The bposit value proposition isn't "faster than bf16" — it's
"reproducible + accurate + memory-efficient at parity-or-near-parity
speed." But "near-parity" needs to be ≤ 10% behind bf16 for customers
to accept the trade. Currently 25% behind end-to-end, which is too much
for routine deployment. Closing items 1 + 2 above makes the pitch
defensible without creative rhetoric.

## Outcome — item 1 shipped

Removed the per-call `cudaDeviceSynchronize()`. Both callers are safe
without it: `mosyne_bposit_linear_w8a8_host` follows with a synchronous
`cudaMemcpy(D2H)` (stream-ordered), and the device-native PyTorch path
is async-by-default (subsequent ops see correct results via stream
ordering). The test that asserts bit-exact agreement between host and
device paths still passes.

| Shape          | bf16 (µs) | bposit iter-10 | bposit iter-11 | Δ      | bp/bf16 iter-11 |
|----------------|-----------|----------------|----------------|--------|-----------------|
| FFN-gate       | 239       | 389            | 340            | -49 µs | 1.42×           |
| Decode         | 47        | 61             | 46             | -15 µs | **0.98×**       |
| Small square   | 24        | 70             | 48             | -22 µs | 2.04×           |

End-to-end Qwen2.5-Coder-1.5B autoregressive generation, same prompt,
deterministic decoding, 128 new tokens × 3 trials, 84 FFN linears
(28 layers × {gate,up,down}_proj) replaced:

| metric              | iter-9   | iter-11    |
|---------------------|----------|------------|
| bf16 baseline       | 133 tok/s| 133.0 tok/s|
| bposit-W8A8 (FFN)   | 100 tok/s| 129.7 tok/s|
| **gap to bf16**     | **−25%** | **−2.4%**  |

Within the ≤10% deployment-acceptance bar. Items 2–4 from the ranked
list above are no longer load-bearing for the deployment story; they
remain candidates if we ever want positive throughput claims rather
than "within noise of bf16."

## Outcome — item 3 shipped

Added a templated `k_per_token_quantize_fused<THREADS, T>` kernel and a
new C entry point `mosyne_bposit_linear_w8a8_bf16` that accepts
column-major bf16 input directly. The torch wrapper now routes bf16
activations to this path, skipping the bf16→fp32 cast at the library
boundary. Output stays fp32 (the output-side cast is its own separate
question).

Per-call medians, RTX 3090 (200 iter, bf16 input):

| Shape          | iter-11 | iter-12 | Δ        |
|----------------|---------|---------|----------|
| FFN-gate       | 340 µs  | 336 µs  | -4 µs    |
| Decode         | 46 µs   | 44 µs   | -2 µs    |
| Small square   | 48 µs   | 45 µs   | -3 µs    |

Per-call savings are smaller than the iter-10 estimate (~5 µs at
FFN-gate) because PyTorch's existing fused cast-during-copy was already
quite efficient — we trade torch's cast for a `__bfloat162float()`
inside the quantize kernel. Net effect: small but real.

End-to-end Qwen2.5-Coder-1.5B autoregressive generation:

| metric              | iter-11    | iter-12    |
|---------------------|------------|------------|
| bf16 baseline       | 133.0 tok/s| 132.8 tok/s|
| bposit-W8A8 (FFN)   | 129.7 tok/s| 131.5 tok/s|
| **gap to bf16**     | **−2.4%**  | **−1.0%**  |

Cumulative effect across 84 layers × 128 decode tokens shows up: per
call we saw ~2 µs at the decode shape, scaled to ≈21 ms over the full
generation, which is ~2% — and that's exactly what the e2e number
moved by. Effectively at parity for deployment.

## Outcome — bf16-native hot path complete (iter-13)

Symmetric to iter-12 but on the output side. Templated `k_dequant<T>`
with specializations for `float` and `__nv_bfloat16`. New entry point
`mosyne_bposit_linear_w8a8_bf16_io` accepts bf16 input and produces
bf16 output directly — no fp32 buffers in the hot path. The torch
wrapper now routes bf16 activations through this fully-bf16 path; the
final `y.copy_(y_cm.t())` becomes a transpose-only kernel (no cast)
since `y_cm.dtype == y.dtype == bf16`.

Per-call medians, RTX 3090 (200 iter, bf16 input + bf16 output):

| Shape          | iter-12 | iter-13 | Δ        |
|----------------|---------|---------|----------|
| FFN-gate       | 336 µs  | 333 µs  | -3 µs    |
| Decode         | 44 µs   | 43 µs   | -1 µs    |
| Small square   | 45 µs   | 43 µs   | -2 µs    |

Decode is now 0.91× of bf16 — bposit is ~9% **faster** than bf16 at
the per-call decode shape.

End-to-end Qwen2.5-Coder-1.5B autoregressive generation:

| metric              | iter-12    | iter-13    |
|---------------------|------------|------------|
| bf16 baseline       | 132.8 tok/s| 132.8 tok/s|
| bposit-W8A8 (FFN)   | 131.5 tok/s| 133.2 tok/s|
| **gap to bf16**     | **−1.0%**  | **+0.3%**  |

Crossed parity. Within run-to-run noise of bf16, with no deficit. The
deployment story is complete: reproducibility + accuracy + memory at
no throughput cost.

## Trajectory across the loop

| iter | change                            | gap to bf16 e2e |
|------|-----------------------------------|-----------------|
| 9    | baseline measurement              | -25.0%          |
| 10   | profile; cast+transpose fusion    | -25%-ish        |
| 11   | drop cudaDeviceSynchronize        | -2.4%           |
| 12   | bf16-direct input                 | -1.0%           |
| 13   | bf16-native output (fully bf16)   | +0.3%           |

Items 2 (CUTLASS dequant epilogue) and 4 (caller-controlled streams)
remain on the ranked list as candidates for *positive* throughput
claims — push past parity into "1.1× faster than bf16 end-to-end"
territory — but they're no longer load-bearing for deployment.

## Reproduction

- Per-call shape table: `packages/mosyne_bposit/examples/per_shape_bench.py`
- End-to-end Qwen autoregressive: `packages/mosyne_bposit/examples/qwen_generate_bench.py`
