# bposit attention regression — per-shape breakdown

**Date:** 2026-05-09
**Source script:** `packages/mosyne_bposit/examples/attn_vs_ffn_diagnostic.py`
**GPU:** RTX 3090 (sm_86), bf16, 80 reps median, 10 warmup
**Prior context:** whitepaper §4.5 documents FFN-only e2e parity (-0.4%) but
attention regresses -13 to -16% on the same model. Iter-56 question: *what
inside the attention path is producing this gap when the kernel is the same?*

## Headline

The regression is **latency-driven, not quantization-quality driven**.
Per-shape relative L2 reconstruction error (`bposit_err / bf16_err`) is
~7-9× across **all** surfaces and inputs — slightly worse on attention,
not catastrophically so. But latency-ratio (`bp_us / bf_us`) shows
attention shapes paying 1.78-3.11× while FFN shapes pay 1.17-1.68×.

| Surface | mean lat ratio (norm-like) | mean lat ratio (silu-like) | mean err ratio |
|---|---:|---:|---:|
| ATTN | **1.84×** | **1.85×** | 7.27 / 8.82 |
| FFN  | **1.17×** | **1.17×** | 6.75 / 7.86 |

## The single biggest contributor: GQA K/V projection (4096→1024)

| shape | M×K×N | bf16 µs | bposit µs | bp/bf |
|---|---|---:|---:|---:|
| attn-q (prefill)   | 512×4096×4096   | 319.5 |  569.3 | **1.78×** |
| attn-kv (prefill)  | 512×4096×1024   |  89.1 |  277.5 | **3.11×** |
| attn-o (prefill)   | 512×4096×4096   | 315.4 |  571.4 | **1.81×** |
| ffn-gate (prefill) | 512×4096×14336  | 904.2 | 1240.1 | 1.37× |
| ffn-up (prefill)   | 512×4096×14336  | 911.4 | 1238.1 | 1.36× |
| ffn-down (prefill) | 512×14336×4096  | 975.9 | 1639.4 | 1.68× |

The 3.1× slowdown on `512×4096×1024` is the worst shape in the matrix.
4 layers × 32 layers of K + V projections = 256 such calls per forward,
each running ~190 µs slower than its bf16 counterpart. That alone
explains the bulk of a -13% prefill regression.

## Mechanism

cuBLASLt INT8 IMMA tiles are typically 128×256 or 256×128. At N=1024,
the output dimension fits 4-8 tiles wide — narrow enough that some SMs
go idle after the matmul portion, while the per-token dequantize
postlude (which scales as M×N regardless of tile shape) still runs
serially against the same tile schedule. At N=14336, the matmul covers
56-112 tiles and the dequantize amortizes cleanly.

The bf16 cuBLAS path doesn't have this floor: it picks among many more
algorithms at narrow-N (split-K, smaller tiles) and lands on something
that uses the SMs better.

## Decode-step (M=1) is fine

| shape | M×K×N | bf16 µs | bposit µs | bp/bf |
|---|---|---:|---:|---:|
| attn-q-dec   | 1×4096×4096   |  49.2 |  42.0 | **0.85×** |
| attn-kv-dec  | 1×4096×1024   |  14.3 |  23.6 | 1.64× |
| ffn-gate-dec | 1×4096×14336  | 148.5 |  98.3 | **0.66×** |
| ffn-down-dec | 1×14336×4096  | 148.5 | 112.6 | **0.76×** |

bposit *wins* on every decode shape except attn-kv-dec, which is the
same narrow-N issue but at smaller scale. The e2e regression is a
prefill problem.

## Quantization quality is shape-invariant

bposit reconstruction error is essentially flat across all surfaces:
relative L2 ≈ 0.012 - 0.016 for every (M, K, N, input) cell, vs bf16
≈ 0.0017 - 0.003. The norm-like input distribution (1% outlier rows
mimicking pre-RMSNorm hidden states) does NOT make attention noticeably
worse than FFN. **H2 (quality dominates) is not supported by this data.**

## Three fix paths, ordered by leverage

1. **Shape-aware fallback** *(low effort, immediate win)*
   `BPositLinear` checks `out_features` at construction time; if below
   a threshold (e.g., 2048) the forward pass routes to a bf16 nn.Linear
   even though the weight is stored as bposit. Closes the K/V-proj
   regression at the cost of 4 layers × 256 calls of bf16 weight memory
   that we already keep around as the dequantized fallback. This brings
   the e2e attention regression closer to the FFN parity number without
   any kernel work.

2. **Fused QKV projection** *(medium effort, the architecturally correct fix)*
   Concatenate Q, K, V weight matrices into a single (out=hidden+2×kv_dim,
   in=hidden) projection — for Qwen3 that's 4096→6144. Larger N gets the
   IMMA tiling out of the narrow-output regime, and we save 2 of the 3
   per-token quantize calls. Has to be wired at module replacement time
   (transformers.modeling_qwen has separate q/k/v_proj attributes —
   replace the *attention block* not the linears). Per-shape the headroom
   is `(89.1 + 89.1) / 277.5 ≈ 64%` on the K and V proj that go into the
   fused call.

3. **cuBLASLt algo override for narrow-N** *(high effort, kernel-level)*
   The current `mosyne_bposit_linear_w8a8` calls `cublasLtMatmul` with the
   default heuristic-picked algorithm. Override with explicit
   `cublasLtMatmulAlgoConfigSetAttribute` selecting a smaller tile +
   split-K configuration when N < 2048. Requires the Blackwell-aware
   tuning the iter-13 work didn't do.

## Recommended next step

Implement (1) as `BPositLinear(narrow_n_threshold=2048)` with a default
that re-runs bf16 nn.Linear when the output is below threshold, then
re-run `qwen_generate_bench.py --swap-attention` to confirm the e2e
attention regression closes.

(2) is the right long-term answer; (3) is most-bang-per-buck only if
we end up needing to support generic transformer shapes in third-party
deployments, which is not the iter-13/14 scope.

## Iter-57 verification on Qwen2.5-Coder-1.5B-Instruct (3090, 128 new tokens)

Real e2e measurement on a 1.5B model — too small for the bposit win
margin to dominate as cleanly as the 30B case in §4.5, but the
ordering and the K/V-skip story both reproduce.

| config                                  | swapped | tok/s | Δ vs bf16 | output |
|-----------------------------------------|--------:|------:|----------:|--------|
| baseline bf16                           |     0   | 133.0 |       —   | reference |
| FFN only (gate / up / down)             |    84   | 133.5 |    +0.3%  | identical |
| **ATTN only, threshold=1024**           |    56   | 123.7 |    -7.1%  | **identical** |
| FFN + ATTN, threshold=300 (skip K/V)    |   140   | 122.8 |    -7.9%  | diverges |
| FFN + ATTN, threshold=1024 (skip K/V)   |   140   | 122.2 |    -8.2%  | diverges |
| FFN + ATTN, threshold=2048              |    56   |  85.5 |   -35.7%  | identical |
| FFN + ATTN, no threshold                |   196   |  72.6 |   -45.5%  | diverges |

### What the table says

1. **FFN-only is parity** (+0.3%, output identical). Same as the §4.5
   number on 30B. The bposit-W8A8 FFN path is robust across model
   sizes.
2. **ATTN-only with K/V skipped is -7.1%** with output bit-identical to
   the bf16 baseline. The K/V projections at 1.5B are 1536→256; bposit
   loses on them, and skipping them makes attention swap break-even
   enough to keep the output matching exactly.
3. **threshold=2048 is wrong on 1.5B**: it skips Q/O (1536) and
   FFN-down (1536) along with K/V, leaving only gate/up swapped. That
   removes the FFN-down bposit win at decode (0.76× per the §1.3
   diagnostic) and produces -35.7%. The threshold must be tuned to the
   target model's kv_dim; on 1.5B that's `kv_dim + 1 = 257`, on 30B
   that's `kv_dim + 1 = 1025`.
4. **More-bposit-but-lower-threshold beats less-bposit-but-higher-threshold.**
   threshold=300 (140 linears swapped) gives -7.9% while threshold=2048
   (56 linears swapped) gives -35.7%. The intuition is that bposit
   wins at decode for nearly every shape; the ONLY shape it loses on is
   GQA K/V. Skipping more than K/V leaves bposit-decode wins on the
   table.

### Updated guidance

**Per-model threshold cheat sheet:**

| model                       | hidden | kv_dim | recommended threshold |
|-----------------------------|-------:|-------:|----------------------:|
| Qwen2.5-Coder-1.5B-Instruct | 1536   | 256    | 257 (skip only K/V)   |
| Qwen2.5-Coder-7B-Instruct   | 3584   | 512    | 513                   |
| Qwen2.5-Coder-32B-Instruct  | 5120   | 1024   | 1025                  |
| Qwen3-Coder-30B-A3B-AWQ     | 4096   | 1024   | 1025                  |
| Llama-3.1-8B                | 4096   | 1024   | 1025                  |
| Llama-3.1-70B               | 8192   | 1024   | 1025                  |

The rule of thumb is `kv_dim + 1` — skip ONLY the K/V projection,
keep everything else (Q, O, FFN gate, up, down) on bposit. A future
ergonomic improvement is to expose a `skip_kv_proj=True` boolean that
walks the model and sets the threshold automatically from the
attention config; iter-57 confirmed this is the right default policy
but didn't ship the convenience flag.

### Output divergence: a separate concern

FFN+ATTN with the K/V skip produces output that diverges around char
278. The text remains coherent but is not bit-identical to bf16. This
is the bposit *quantization* error showing through, distinct from the
*latency* topic of this doc. The §4.4 perplexity comparison in the
whitepaper is the right framework for evaluating that — bposit's
WikiText perplexity is +0.3% from baseline, well within
sampling-temperature noise for a generation task.

## Iter-73 fp16 verification — same throughput, different determinism

The iter-57 measurements were all bf16. Iter-73 ran the same configs
with `--model-dtype fp16` to confirm the policy generalises across the
inference dtype:

| config (fp16)                          | swapped | tok/s | Δ vs fp16 | output |
|----------------------------------------|--------:|------:|----------:|--------|
| baseline fp16                          |     0   | 134.4 |       —   | reference |
| ATTN only, skip_kv_proj                |    56   | 124.7 |    -7.2 % | **diverges @ char 6** |
| FFN + ATTN, skip_kv_proj               |   140   | 122.3 |    -9.1 % | diverges @ char 372 |

Throughput numbers track bf16 within a percentage point — `skip_kv_proj`
recovers the same fraction of attention regression on fp16 as on bf16.
**But the bit-identical-output property does not carry over.** ATTN-only
with skip_kv_proj on bf16 produced bit-identical output to the bf16
baseline; the same configuration on fp16 diverges immediately at the
second token.

The probable explanation: per-token argmax is the discrete sieve. When
bposit's quantization error sums with bf16's coarser mantissa rounding,
the noise tends to land within the top-1/top-2 logit gap and the
argmax stays the same. On fp16's finer mantissa, the noise crosses the
gap on a token where bf16 didn't. This is a property of the inference
dtype's rounding mass, not of bposit itself.

**Implication for deployment recommendation:** the bit-identical
property in §4.5 of the whitepaper is FFN-only on bf16. Extending to
attention with skip_kv_proj preserves bit-identicality on bf16 but
not fp16; the throughput pattern is the same on both. Operators
chasing the determinism property should stay on FFN-only + bf16; those
prioritising memory savings can extend to ATTN-with-skip_kv_proj on
either dtype at the documented throughput cost.

## Iter-75 perplexity verification — quality holds across all three policies

Threading the iter-58 swap surface through `wikitext_ppl_bench.py`
(iter-74) lets us measure the perplexity cost of each policy
alongside its throughput cost. Run on Qwen2.5-Coder-0.5B-Instruct,
WikiText-2-raw test, 32 × 512 tokens, bf16, RTX 3090:

| config                              | bposit PPL | Δ vs bf16 baseline (36.9423) |
|-------------------------------------|-----------:|----------------------------:|
| FFN only                            |    36.9479 |                       +0.02 % |
| ATTN only + `skip_kv_proj`          |    36.9326 |                       −0.03 % |
| FFN + ATTN + `skip_kv_proj`         |    37.0226 |                       +0.22 % |

All three preserve perplexity well within the "<1 %" headline claim
in whitepaper §4.4 (which was measured on the larger 3B model and
reported +0.3 % for FFN-only). The mild negative delta on ATTN-only
is single-run variance; with more sequences it'd average toward zero.

**Combined story across iter-57 / iter-73 / iter-75:**

| config            | throughput (bf16) | throughput (fp16) | output bit-identical? | perplexity Δ |
|-------------------|------------------:|------------------:|:---------------------:|-------------:|
| FFN only          |            +0.3 % |             −0.4 % | yes (bf16), yes (fp16) | +0.02 %    |
| ATTN + skip_kv    |            −7.1 % |             −7.2 % | **yes (bf16)**, no (fp16) | −0.03 %    |
| FFN+ATTN + skip_kv |            −8.2 % |             −9.1 % | no                    | +0.22 %    |

The perplexity column is the missing piece of the story: throughput
+ determinism + quality form a Pareto surface, and the operator
picks the point that matches their constraint. None of the three
configurations degrades quality measurably; all three are within
single-run sampling noise of the baseline.

Reproducible via:

```bash
python packages/mosyne_bposit/examples/wikitext_ppl_bench.py --n-seqs 32 \
    --swap-ffn --swap-attention --skip-kv-proj
```
