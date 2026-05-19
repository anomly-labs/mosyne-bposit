# Contributing to mosyne-bposit

Thanks for the interest. Below is what would genuinely help us
right now, in rough priority order. The list is honest: some
items are easy short-effort contributions (#1), others are
multi-week research projects (#5). We are not looking to grow
the contributor base for its own sake — we'd rather have five
high-leverage contributions than fifty drive-by ones.

## Where help would land hardest

### 1. Cross-GPU bit-exactness verification on hardware we don't have

**Effort:** 5 minutes of GPU time + sending us a JSON.
**Why it matters:** The headline reproducibility claim is verified
empirically on RTX 3090 (Ampere) and RTX 5090 (Blackwell). The
structural argument — integer addition is associative; the INT8
IMMA path is identical across NVIDIA arches — applies to every
NVIDIA GPU from Volta forward, but we have no empirical coverage
on H100 (Hopper), A100 (Ampere datacentre), 4090 (Ada), L40,
T4 (Turing), or Volta. Closing that gap moves the README claim
from "structurally expected, verified on Ampere + Blackwell" to
"empirically verified across the modern NVIDIA fleet."

**What to do:**
```bash
pip install mosyne-bposit[torch]
mosyne-bposit-build                          # one-time .so build
mosyne-bposit-probe --json > probe_yourgpu.json
```

Then either:
- Open an issue titled `probe: <GPU model>` and paste the JSON, or
- Email it to ry@anomly.com

We'll add your card to the verified-bit-exact list in the README,
credit you in the next release, and tag you in the launch post
(unless you'd rather stay anonymous, in which case just say so).

If your probe output *doesn't* match the canonical 3090/5090
hashes, that's actually the most interesting result — it means
we've found a real bug. Please file the issue regardless.

### 2. FP8 head-to-head benchmark on H100 / H200

**Effort:** 2–4 hours of H100 time + writing up the results.
**Why it matters:** The whitepaper benchmarks against BF16 (the
realistic deployment baseline) rigorously. We have not benchmarked
against FP8 (E4M3 / E5M2) because we don't have H100/H200 hardware
in-house. FP8 is fast; we're not claiming faster than FP8 — but
the *reproducibility* property is something we believe FP8
structurally cannot deliver, and a clean apples-to-apples
comparison is one of the highest-impact things anyone with H100
access could run.

**What to compare:** at the Llama FFN-gate shape
(M=128, K=4096, N=11008):
- BF16 HMMA throughput (baseline)
- FP8 E4M3 throughput + bit-exactness across 5 repeated runs
- bposit-IMMA throughput + bit-exactness across 5 repeated runs

We'd publish the comparison in the whitepaper, fully credited.
Open an issue first so we can sync on protocol — we want the
bench to be airtight before it ships.

### 3. Verified HuggingFace integration recipes for additional model families

**Effort:** ~30 min per model family — install, swap layers,
run `wikitext_ppl_bench.py`, file an issue or PR with the numbers.
**Why it matters:** The whitepaper validates the W8A8 accuracy on
Qwen2.5-Coder (0.5B / 1.5B / 3B). The same path should work
out-of-the-box on Llama, Mistral, Gemma, DeepSeek and other
standard-`nn.Linear`-using transformers — but "should" is not
"verified." Per-family recipes turn the README's claim
("essentially every public open-weights model") into
"specifically these families, with these PPL numbers."

**What to do:** `python examples/wikitext_ppl_bench.py
--model <hf-model-name>` and file the resulting PPL delta as an
issue or a PR adding the numbers to the README. Models we'd most
value: any Llama-3 variant, Mistral-7B-Instruct, Gemma-2 family,
DeepSeek-Coder.

### 4. Attention quantisation accuracy characterisation

**Effort:** ~1 day of GPU time + analysis.
**Why it matters:** The default `replace_linear_modules(model)`
quantises FFN linears (gate / up / down) and leaves attention +
`lm_head` in BF16. We ship a `--swap-attention --skip-kv-proj`
flag that extends quantisation to attention Q / O projections,
and informally it preserves accuracy on Qwen2.5-Coder-1.5B — but
this hasn't been characterised at the same rigour as the
FFN-only path. Closing this gap moves the model-wide memory
reduction from "~30–35%" (today's default) to "closer to 2×"
(the full-quantisation upper bound).

**What's needed:** WikiText-2 PPL delta with
`--swap-ffn --swap-attention --skip-kv-proj`, on at least one
Llama-class and one Mistral-class model. Open issue first;
this is the kind of contribution that needs design discussion
before code lands.

### 5. Substrate ports — RISC-V / Tenstorrent / AMD CDNA / Intel Gaudi

**Effort:** multi-week research projects, each.
**Why it matters:** The bposit compute path is integer-only and
substrate-portable by construction. The reference Python
implementation
(`packages/mosyne_bposit/deploy/5dst_cuda/bposit16_reference.py`
in the parent repo) is ~200 lines and bit-exact against the CUDA
kernels — easy to port. Research docs for each substrate port
live in `docs/research/` of the parent repo:
- `docs/research/tenstorrent_bposit_porting_plan_2026-05-13.md`
- `docs/research/amd_mi300x_bposit_porting_plan_2026-05-14.md`
- `docs/research/intel_gaudi3_bposit_porting_plan_2026-05-14.md`

These are real research projects, not weekend hacks. Open an
issue describing what substrate you want to port to and the
hardware you have access to; we'd love to collaborate from the
plan stage.

## What we're not looking for help with right now

So contributors don't sink time into things that won't land:

- **Container / Docker packaging.** The install path is
  intentionally `pip install + mosyne-bposit-build` (one nvcc
  invocation). Adding Docker layers above that complicates the
  story for the "one-line PyTorch swap" pitch. We may revisit
  later but not now.
- **Replacing the cuBLASLt INT8 IMMA backend with custom CUDA
  kernels.** The decision to route through cuBLASLt is
  load-bearing — it's why the matmul throughput is competitive
  with native BF16 HMMA without us having to maintain a kernel
  schedule per arch. Custom kernels would re-introduce per-arch
  fragmentation we deliberately avoid.
- **Training quantisation.** This package is inference-only.
  Posit-aware training is a real research direction, but it's
  far enough from the current scope that a PR adding it would
  be hard to review — better to spin up a separate research
  thread (file an issue) than land code.
- **CI for the GPU-only tests.** The CPU-only surface tests are
  in CI; the actual numerical claims need a GPU we don't have a
  good CI story for yet. We're aware; the right fix is hosted
  GPU runners (Lambda / Modal / etc.) and we'll get to it.

## Development setup

```bash
# In the public github clone:
pip install -e .[torch]
mosyne-bposit-build           # one-time nvcc 12.x build
pytest packages/mosyne_bposit/tests/      # 50+ CPU-only surface tests
mosyne-bposit-demo --strict   # actual numerical claims (needs GPU)
```

**GPU compatibility:** the default `pip install -e .[torch]`
pulls PyTorch with sm_50..sm_90 support (Volta → Hopper). If
you're developing on a Blackwell card (RTX 5090 / RTX PRO 6000,
sm_120), upgrade PyTorch to ≥ 2.7 with CUDA ≥ 12.8 — the demo
and probe will print an actionable hint at startup if your
PyTorch is too old for the active GPU:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

The compiled `libmosyne_bposit.so` itself targets both sm_86 and
sm_120 (see `mosyne-bposit-build`), so no rebuild is needed when
upgrading PyTorch.

For the parent framework (private gitlab repo — `gitlab.com/anomly/mosyne`):
- `pip install -e .` from the repo root
- `pytest` — 2670+ tests
- `ruff check .` — lint
- `mypy src` — typecheck

## PR conventions

- One change per PR. Two unrelated tightenings = two PRs.
- Tests required for new behaviour. CPU-only tests if the
  behaviour can be exercised on CPU; explicit GPU-required
  tests (skipped when CUDA absent) otherwise.
- Commit messages: imperative present tense, name what changes
  and why. Reference an issue number if one exists.
- Don't reformat unrelated code in the same PR. If you spot
  something while in the file, open a separate "drive-by cleanup"
  PR — easier to review.
- Don't add Claude / AI-coauthor attribution to commits. The
  `.claude` settings in this repo's parent already strip these;
  this is a belt-and-suspenders mention for clarity.

## Communication

- **GitHub Issues** for bug reports, feature discussions,
  porting plan discussions, probe-result submissions
- **GitHub Discussions** for open-ended questions ("would this
  work for X?") — keeps Issues focused on actionable items
- **ry@anomly.com** for anything that genuinely doesn't fit a
  public channel (security disclosures, BD / partnership
  conversations, sensitive substrate-port collaborations)

## Code of conduct

Standard "be respectful, don't be a jerk, treat criticism of code
as criticism of code (not of you)." We reserve the right to
moderate / block / un-merge anything that turns the project into
an unpleasant place to spend time. So far this has not been
necessary; we'd prefer to keep it that way.

## License

By contributing, you agree your contribution is licensed under
Apache-2.0 (the project license).
