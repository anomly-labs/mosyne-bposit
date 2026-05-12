"""Verify ``narrow_n_threshold`` skips narrow-output projections.

Iter-56 added the threshold to ``replace_linear_modules`` so callers can
avoid the bposit-vs-bf16 latency floor at narrow N (the GQA K/V proj
is the canonical 4096→1024 case where bposit-IMMA loses ~3×). The
diagnostic doc lives at
``docs/research/bposit_attn_regression_breakdown_2026-05-09.md``.
"""
from __future__ import annotations

import pytest
import torch.nn as nn


# Skip the whole module if the device-side library isn't available.
# replace_linear_modules instantiates BPositLinear which loads the .so.
def _library_available() -> bool:
    try:
        from mosyne_bposit import library_path
    except ImportError:
        return False
    return library_path().exists()


pytestmark = pytest.mark.skipif(
    not _library_available(),
    reason="libmosyne_bposit.so not built; run mosyne-bposit-build",
)


class _ToyAttention(nn.Module):
    """Mimics Qwen-style GQA attention: q/o are wide, k/v are narrow."""
    def __init__(self, hidden=4096, kv_dim=1024):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _ToyFFN(nn.Module):
    def __init__(self, hidden=4096, ffn_dim=14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden, bias=False)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _ToyAttention()
        self.mlp = _ToyFFN()


def _count_bposit(model: nn.Module) -> int:
    from mosyne_bposit.torch_compat import BPositLinear
    return sum(1 for m in model.modules() if isinstance(m, BPositLinear))


def _count_linear(model: nn.Module) -> int:
    from mosyne_bposit.torch_compat import BPositLinear
    return sum(
        1 for m in model.modules()
        if isinstance(m, nn.Linear) and not isinstance(m, BPositLinear)
    )


def test_threshold_none_replaces_everything():
    from mosyne_bposit.torch_compat import replace_linear_modules
    block = _ToyBlock()
    n_before = _count_linear(block)
    replace_linear_modules(block)
    assert _count_bposit(block) == n_before
    assert _count_linear(block) == 0


def test_threshold_skips_narrow_n():
    from mosyne_bposit.torch_compat import replace_linear_modules
    block = _ToyBlock()
    replace_linear_modules(block, narrow_n_threshold=2048)
    # k_proj and v_proj have out_features=1024 < 2048; they stay nn.Linear.
    assert isinstance(block.attn.k_proj, nn.Linear)
    assert isinstance(block.attn.v_proj, nn.Linear)
    from mosyne_bposit.torch_compat import BPositLinear
    assert not isinstance(block.attn.k_proj, BPositLinear)
    assert not isinstance(block.attn.v_proj, BPositLinear)
    # q_proj (4096), o_proj (4096), gate_proj (14336), up_proj (14336),
    # down_proj (4096) all >= 2048 → swapped.
    assert isinstance(block.attn.q_proj, BPositLinear)
    assert isinstance(block.attn.o_proj, BPositLinear)
    assert isinstance(block.mlp.gate_proj, BPositLinear)
    assert isinstance(block.mlp.up_proj, BPositLinear)
    assert isinstance(block.mlp.down_proj, BPositLinear)


def test_threshold_combines_with_predicate():
    from mosyne_bposit.torch_compat import replace_linear_modules, BPositLinear
    block = _ToyBlock()

    # Predicate keeps only attention. Threshold then narrows further.
    # The toy model's qualified names look like "attn.q_proj" (no outer
    # wrapper) — a real Qwen path is "model.layers.0.self_attn.q_proj"
    # so production callers would use ".self_attn." as the substring.
    def keep_attention(name, mod):
        return name.startswith("attn.")

    replace_linear_modules(block, predicate=keep_attention,
                           narrow_n_threshold=2048)
    # FFN untouched (predicate excludes them).
    assert not isinstance(block.mlp.gate_proj, BPositLinear)
    assert not isinstance(block.mlp.down_proj, BPositLinear)
    # Q and O swapped.
    assert isinstance(block.attn.q_proj, BPositLinear)
    assert isinstance(block.attn.o_proj, BPositLinear)
    # K and V skipped (narrow N).
    assert not isinstance(block.attn.k_proj, BPositLinear)
    assert not isinstance(block.attn.v_proj, BPositLinear)


def test_threshold_zero_replaces_everything():
    """Threshold of 0 should be a no-op (nothing has out_features < 0)."""
    from mosyne_bposit.torch_compat import replace_linear_modules
    block = _ToyBlock()
    n_before = _count_linear(block)
    replace_linear_modules(block, narrow_n_threshold=0)
    assert _count_bposit(block) == n_before


def test_threshold_extreme_skips_all():
    """Threshold above every out_features should leave the model unchanged."""
    from mosyne_bposit.torch_compat import replace_linear_modules
    block = _ToyBlock()
    n_before = _count_linear(block)
    replace_linear_modules(block, narrow_n_threshold=100_000)
    assert _count_bposit(block) == 0
    assert _count_linear(block) == n_before


# --- skip_kv_proj ---------------------------------------------------------
# Iter-58: model-agnostic alternative to narrow_n_threshold=kv_dim+1.

def test_skip_kv_proj_swaps_q_o_ffn_only():
    from mosyne_bposit.torch_compat import replace_linear_modules, BPositLinear
    block = _ToyBlock()
    replace_linear_modules(block, skip_kv_proj=True)
    # K/V skipped by name regardless of their out_features.
    assert isinstance(block.attn.k_proj, nn.Linear)
    assert isinstance(block.attn.v_proj, nn.Linear)
    assert not isinstance(block.attn.k_proj, BPositLinear)
    assert not isinstance(block.attn.v_proj, BPositLinear)
    # Everything else swapped.
    assert isinstance(block.attn.q_proj, BPositLinear)
    assert isinstance(block.attn.o_proj, BPositLinear)
    assert isinstance(block.mlp.gate_proj, BPositLinear)
    assert isinstance(block.mlp.up_proj, BPositLinear)
    assert isinstance(block.mlp.down_proj, BPositLinear)


def test_skip_kv_proj_default_false_keeps_old_behaviour():
    """Default skip_kv_proj=False matches the iter-56 behaviour
    (swap everything that matches predicate)."""
    from mosyne_bposit.torch_compat import replace_linear_modules
    block = _ToyBlock()
    n_before = _count_linear(block)
    replace_linear_modules(block)  # all defaults
    assert _count_bposit(block) == n_before


def test_skip_kv_proj_composes_with_predicate():
    from mosyne_bposit.torch_compat import replace_linear_modules, BPositLinear
    block = _ToyBlock()

    def keep_attention(name, mod):
        return name.startswith("attn.")

    replace_linear_modules(block, predicate=keep_attention,
                           skip_kv_proj=True)
    # FFN untouched (predicate excludes them).
    assert not isinstance(block.mlp.gate_proj, BPositLinear)
    # Q/O swapped.
    assert isinstance(block.attn.q_proj, BPositLinear)
    assert isinstance(block.attn.o_proj, BPositLinear)
    # K/V skipped.
    assert not isinstance(block.attn.k_proj, BPositLinear)
    assert not isinstance(block.attn.v_proj, BPositLinear)


def test_skip_kv_proj_composes_with_threshold():
    """Both gates apply — a module is skipped if EITHER rejects it.
    A genuinely-composing case is one where the threshold alone would
    leave K/V in (e.g. threshold=500 on a model with kv_dim=1024) and
    ``skip_kv_proj`` is what catches them."""
    from mosyne_bposit.torch_compat import replace_linear_modules, BPositLinear
    block = _ToyBlock()  # hidden=4096, kv_dim=1024
    replace_linear_modules(block, narrow_n_threshold=500, skip_kv_proj=True)
    # K/V (1024) would survive the threshold (1024 >= 500); skip_kv_proj
    # is what rejects them.
    assert not isinstance(block.attn.k_proj, BPositLinear)
    assert not isinstance(block.attn.v_proj, BPositLinear)
    # Q/O (4096) and FFN all survive both gates.
    assert isinstance(block.attn.q_proj, BPositLinear)
    assert isinstance(block.attn.o_proj, BPositLinear)
    assert isinstance(block.mlp.gate_proj, BPositLinear)
    assert isinstance(block.mlp.up_proj, BPositLinear)
    assert isinstance(block.mlp.down_proj, BPositLinear)


def test_skip_kv_proj_matches_only_by_suffix():
    """Submodule attrs that *contain* k_proj as a substring (e.g.
    a hypothetical 'pre_k_proj_gate') must NOT be skipped — only
    full ``.k_proj`` / ``.v_proj`` suffix matches do."""
    from mosyne_bposit.torch_compat import replace_linear_modules, BPositLinear

    class _SuffixTrap(nn.Module):
        def __init__(self):
            super().__init__()
            # Innocuous linear whose attr name contains 'k_proj' as a
            # prefix substring; the suffix matcher must not catch this.
            self.k_proj_norm = nn.Linear(128, 128, bias=False)
            self.k_proj = nn.Linear(128, 128, bias=False)  # legit kv

    m = _SuffixTrap()
    replace_linear_modules(m, skip_kv_proj=True)
    assert isinstance(m.k_proj_norm, BPositLinear)        # not skipped
    assert isinstance(m.k_proj, nn.Linear)                # skipped
    assert not isinstance(m.k_proj, BPositLinear)
