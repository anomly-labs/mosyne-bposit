"""Drift-guard for the calibrated launch-readiness language in the
public bposit README.

The README went through a multi-iter calibration sweep this week
(post the Kaplan-correction incident) to make every claim
defensible under reviewer scrutiny:

  - Memory claim scoped (per-FFN-module 2× / ~30-35% whole-model)
  - Reproducibility claim scoped (Ampere + Blackwell verified;
    structural-by-construction elsewhere)
  - "What this doesn't claim" section added with 7 explicit
    corners
  - CI-gated reproducer + cross-GPU probe surfaced
  - CONTRIBUTING.md added with 5 prioritised asks

This test pins those invariants so a future iter cannot silently
remove them. The bar is "would a reviewer arriving at the README
catch a missing invariant in the first 60 seconds?" — if yes, the
invariant belongs here.

What this test does NOT do: enforce specific wording. The
calibrated language can evolve as long as the substantive
properties are preserved.
"""
from __future__ import annotations

from pathlib import Path

README = (Path(__file__).parent.parent / "README.md")
CONTRIBUTING = (Path(__file__).parent.parent / "CONTRIBUTING.md")


def _readme() -> str:
    return README.read_text()


# ---------------------------------------------------------------
# Section structure
# ---------------------------------------------------------------

def test_what_this_doesnt_claim_section_present() -> None:
    """The single most load-bearing section for launch-readiness.
    Disarms the HN failure mode where commenters race to point out
    unaddressed corners. Removing this section silently would
    unstick the Kaplan-correction calibration we did this week."""
    text = _readme()
    assert "## What this doesn't claim" in text, (
        "the bposit README lost its 'What this doesn't claim' "
        "section — this is the load-bearing launch-readiness "
        "calibration that disarms HN-commenter corner-finding. "
        "Restore it; do not silently remove."
    )


def test_section_order_pitch_then_envelope_then_integration() -> None:
    """The flow that makes the README read well: 'Why use this'
    (the pitch) → 'What this doesn't claim' (the envelope) →
    'PyTorch / HuggingFace integration' (how to actually use it).
    Reordering breaks the rhetorical flow that the launch-readiness
    sweep deliberately set up."""
    text = _readme()
    i_why = text.find("## Why use this")
    i_envelope = text.find("## What this doesn't claim")
    i_integration = text.find("## PyTorch / HuggingFace integration")
    assert i_why >= 0 and i_envelope >= 0 and i_integration >= 0, (
        "one of the three load-bearing section headings is missing"
    )
    assert i_why < i_envelope < i_integration, (
        f"section order broken: 'Why use this' (idx {i_why}), "
        f"'What this doesn't claim' (idx {i_envelope}), "
        f"'PyTorch / HuggingFace integration' (idx {i_integration}) "
        "must appear in that order"
    )


def test_contributing_section_present_and_links_to_contributing_md() -> None:
    """README's Contributing section is the teaser; CONTRIBUTING.md
    is the detail. Removing the teaser hides the file from any
    README skimmer; removing the link makes the teaser dead."""
    text = _readme()
    assert "## Contributing" in text, (
        "README lost its 'Contributing' section — the public "
        "doorway for HN readers wanting to help with the "
        "cross-GPU probe / FP8 bench / HF recipes."
    )
    assert "CONTRIBUTING.md" in text, (
        "README's Contributing section no longer points at "
        "CONTRIBUTING.md — the teaser is now dead-end."
    )
    assert CONTRIBUTING.exists(), "CONTRIBUTING.md itself is missing"


# ---------------------------------------------------------------
# "What this doesn't claim" — all 7 corners must still be present
# ---------------------------------------------------------------

def _envelope_section() -> str:
    text = _readme()
    start = text.find("## What this doesn't claim")
    # Section ends at the next H2.
    end = text.find("\n## ", start + 1)
    if end == -1:
        end = len(text)
    return text[start:end]


def test_envelope_names_memory_scope() -> None:
    """The per-FFN-module 2× vs ~30-35% whole-model distinction.
    This is the exact claim Kaplan caught us overclaiming on;
    losing the scope qualifier would re-introduce the failure
    mode."""
    section = _envelope_section()
    # Phrases that pin the scope (any one is sufficient; we're
    # tolerating wording evolution, not silent removal).
    assert ("FFN linears" in section or
            "per-module" in section or
            "modules we replace" in section), (
        "envelope no longer scopes the 2× memory claim to "
        "per-FFN-module — risks re-introducing the whole-model "
        "halving overclaim Kaplan caught."
    )
    assert ("whole-model" in section or
            "30" in section), (
        "envelope no longer mentions the whole-model memory "
        "number (~30-35%) — without it, a reader infers "
        "model-wide halving from the per-FFN 2×."
    )


def test_envelope_names_verified_gpu_scope() -> None:
    """Ampere + Blackwell verified; Hopper/Ada/Turing/Volta
    structural-not-empirical. Same launch-readiness invariant the
    Kaplan correction landed."""
    section = _envelope_section()
    assert "Ampere" in section and "Blackwell" in section, (
        "envelope no longer names the verified GPU arches "
        "(Ampere + Blackwell) — the structural claim alone is "
        "weaker than 'verified here, structural-by-construction "
        "elsewhere'."
    )
    # At least one not-yet-verified arch should be flagged.
    assert any(arch in section for arch in ("H100", "A100", "4090",
                                            "Hopper", "Ada", "Turing")), (
        "envelope no longer names a not-yet-verified arch (H100 / "
        "A100 / 4090 / Hopper / Ada / Turing) — without that, the "
        "reader can't tell what's structural-only vs empirical."
    )


def test_envelope_names_throughput_loss_corner() -> None:
    """The honest 'we lose on small decode shapes' admission. This
    is the corner that an HN commenter would catch first if it
    were missing."""
    section = _envelope_section()
    # Numeric anchor — 0.76× geomean, or 0.27 on decode FFN-down,
    # or just the qualitative claim.
    assert ("0.76" in section or
            "0.27" in section or
            "shape" in section.lower()), (
        "envelope no longer mentions the throughput-by-shape "
        "honesty — sets up the 'why is it slow on X' surprise "
        "comment we deliberately disarm."
    )


def test_envelope_names_fp8_gap() -> None:
    section = _envelope_section()
    assert "FP8" in section, (
        "envelope no longer admits the no-FP8-head-to-head gap "
        "— a senior reviewer (especially at NVIDIA) will catch "
        "this immediately if it's absent."
    )


def test_envelope_names_inference_only() -> None:
    section = _envelope_section()
    assert ("training" in section.lower() or
            "Inference only" in section), (
        "envelope no longer names the inference-only scope — "
        "without it, a model-training reviewer expects training "
        "support and is disappointed."
    )


# ---------------------------------------------------------------
# Reproducer + probe surface in the README
# ---------------------------------------------------------------

def test_strict_json_ci_oneliner_documented() -> None:
    """The launch-readiness reproducer flag combo. Removing this
    means HN readers can't find the CI-gating one-liner."""
    text = _readme()
    assert "--strict" in text and "--json" in text, (
        "README no longer documents the --strict --json CI "
        "gating one-liner — HN/CI readers need to know about "
        "this within 60 seconds of landing on the README."
    )


def test_cross_gpu_probe_mentioned() -> None:
    """The mosyne-bposit-probe command is the path for closing
    the H100/A100/4090 empirical-coverage gap from the outside.
    If it's not in the README, the launch-readiness CONTRIBUTING
    ask #1 has no surface mention to point readers at."""
    text = _readme()
    assert "mosyne-bposit-probe" in text, (
        "README no longer mentions mosyne-bposit-probe — closes "
        "off the highest-leverage CONTRIBUTING ask (cross-GPU "
        "verification on hardware we don't have)."
    )


# ---------------------------------------------------------------
# CONTRIBUTING.md substantive coverage
# ---------------------------------------------------------------

def _contributing() -> str:
    return CONTRIBUTING.read_text()


def test_contributing_lists_cross_gpu_probe_as_top_ask() -> None:
    """The probe is ranked #1 specifically because it's the
    lowest-effort, highest-strategic-value contribution available.
    A future iter that re-prioritises away from this would
    deprioritise the launch-readiness path."""
    text = _contributing()
    # Locate the first ask heading; check it mentions GPU verification.
    first_ask = text[text.find("### 1."):text.find("### 2.")]
    assert "probe" in first_ask.lower() or "verification" in first_ask.lower(), (
        "CONTRIBUTING.md's #1 ask is no longer cross-GPU probe "
        "verification — that ask is the highest-leverage one and "
        "explicitly closes the empirical-coverage launch gate."
    )


def test_contributing_names_what_we_are_not_looking_for() -> None:
    """Mirror of the README's 'What this doesn't claim' discipline
    for contributions. Without it, contributors sink time into PRs
    that won't merge."""
    text = _contributing()
    assert ("not looking for" in text.lower() or
            "not looking" in text.lower()), (
        "CONTRIBUTING.md no longer has a 'what we're NOT looking "
        "for' section — contributors will start sinking time into "
        "Docker packaging / custom CUDA kernels / training paths "
        "that won't land."
    )
