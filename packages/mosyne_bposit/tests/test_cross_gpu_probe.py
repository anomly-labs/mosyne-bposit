"""mosyne-bposit-probe cross-GPU bit-exactness probe — surface tests.

The actual probe needs a CUDA GPU + built .so to run, so we can't
exercise the numerical path here. What we CAN test in CPU-only CI:

  - The module imports cleanly without torch / CUDA
  - The argparse surface declares --json, --compare-with, --seed
  - The compare() function correctly identifies PASS / FAIL / missing
  - ProbeRecord serialises cleanly to JSON
  - The no-CUDA error path returns 1 with an actionable message
  - End-to-end: --json output is parseable JSON with the expected
    schema (the field names CI scripts will rely on)

What we can NOT test here: the actual cross-GPU bit-exactness. That
requires renting H100 / A100 / 4090 hardware and comparing hashes
to the 3090 / 5090 reference.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------
# Module imports + ProbeRecord shape
# ---------------------------------------------------------------

def test_module_imports_without_cuda() -> None:
    """Probe module must be importable in any environment — argparse
    inspection has to work even when torch isn't installed."""
    mod = importlib.import_module("mosyne_bposit.cross_gpu_probe")
    assert hasattr(mod, "main")
    assert hasattr(mod, "probe")
    assert hasattr(mod, "compare")
    assert hasattr(mod, "ProbeRecord")
    assert hasattr(mod, "CANONICAL_SHAPES")
    assert hasattr(mod, "CANONICAL_SEED")


def test_canonical_shapes_stable() -> None:
    """Drift-guard: changing the canonical shapes invalidates all
    prior reference hashes (3090, 5090, eventually H100/A100/4090).
    Test pins the three shape keys + their dimensions so a rename
    or dimension change breaks loudly in CI."""
    from mosyne_bposit.cross_gpu_probe import CANONICAL_SHAPES
    assert set(CANONICAL_SHAPES) == {"small", "llama_ffn", "square"}
    assert CANONICAL_SHAPES["small"] == (64, 2048, 2048)
    assert CANONICAL_SHAPES["llama_ffn"] == (128, 4096, 11008)
    assert CANONICAL_SHAPES["square"] == (256, 4096, 4096)


def test_probe_record_serialises_to_json() -> None:
    """The --json output is the artifact compared across machines.
    Test that the ProbeRecord round-trips through asdict + json.dumps
    without losing fields — catches dataclass renames."""
    from mosyne_bposit.cross_gpu_probe import ProbeRecord
    r = ProbeRecord(
        gpu_name="RTX 3090",
        compute_capability="sm_86",
        cuda_version="12.8",
        pytorch_version="2.5.0",
        seed=1234,
        hashes={"small": "abcdef0123456789", "llama_ffn": "fedcba9876543210"},
    )
    d = asdict(r)
    for field in ("gpu_name", "compute_capability", "cuda_version",
                  "pytorch_version", "seed", "hashes"):
        assert field in d, f"ProbeRecord missing field {field}"
    # Full round-trip through json must preserve everything.
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed == d


# ---------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------

def _capture_parser() -> argparse.ArgumentParser:
    from mosyne_bposit import cross_gpu_probe

    captured: dict = {}
    real_init = argparse.ArgumentParser.__init__

    def _capture(self, *a, **kw):
        real_init(self, *a, **kw)
        captured["parser"] = self

    def _abort(self, *a, **kw):  # noqa: ARG001
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "__init__", _capture), \
         patch.object(argparse.ArgumentParser, "parse_args", _abort):
        with pytest.raises(SystemExit):
            cross_gpu_probe.main([])
    return captured["parser"]


def test_argparse_has_json_flag() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "json" in dests, "--json missing (cross-machine comparison regression)"


def test_argparse_has_compare_with_flag() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "compare_with" in dests, "--compare-with missing"


def test_argparse_has_seed_flag() -> None:
    parser = _capture_parser()
    dests = {a.dest for a in parser._actions}
    assert "seed" in dests, "--seed missing"


# ---------------------------------------------------------------
# compare() correctness
# ---------------------------------------------------------------

def _record_with(hashes: dict) -> "ProbeRecord":   # noqa: F821
    from mosyne_bposit.cross_gpu_probe import ProbeRecord
    return ProbeRecord(
        gpu_name="STUB", compute_capability="sm_99",
        cuda_version="12.8", pytorch_version="2.5.0",
        seed=1234, hashes=hashes,
    )


def test_compare_passes_when_all_hashes_match() -> None:
    from mosyne_bposit.cross_gpu_probe import compare
    current = _record_with({"small": "aaa", "llama_ffn": "bbb", "square": "ccc"})
    reference = {"hashes": {"small": "aaa", "llama_ffn": "bbb", "square": "ccc"}}
    all_pass, lines = compare(current, reference)
    assert all_pass is True
    assert all("PASS" in line for line in lines)


def test_compare_fails_on_any_mismatch() -> None:
    """The whole point of the probe: any per-shape mismatch fails the
    cross-arch reproducibility claim."""
    from mosyne_bposit.cross_gpu_probe import compare
    current = _record_with({"small": "aaa", "llama_ffn": "DIFFERENT", "square": "ccc"})
    reference = {"hashes": {"small": "aaa", "llama_ffn": "bbb", "square": "ccc"}}
    all_pass, lines = compare(current, reference)
    assert all_pass is False
    # The failing shape names both sides in the diff line.
    fail_lines = [line for line in lines if "FAIL" in line]
    assert any("llama_ffn" in line and "DIFFERENT"[:16] in line
               for line in fail_lines)


def test_compare_flags_missing_keys_in_either_direction() -> None:
    from mosyne_bposit.cross_gpu_probe import compare
    # Reference has an extra shape that current is missing.
    current = _record_with({"small": "aaa", "llama_ffn": "bbb"})
    reference = {"hashes": {"small": "aaa", "llama_ffn": "bbb", "square": "ccc"}}
    all_pass, lines = compare(current, reference)
    assert all_pass is False
    assert any("square" in line and "MISSING in current" in line
               for line in lines)

    # And the reverse: current has an extra shape.
    current2 = _record_with({"small": "aaa", "llama_ffn": "bbb", "extra": "ddd"})
    reference2 = {"hashes": {"small": "aaa", "llama_ffn": "bbb"}}
    all_pass2, lines2 = compare(current2, reference2)
    assert all_pass2 is False
    assert any("extra" in line and "MISSING in reference" in line
               for line in lines2)


# ---------------------------------------------------------------
# No-CUDA error path
# ---------------------------------------------------------------

def test_returns_1_when_torch_missing(monkeypatch, capsys) -> None:
    from mosyne_bposit import cross_gpu_probe

    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict) else __builtins__.__import__

    def _block_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _block_torch)
    rc = cross_gpu_probe.main([])
    assert rc == 1
    assert "requires PyTorch" in capsys.readouterr().err


def test_returns_1_when_cuda_unavailable(monkeypatch, capsys) -> None:
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import cross_gpu_probe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rc = cross_gpu_probe.main([])
    assert rc == 1
    assert "CUDA not available" in capsys.readouterr().err


# ---------------------------------------------------------------
# End-to-end --json output structure (probe stubbed)
# ---------------------------------------------------------------

def test_json_mode_emits_parseable_record(monkeypatch, capsys) -> None:
    """The --json output is what cross-machine comparisons will rely
    on. Catches drift in the JSON schema."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import cross_gpu_probe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB-GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 9))
    # iter-299: short-circuit the shared GPU compat pre-flight.
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)

    def _stub_probe(seed=1234, shapes=None):
        return cross_gpu_probe.ProbeRecord(
            gpu_name="STUB-GPU",
            compute_capability="sm_99",
            cuda_version="12.8",
            pytorch_version=torch.__version__,
            seed=seed,
            hashes={k: f"stub_{k}_hash" for k in cross_gpu_probe.CANONICAL_SHAPES},
        )

    monkeypatch.setattr(cross_gpu_probe, "probe", _stub_probe)

    rc = cross_gpu_probe.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_name"] == "STUB-GPU"
    assert payload["compute_capability"] == "sm_99"
    assert payload["seed"] == 1234
    assert set(payload["hashes"]) == set(cross_gpu_probe.CANONICAL_SHAPES)


def test_compare_with_passes_when_hashes_match(
    monkeypatch, tmp_path, capsys
) -> None:
    """End-to-end: --compare-with returns 0 when the reference file's
    hashes match the current run."""
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import cross_gpu_probe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB-GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 9))
    # iter-299: short-circuit the shared GPU compat pre-flight.
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)

    canonical_hashes = {k: f"stub_{k}_hash" for k in cross_gpu_probe.CANONICAL_SHAPES}

    def _stub_probe(seed=1234, shapes=None):
        return cross_gpu_probe.ProbeRecord(
            gpu_name="STUB-GPU", compute_capability="sm_99",
            cuda_version="12.8", pytorch_version=torch.__version__,
            seed=seed, hashes=canonical_hashes,
        )

    monkeypatch.setattr(cross_gpu_probe, "probe", _stub_probe)

    reference_path = tmp_path / "probe_3090.json"
    reference_path.write_text(json.dumps({
        "gpu_name": "RTX 3090", "compute_capability": "sm_86",
        "hashes": canonical_hashes,
    }))

    rc = cross_gpu_probe.main(["--compare-with", str(reference_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "bit-identical" in out


def test_compare_with_returns_1_on_any_mismatch(
    monkeypatch, tmp_path
) -> None:
    pytest.importorskip("torch")
    import torch
    from mosyne_bposit import cross_gpu_probe

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "STUB-GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 9))
    # iter-299: short-circuit the shared GPU compat pre-flight.
    from mosyne_bposit import _gpu_compat
    monkeypatch.setattr(_gpu_compat, "check_gpu_compat", lambda: 0)

    def _stub_probe(seed=1234, shapes=None):
        return cross_gpu_probe.ProbeRecord(
            gpu_name="STUB-GPU", compute_capability="sm_99",
            cuda_version="12.8", pytorch_version=torch.__version__,
            seed=seed, hashes={"small": "a", "llama_ffn": "DIFFERENT", "square": "c"},
        )

    monkeypatch.setattr(cross_gpu_probe, "probe", _stub_probe)

    reference_path = tmp_path / "probe_3090.json"
    reference_path.write_text(json.dumps({
        "gpu_name": "RTX 3090", "compute_capability": "sm_86",
        "hashes": {"small": "a", "llama_ffn": "b", "square": "c"},
    }))

    rc = cross_gpu_probe.main(["--compare-with", str(reference_path)])
    assert rc == 1
