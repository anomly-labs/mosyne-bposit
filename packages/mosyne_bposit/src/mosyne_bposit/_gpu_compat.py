"""GPU / PyTorch compatibility check used by mosyne-bposit-demo
and mosyne-bposit-probe.

The default ``pip install mosyne-bposit[torch]`` pulls in
PyTorch >= 2.0, which only ships kernels for sm_50..sm_90 (so
Volta through Hopper). NVIDIA Blackwell (sm_120 — RTX 5090,
RTX PRO 6000) needs PyTorch >= 2.7 with CUDA >= 12.8. Hitting
this incompatibility from a fresh install prints a cryptic
"CUDA error: no kernel image is available for execution on the
device" the moment the first torch op hits the GPU.

This module's ``check_gpu_compat()`` catches that case up front
and prints an actionable install hint instead.
"""
from __future__ import annotations

import sys


def check_gpu_compat() -> int:
    """Return 0 if the active GPU is supported by the installed
    PyTorch build, or 1 (after printing an actionable error to
    stderr) if not.

    Caller should propagate the return code as the process exit
    code — the CLIs do this in their main() startup."""
    try:
        import torch
    except ImportError:
        print("ERROR: this command requires PyTorch. "
              "Install with `pip install mosyne-bposit[torch]`.",
              file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This command requires "
              "a CUDA-capable NVIDIA GPU.", file=sys.stderr)
        return 1

    major, minor = torch.cuda.get_device_capability()
    device_sm = f"sm_{major}{minor}"
    try:
        supported = set(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        # If we can't introspect supported archs, fall through and
        # let the first kernel launch fail with whatever message
        # PyTorch produces — better than a false-positive abort.
        return 0

    if not supported:
        # PyTorch built without CUDA arch support at all — unusual.
        return 0

    if device_sm not in supported:
        gpu = torch.cuda.get_device_name(0)
        print(file=sys.stderr)
        print(f"ERROR: your GPU ({gpu}, {device_sm}) is not in the "
              f"set of architectures the installed PyTorch was built "
              f"for.", file=sys.stderr)
        print(f"       PyTorch {torch.__version__} supports: "
              f"{', '.join(sorted(supported))}", file=sys.stderr)
        print(file=sys.stderr)
        if major >= 12:
            # Blackwell or newer — the common case this check exists for.
            print("Blackwell (sm_120, e.g. RTX 5090) needs PyTorch "
                  ">= 2.7 with CUDA >= 12.8. Install via:", file=sys.stderr)
            print("    pip install --upgrade --index-url "
                  "https://download.pytorch.org/whl/cu128 torch",
                  file=sys.stderr)
            print("Or use a PyTorch nightly build — see "
                  "https://pytorch.org/get-started/locally/",
                  file=sys.stderr)
        else:
            # Older arch missing — unusual, just point at the matrix.
            print("See PyTorch install matrix at "
                  "https://pytorch.org/get-started/locally/ for a "
                  f"build that includes {device_sm}.", file=sys.stderr)
        print(file=sys.stderr)
        return 1

    return 0
