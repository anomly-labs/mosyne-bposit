"""mosyne-bposit-build — compile libmosyne_bposit.so via nvcc.

Run once after ``pip install mosyne-bposit``:

    $ mosyne-bposit-build

Requires ``nvcc`` on PATH (CUDA 12.x). Default arch set covers
the production NVIDIA GPU fleet:

  sm_80  — NVIDIA A100 (datacentre Ampere)
  sm_86  — RTX 3090 / 3080 / A10 (consumer Ampere)
  sm_89  — RTX 4090 / Ada PRO (Ada Lovelace)
  sm_90  — NVIDIA H100 / H200 (Hopper)
  sm_120 — RTX 5090 / RTX PRO 6000 (Blackwell)

Without explicit sm_X codegen, an unrecognised GPU runs via
PTX JIT on first invocation — adds startup latency and may not
produce arch-optimal SASS. Each extra arch adds ~30 KB; the
full set lands well under 1 MB for the .so itself.

iter-300 (2026-05-19) added sm_80, sm_89, sm_90 to the default
set after the forge corpus surfaced the gap — without those
arches, A100 / 4090 / H100 fall back to PTX JIT, losing the
arch-specific optimisations bposit's quire / decode / encode
kernels benefit from. Override with --archs if you need a
different set.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_CU = _PKG_DIR / "_cuda" / "libmosyne_bposit.cu"
_OUT = _PKG_DIR / "libmosyne_bposit.so"


def build(*, archs: list[str] | None = None, verbose: bool = True) -> Path:
    """Compile the CUDA shared library and return its path.

    Raises RuntimeError if nvcc is missing or the build fails.
    """
    if not _CU.exists():
        raise RuntimeError(f"CUDA source missing: {_CU}")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc not found on PATH.  Install CUDA 12.x and ensure "
            "`nvcc --version` works before running mosyne-bposit-build."
        )

    # iter-300: full production-fleet arch set. See module docstring.
    archs = archs or ["80", "86", "89", "90", "120"]
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-diag-suppress", "186",
    ]
    for a in archs:
        cmd += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    cmd += ["-lcublasLt", "-lcudart", str(_CU), "-o", str(_OUT)]

    if verbose:
        print(" ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"nvcc failed (rc={rc})")

    if verbose:
        size = _OUT.stat().st_size / 1024
        print(f"\nbuilt {_OUT}  ({size:.1f} KB)")
    return _OUT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archs",
        default="80,86,89,90,120",
        help="Comma-separated SM archs to build for (default: "
             "80,86,89,90,120 — covers A100/3090/4090/H100/5090 "
             "without PTX-JIT fallback).",
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    try:
        build(archs=archs, verbose=not args.quiet)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
