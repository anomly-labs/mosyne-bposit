# Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""mosyne-bposit-build — compile libmosyne_bposit.so via nvcc.

Run once after ``pip install mosyne-bposit``:

    $ mosyne-bposit-build

Requires ``nvcc`` on PATH (CUDA 12.x).  Compiles for sm_86 (Ampere) and
sm_120 (Blackwell) so the same .so works on RTX 3090 / 4090 / 5090.
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

    archs = archs or ["86", "120"]
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
        default="86,120",
        help="Comma-separated SM archs to build for (default: 86,120 for Ampere+Blackwell).",
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