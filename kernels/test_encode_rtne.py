#!/usr/bin/env python3
# Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the round-to-nearest-ties-to-even mode of the Python bposit16
reference encoder.

The default encoder mode is ``"truncate"`` — that's the bit-exact match for
the CUDA encoder and the baked LUTs. This test exercises the alternative
``mode="rtne"`` path:

  * Default mode is preserved (truncate) — every existing call site sees
    identical output to the original implementation.
  * RTNE never increases the round-trip error vs truncate, and it strictly
    reduces it on at least one input where the round bit is 1.
  * RTNE saturates at maxPos rather than wrapping.
  * Special inputs (0, ±maxPos, ±minPos) match the truncate path.

Run as a standalone script:

    cd kernels
    python test_encode_rtne.py
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

# Allow running from anywhere — locate sibling reference module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bposit16_reference import (  # noqa: E402
    decode_bposit16,
    decoded_to_fraction,
    encode_bposit16,
)


def _bp16_to_fraction(bp: int) -> Fraction:
    d = decode_bposit16(bp)
    if d.is_special == "zero":
        return Fraction(0)
    return decoded_to_fraction(d)


def _round_trip_relerr(value: float, mode: str) -> float:
    bp = encode_bposit16(value, mode=mode)
    decoded = float(_bp16_to_fraction(bp))
    if value == 0:
        return abs(decoded)
    return abs(decoded - value) / abs(value)


def test_default_mode_is_truncate():
    """Calling encode_bposit16(x) with no mode arg must give identical
    output to encode_bposit16(x, mode='truncate')."""
    for v in [1.0, 1.3, 0.7, 7.5, math.pi, 1e-4, 12345.0, -42.0]:
        assert encode_bposit16(v) == encode_bposit16(v, mode="truncate"), v


def test_rtne_matches_truncate_when_no_rounding_needed():
    """Exactly representable values shouldn't change between modes."""
    # 1.0, 2.0, 4.0, 1/2, 1/4 etc. are bit-exact representable.
    for v in [1.0, 2.0, 4.0, 0.5, 0.25, 0.125, -1.0, -8.0]:
        a = encode_bposit16(v, mode="truncate")
        b = encode_bposit16(v, mode="rtne")
        assert a == b, f"{v}: truncate={a:#06x} rtne={b:#06x}"


def test_rtne_never_worse_than_truncate():
    """Across a swept set of values, RTNE round-trip error must be <= truncate."""
    rng_seed = 0xB0517
    import random

    rng = random.Random(rng_seed)
    values = [
        rng.uniform(-1e3, 1e3) for _ in range(2_000)
    ] + [
        rng.uniform(-1e-3, 1e-3) for _ in range(2_000)
    ]
    worse_count = 0
    strict_better = 0
    for v in values:
        if v == 0:
            continue
        e_trunc = _round_trip_relerr(v, "truncate")
        e_rtne = _round_trip_relerr(v, "rtne")
        if e_rtne > e_trunc + 1e-15:  # tiny FP slack on the comparison itself
            worse_count += 1
        if e_rtne + 1e-15 < e_trunc:
            strict_better += 1
    assert worse_count == 0, f"RTNE was worse than truncate on {worse_count} inputs"
    assert strict_better > 0, "RTNE never improved on truncate — sweep is too narrow"


def test_rtne_specific_round_up_case():
    """Pick a value that lands with round_bit=1 + sticky bits, verify
    RTNE encodes the next-larger representable while truncate encodes the
    current. This is the smoking-gun test that the round-up path runs."""
    # Find any value where the two modes differ. With a wide enough sweep
    # we will see at least one — assert that and check the larger field is RTNE.
    found = None
    for n in range(1, 4096):
        v = 1.0 + n / 8192.0  # subtle fractional perturbations near 1.0
        a = encode_bposit16(v, mode="truncate")
        b = encode_bposit16(v, mode="rtne")
        if a != b:
            found = (v, a, b)
            break
    assert found is not None, "no round-up case found in the sweep — encoder logic suspect"
    v, a, b = found
    assert b == a + 1, f"RTNE round-up expected a+1; got truncate={a:#06x} rtne={b:#06x}"


def test_rtne_saturates_at_maxpos():
    """Values >> maxPos must saturate rather than wrap, in both modes."""
    huge = 2.0 ** 60  # well past 2^48 = ~2.8e14
    assert encode_bposit16(huge, mode="truncate") == 0x7FFF
    assert encode_bposit16(huge, mode="rtne") == 0x7FFF


def test_invalid_mode_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown rounding mode"):
        encode_bposit16(1.0, mode="banker")


def test_zero_and_specials_unchanged():
    assert encode_bposit16(0.0, mode="rtne") == 0
    assert encode_bposit16(0.0, mode="truncate") == 0
    assert encode_bposit16(0, mode="rtne") == 0


def main() -> int:
    tests = [
        test_default_mode_is_truncate,
        test_rtne_matches_truncate_when_no_rounding_needed,
        test_rtne_never_worse_than_truncate,
        test_rtne_specific_round_up_case,
        test_rtne_saturates_at_maxpos,
        test_zero_and_specials_unchanged,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        return 1
    print(f"\n{len(tests)}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
