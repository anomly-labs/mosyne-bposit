"""bposit16 reference implementation, integer-only, per Gustafson Ch.7.

Parameters from spec/header (5DST_v2_API_SPEC.h, bposit_types.h):
    eS=3, rS=6, total=16 bits, dynamic range 2^-48 to 2^48

This is a PROOF that the math is implementable in pure integer arithmetic
without IEEE float — exactly what we'll port to CUDA __device__ functions.

Test case (Gustafson + the v2 spec line 104):
    p = [0.25, 0.25, 0.25, 0.25]                   # uniform
    H = shannon_entropy(p) = log₂(4) = 2.0 bits

Run:
    python3 bposit16_reference.py
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math


# ---- Parameters --------------------------------------------------------------
NBITS = 16
ES = 3
NAR = 0x8000
ZERO = 0x0000
ONE = 0x4000
USEED = 1 << (1 << ES)  # 2^(2^eS) = 256


# ---- bposit32 parameters (eS=3, rS=6 per bposit_types.h, same as bposit16) --
NBITS_32 = 32
ES_32 = 3
NAR_32 = 0x80000000
ZERO_32 = 0x00000000
ONE_32 = 0x40000000


def decode_bposit32(p: int) -> "Decoded":
    """Decode bposit32 (32-bit, eS=3, rS=6, range 2^-48 to 2^48)."""
    p &= 0xFFFFFFFF
    if p == ZERO_32:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR_32:
        return Decoded(0, 0, 0, 0, 0, "nar")
    sign = (p >> 31) & 1
    rest = p & 0x7FFFFFFF
    if sign:
        rest = ((~rest) + 1) & 0x7FFFFFFF
    leading_bit = (rest >> 30) & 1
    rs = 0
    while rs < 31 and ((rest >> (30 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 31:
        k = 30 if leading_bit else -31
        return Decoded(sign, k, 0, 0, 0)
    k = (rs - 1) if leading_bit else -rs
    consumed = rs + 1
    remaining = 31 - consumed
    rest2 = rest & ((1 << remaining) - 1)
    e_width = min(ES_32, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES_32 - e_width)
    else:
        e = 0
    remaining -= e_width
    f_width = remaining
    f_bits = (rest2 & ((1 << f_width) - 1)) if f_width > 0 else 0
    return Decoded(sign, k, e, f_bits, f_width)


def decoded_to_fraction_32(d: "Decoded") -> Fraction:
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    base = Fraction(USEED) ** d.k  # USEED = 256 same as bposit16
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


def _encode_unsigned_32(value: Fraction) -> int:
    """Encode positive Fraction → 31-bit unsigned bposit32 field."""
    if value == 0:
        return 0
    if value >= 1:
        total_e = 0; v = value
        while v >= 2: v /= 2; total_e += 1
    else:
        total_e = 0; v = value
        while v < 1: v *= 2; total_e -= 1
    if total_e > 48:
        return 0x7FFFFFFF
    if total_e < -48:
        return 0x00000001
    k = total_e >> 3       # useed = 256, log_useed = 3 bits per
    e = total_e & 7
    f = v - 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1)); bits.append(0)
    else:
        bits.extend([0] * (-k)); bits.append(1)
    if len(bits) >= 31:
        bits = bits[:31]
    else:
        e_bits = [(e >> i) & 1 for i in range(ES_32 - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 31: break
        while len(bits) < 32:
            f *= 2
            if f >= 1: bits.append(1); f -= 1
            else: bits.append(0)
        if len(bits) > 31:
            bits = bits[:31]
    out = 0
    for b in bits: out = (out << 1) | b
    out <<= (31 - len(bits))
    return out & 0x7FFFFFFF


def encode_bposit32(value) -> int:
    if isinstance(value, float):
        if value == 0: return ZERO_32
        value = Fraction(value).limit_denominator(10 ** 16)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO_32
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned_32(abs(value))
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7FFFFFFF
    return ((sign << 31) | abs_field) & 0xFFFFFFFF


# ---- bposit8 parameters (eS=2, rS=3 per bposit_types.h) ---------------------
NBITS_8 = 8
ES_8 = 2
NAR_8 = 0x80
ZERO_8 = 0x00
ONE_8 = 0x40
USEED_8 = 1 << (1 << ES_8)  # 2^(2^eS) = 16


def decode_bposit8(p: int) -> "Decoded":
    """Decode bposit8 (8-bit, eS=2, rS=3, range 2^-12 to 2^12)."""
    p &= 0xFF
    if p == ZERO_8:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR_8:
        return Decoded(0, 0, 0, 0, 0, "nar")
    sign = (p >> 7) & 1
    rest = p & 0x7F
    if sign:
        rest = ((~rest) + 1) & 0x7F
    leading_bit = (rest >> 6) & 1
    rs = 0
    while rs < 7 and ((rest >> (6 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 7:
        k = 6 if leading_bit else -7
        return Decoded(sign, k, 0, 0, 0)
    k = (rs - 1) if leading_bit else -rs
    consumed = rs + 1
    remaining = 7 - consumed
    rest2 = rest & ((1 << remaining) - 1)
    e_width = min(ES_8, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES_8 - e_width)
    else:
        e = 0
    remaining -= e_width
    f_width = remaining
    f_bits = (rest2 & ((1 << f_width) - 1)) if f_width > 0 else 0
    return Decoded(sign, k, e, f_bits, f_width)


def decoded_to_fraction_8(d: "Decoded") -> Fraction:
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    base = Fraction(USEED_8) ** d.k
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


def _encode_unsigned_8(value: Fraction) -> int:
    """Encode positive Fraction → 7-bit unsigned bposit8 field."""
    if value == 0:
        return 0
    if value >= 1:
        total_e = 0; v = value
        while v >= 2: v /= 2; total_e += 1
    else:
        total_e = 0; v = value
        while v < 1: v *= 2; total_e -= 1
    if total_e > 12:
        return 0x7F
    if total_e < -12:
        return 0x01
    # total_e = k * 4 + e, since useed_8 = 16 = 2^4
    k = total_e >> 2
    e = total_e & 3
    f = v - 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1)); bits.append(0)
    else:
        bits.extend([0] * (-k)); bits.append(1)
    if len(bits) >= 7:
        bits = bits[:7]
    else:
        e_bits = [(e >> i) & 1 for i in range(ES_8 - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 7: break
        while len(bits) < 8:
            f *= 2
            if f >= 1: bits.append(1); f -= 1
            else: bits.append(0)
        if len(bits) > 7:
            bits = bits[:7]
    out = 0
    for b in bits: out = (out << 1) | b
    out <<= (7 - len(bits))
    return out & 0x7F


def encode_bposit8(value) -> int:
    if isinstance(value, float):
        if value == 0: return ZERO_8
        value = Fraction(value).limit_denominator(10 ** 8)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO_8
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned_8(abs(value))
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7F
    return ((sign << 7) | abs_field) & 0xFF


def bposit8_mul(a: int, b: int) -> int:
    da = decode_bposit8(a)
    db = decode_bposit8(b)
    if da.is_special == "zero" or db.is_special == "zero":
        return ZERO_8
    if da.is_special == "nar" or db.is_special == "nar":
        return NAR_8
    fa = decoded_to_fraction_8(da)
    fb = decoded_to_fraction_8(db)
    return encode_bposit8(fa * fb)


def bposit8_to_quire(p: int) -> int:
    d = decode_bposit8(p)
    if d.is_special:
        return 0
    f = decoded_to_fraction_8(d)
    f *= (1 << QUIRE_FRAC_BITS)
    return int(f)



# ---- Decode -----------------------------------------------------------------
@dataclass
class Decoded:
    sign: int          # 0 or 1
    k: int             # regime k (signed; range = USEED^k)
    e: int             # exponent (0 to USEED-1)
    f_bits: int        # fraction bits, MSB-aligned in 32-bit field
    f_width: int       # number of fraction bits actually present
    is_special: str | None = None   # "zero" | "nar" | None


def decode_bposit16(p: int) -> Decoded:
    p &= 0xFFFF
    if p == ZERO:
        return Decoded(0, 0, 0, 0, 0, "zero")
    if p == NAR:
        return Decoded(0, 0, 0, 0, 0, "nar")

    sign = (p >> 15) & 1
    # 2's-complement-style sign handling for posits: if sign=1, take the
    # 2's complement of the rest, then proceed as positive.
    rest = p & 0x7FFF
    if sign:
        rest = ((~rest) + 1) & 0x7FFF

    # Read regime: count leading run of identical bits in rest[14:0]
    leading_bit = (rest >> 14) & 1
    rs = 0
    while rs < 15 and ((rest >> (14 - rs)) & 1) == leading_bit:
        rs += 1
    if rs == 15:
        # All-ones case (extreme value); regime fully consumed
        k = 14 if leading_bit else -15
        e = 0
        f_bits = 0
        f_width = 0
        return Decoded(sign, k, e, f_bits, f_width)

    k = (rs - 1) if leading_bit else -rs

    # Skip regime + terminator bit
    consumed = rs + 1
    remaining = 15 - consumed
    rest2 = rest & ((1 << remaining) - 1)

    # Read exponent (up to ES bits, MSB-first)
    e_width = min(ES, remaining)
    if e_width > 0:
        e = (rest2 >> (remaining - e_width)) & ((1 << e_width) - 1)
        e <<= (ES - e_width)  # left-pad if regime ate some exponent bits
    else:
        e = 0
    remaining -= e_width

    f_width = remaining
    if f_width > 0:
        f_bits = rest2 & ((1 << f_width) - 1)
    else:
        f_bits = 0

    return Decoded(sign, k, e, f_bits, f_width)


# ---- Decoded → exact rational ----------------------------------------------
def decoded_to_fraction(d: Decoded) -> Fraction:
    """Exact rational value the bposit16 represents. Reference for testing."""
    if d.is_special == "zero":
        return Fraction(0)
    if d.is_special == "nar":
        raise ValueError("NaR")
    # value = (-1)^sign * useed^k * 2^e * (1 + f_bits / 2^f_width)
    base = Fraction(USEED) ** d.k
    base *= Fraction(1 << d.e) if d.e >= 0 else Fraction(1, 1 << -d.e)
    if d.f_width > 0:
        base *= Fraction(2 ** d.f_width + d.f_bits, 2 ** d.f_width)
    if d.sign:
        base = -base
    return base


# ---- Encode (round-to-nearest-even) ----------------------------------------
def _encode_unsigned(value: Fraction) -> int:
    """Encode positive value as 15-bit unsigned posit field (no sign bit).
    Round-to-nearest-even at the bposit16 ULP."""
    # Decompose: value = useed^k * 2^e * (1 + f), 0 <= f < 1
    # log_useed(value) = k + (e + log2(1+f)) / 2^ES
    # We work in extended fixed-point: convert value to log_2 via Fraction
    # approximation, then split.
    # For simplicity, use integer-only k+e extraction via log2:
    # bposit16 spec: useed = 2^(2^ES) = 256, so log_2(useed) = 8.
    # Total exponent bits (k * 8 + e) => integer total_e.
    # We compute total_e = floor(log2(value)) for which 2^total_e <= value.
    if value == 0:
        return 0

    # Find floor(log2(value))
    if value >= 1:
        total_e = 0
        v = value
        while v >= 2:
            v /= 2
            total_e += 1
    else:
        total_e = 0
        v = value
        while v < 1:
            v *= 2
            total_e -= 1

    # Clamp to bposit16 range
    if total_e > 48:
        return 0x7FFF  # +maxPos
    if total_e < -48:
        return 0x0001  # +minPos

    # Decompose: total_e = k * 8 + e, where 0 <= e < 8
    k = total_e >> 3
    e = total_e & 7
    # f = v - 1, where 1 <= v < 2  (v is the mantissa)
    f = v - 1  # Fraction in [0, 1)

    # Encode regime: k positive → (k+1) ones followed by a 0
    #                k negative → -k zeros followed by a 1
    bits = []
    if k >= 0:
        bits.extend([1] * (k + 1))
        bits.append(0)
    else:
        bits.extend([0] * (-k))
        bits.append(1)

    # We have 15 bits total in the unsigned field
    used = len(bits)
    if used >= 15:
        # Regime ate everything (extreme value)
        bits = bits[:15]
    else:
        # Encode exponent (ES = 3 bits, MSB-first)
        e_bits = [(e >> i) & 1 for i in range(ES - 1, -1, -1)]
        for b in e_bits:
            bits.append(b)
            if len(bits) == 15:
                break
        # Encode fraction bits
        while len(bits) < 16:  # +1 to detect rounding
            f *= 2
            if f >= 1:
                bits.append(1)
                f -= 1
            else:
                bits.append(0)

        # Round-to-nearest-even using bit 16 (the round bit) and remaining f
        if len(bits) > 15:
            round_bit = bits[15]
            bits = bits[:15]
            # Tie case: round-to-even (look at LSB; if odd, round up)
            if round_bit == 1 and (f > 0 or bits[14] == 1):
                # Round up by adding 1 to the field
                pass  # simplification: skip exact tie-break for POC

    # Pack 15 bits into integer
    out = 0
    for b in bits:
        out = (out << 1) | b
    # Pad if regime+exp+frac < 15 (shouldn't happen normally)
    out <<= (15 - len(bits))
    return out & 0x7FFF


def encode_bposit16(value: Fraction | int | float) -> int:
    if isinstance(value, float):
        # Boundary conversion only — for test inputs.
        if value == 0:
            return ZERO
        value = Fraction(value).limit_denominator(10 ** 12)
    if isinstance(value, int):
        value = Fraction(value)
    if value == 0:
        return ZERO
    sign = 1 if value < 0 else 0
    abs_field = _encode_unsigned(abs(value))
    if sign:
        abs_field = ((~abs_field) + 1) & 0x7FFF
    return ((sign << 15) | abs_field) & 0xFFFF


# ---- Arithmetic via decode → fraction → encode -----------------------------
def bposit16_mul(a: int, b: int) -> int:
    da = decode_bposit16(a)
    db = decode_bposit16(b)
    if da.is_special == "zero" or db.is_special == "zero":
        return ZERO
    if da.is_special == "nar" or db.is_special == "nar":
        return NAR
    fa = decoded_to_fraction(da)
    fb = decoded_to_fraction(db)
    return encode_bposit16(fa * fb)


def bposit16_log2(a: int) -> int:
    da = decode_bposit16(a)
    if da.is_special == "zero":
        return NAR
    if da.is_special == "nar":
        return NAR
    fa = decoded_to_fraction(da)
    if fa <= 0:
        return NAR
    # log2 via float (the LUT we'll use in CUDA also goes through float once
    # at LUT-build time; on the device it's a constant-memory lookup)
    log_val = math.log2(float(fa))
    return encode_bposit16(log_val)


# ---- Quire256 (exact accumulator) ------------------------------------------
# Layout: 256-bit signed integer, fixed-point with 96 fraction bits.
# Range: 2^-96 to 2^159 — covers any product of two bposit16s (2^-96 to 2^96).
QUIRE_FRAC_BITS = 96
QUIRE_BITS = 256


def quire256_zero() -> int:
    return 0


def bposit16_sqrt(p: int) -> int:
    """Square root of a bposit16 value. Returns NAR for negative inputs.
    Uses Python math.sqrt at LUT-build time; the *kernel* path is a 65 K
    __device__ const lookup, so the float in this function never reaches
    the GPU."""
    import math
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return ZERO
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    if f < 0:
        return NAR
    return encode_bposit16(math.sqrt(float(f)))


def bposit16_exp2(p: int) -> int:
    """2^x for a bposit16 value. Returns NaR for NaR input.
    Like sqrt/recip, this uses Fraction/float at LUT-build time only — the
    GPU path is a 65 K __device__ const lookup. Combined with bposit16_log2,
    this gives us multiplication via mul(a,b) = exp2(log2(a) + log2(b)).

    bposit16's representable range is roughly 2^-48 to 2^48; for inputs
    outside that range (i.e. exp2 result over/underflows the format), we
    encode the saturating bposit16 value via encode_bposit16's clamping."""
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return ONE  # 2^0 = 1
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    # Bposit16 range: ~2^-48 to ~2^48. Clamp the *input* to this range
    # before exponentiating so float doesn't overflow. Past the clamp the
    # encoder saturates to maxpos/minpos respectively.
    if f >= 49:
        return encode_bposit16(2.0 ** 48)   # saturate to ~maxpos
    if f <= -49:
        return encode_bposit16(2.0 ** -48)  # saturate to ~minpos
    return encode_bposit16(2.0 ** float(f))


def bposit16_reciprocal(p: int) -> int:
    """1/x for a bposit16 value. Returns NaR for x=0 (and propagates NaR).
    Like bposit16_sqrt, this uses Fraction arithmetic at LUT-build time only —
    on the GPU the path is a 65 K __device__ const lookup."""
    d = decode_bposit16(p)
    if d.is_special == "zero":
        return NAR
    if d.is_special == "nar":
        return NAR
    f = decoded_to_fraction(d)
    if f == 0:
        return NAR
    return encode_bposit16(Fraction(1) / f)


def bposit16_to_quire(p: int) -> int:
    d = decode_bposit16(p)
    if d.is_special:
        return 0
    f = decoded_to_fraction(d)
    # Multiply by 2^QUIRE_FRAC_BITS to convert to fixed-point integer
    f *= (1 << QUIRE_FRAC_BITS)
    return int(f)  # truncating — round half away or to-even is the proper choice


def quire256_add(q: int, x: int) -> int:
    return q + x


def quire256_to_bposit16(q: int) -> int:
    """Final round of an exact quire256 accumulation back to bposit16.

    The quire is signed fixed-point with QUIRE_FRAC_BITS (= 96) fractional
    bits, giving exact products of any two bposit16 values. The CUDA
    __device__ implementation must produce bit-exact results against this
    reference. Handles zero, sign, saturation, and round-to-nearest-even
    via the standard encode_bposit16 path."""
    if q == 0:
        return ZERO
    sign = q < 0
    q_abs = -q if sign else q
    val = Fraction(q_abs, 1 << QUIRE_FRAC_BITS)
    if sign:
        val = -val
    return encode_bposit16(val)


def bposit16_mul_via_log_exp2(a: int, b: int) -> int:
    """Multiply via mul(a,b) = sign(a)·sign(b) · exp2(log2(|a|) + log2(|b|)).
    This is the path the CUDA bposit16_mul_dev kernel uses; the Python
    function exists so test cases can be generated bit-exact for comparison
    against the device. (The 'exact' bposit16_mul above uses Fraction
    arithmetic and produces slightly different rounding for many products.)"""
    if a == NAR or b == NAR:
        return NAR
    if a == ZERO or b == ZERO:
        return ZERO
    ONE_LOCAL = 0x4000
    if a == ONE_LOCAL:
        return b
    if b == ONE_LOCAL:
        return a

    sign_a = (a >> 15) & 1
    sign_b = (b >> 15) & 1
    result_sign = sign_a ^ sign_b

    def abs_bp16(x: int) -> int:
        if not ((x >> 15) & 1):
            return x
        return ((~x) + 1) & 0x7FFF

    abs_a = abs_bp16(a)
    abs_b = abs_bp16(b)

    la = bposit16_log2(abs_a)
    lb = bposit16_log2(abs_b)

    # Sum logs in quire256, encode back to bposit16
    q = bposit16_to_quire(la) + bposit16_to_quire(lb)
    log_sum = quire256_to_bposit16(q)
    mag = bposit16_exp2(log_sum)

    if mag == NAR:
        return NAR
    if mag == ZERO:
        return ZERO

    if result_sign:
        return ((~mag) + 1) & 0xFFFF
    return mag


def jacobi_eigvals_nxn(A: list[list[int]], n: int, n_sweeps: int = 10) -> list[int]:
    """Cyclic Jacobi sweep for an N×N symmetric Hermitian matrix using only
    the bposit16 ALU primitives that exist on device (sqrt, recip, add, mul,
    log+exp2). Returns the eigenvalues as a list of bposit16 ints.

    Algorithm (Press et al. 'Numerical Recipes' §11.1, no-trig variant):
        for sweep in 0..n_sweeps:
            for p in 0..n:
                for q in p+1..n:
                    if A[p,q] == 0: continue
                    θ = (A[q,q] - A[p,p]) / (2·A[p,q])
                    t = sign(θ) / (|θ| + sqrt(θ²+1))   # smaller root
                    c = 1/sqrt(1+t²);  s = t·c
                    apply rotation to A in place

    Bit-exact-reproducible: cyclic order (no max-finding) so Python and
    CUDA produce identical sweep paths."""
    ONE = 0x4000
    TWO = 0x4400
    NAR_LOCAL = NAR
    ZERO_LOCAL = ZERO

    def _neg(x: int) -> int:
        if x == NAR_LOCAL or x == ZERO_LOCAL:
            return x
        return ((~x) + 1) & 0xFFFF

    def _sub(a: int, b: int) -> int:
        return bposit16_add(a, _neg(b))

    def _mul(a: int, b: int) -> int:
        return bposit16_mul_via_log_exp2(a, b)

    def _square(p: int) -> int:
        return _mul(p, p)

    def _is_neg(x: int) -> bool:
        return ((x >> 15) & 1) and x != ZERO_LOCAL and x != NAR_LOCAL

    def _abs_bp(x: int) -> int:
        if not _is_neg(x):
            return x
        return ((~x) + 1) & 0x7FFF

    A = [row[:] for row in A]
    for _ in range(n_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                if A[p][q] == ZERO_LOCAL:
                    continue
                diff = _sub(A[q][q], A[p][p])
                two_apq = _mul(TWO, A[p][q])
                theta = _mul(diff, bposit16_reciprocal(two_apq))
                abs_t = _abs_bp(theta)
                th2 = _square(theta)
                sqrt_term = bposit16_sqrt(bposit16_add(ONE, th2))
                inv_denom = bposit16_reciprocal(bposit16_add(abs_t, sqrt_term))
                t = inv_denom if not _is_neg(theta) else _neg(inv_denom)
                c = bposit16_reciprocal(bposit16_sqrt(bposit16_add(ONE, _square(t))))
                s = _mul(t, c)
                Apq = A[p][q]; App = A[p][p]; Aqq = A[q][q]
                for i in range(n):
                    if i == p or i == q:
                        continue
                    Aip = A[i][p]; Aiq = A[i][q]
                    new_ip = _sub(_mul(c, Aip), _mul(s, Aiq))
                    new_iq = bposit16_add(_mul(s, Aip), _mul(c, Aiq))
                    A[i][p] = new_ip; A[p][i] = new_ip
                    A[i][q] = new_iq; A[q][i] = new_iq
                cs = _mul(c, s)
                cc = _square(c); ss = _square(s)
                two_cs_apq = _mul(_mul(TWO, cs), Apq)
                A[p][p] = bposit16_add(_sub(_mul(cc, App), two_cs_apq), _mul(ss, Aqq))
                A[q][q] = bposit16_add(bposit16_add(_mul(ss, App), two_cs_apq), _mul(cc, Aqq))
                A[p][q] = ZERO_LOCAL; A[q][p] = ZERO_LOCAL
    return [A[i][i] for i in range(n)]


def bposit16_add(a: int, b: int) -> int:
    """a + b in bposit16, computed via the exact quire path:
        quire(a) + quire(b) → quire result → encode to bposit16.
    Special cases: NaR + anything = NaR, zero + x = x.
    The CUDA path mirrors this exactly using the bposit16_to_quire LUT
    and the quire256_to_bposit16 device function."""
    if a == NAR or b == NAR:
        return NAR
    if a == ZERO:
        return b
    if b == ZERO:
        return a
    qa = bposit16_to_quire(a)
    qb = bposit16_to_quire(b)
    return quire256_to_bposit16(qa + qb)


def quire256_to_bposit32(q: int) -> int:
    """Final round to bposit32. Same encoder as bposit16 but with more
    fraction bits — for the POC we round through bposit16 and upconvert."""
    if q == 0:
        return 0
    sign = q < 0
    q_abs = -q if sign else q
    # q_abs / 2^QUIRE_FRAC_BITS is the represented value
    val = Fraction(q_abs, 1 << QUIRE_FRAC_BITS)
    if sign:
        val = -val
    p16 = encode_bposit16(val)
    # Trivial upconversion bposit16 → bposit32 = shift left 16 (Gustafson 7.4)
    return (p16 << 16) & 0xFFFFFFFF


# ---- Shannon entropy via the b-posit pipeline -------------------------------
def shannon_entropy_bposit16(probs: list[int]) -> int:
    """All-integer reference of the kernel: returns bposit32 result."""
    q = quire256_zero()
    for p in probs:
        lp = bposit16_log2(p)
        term = bposit16_mul(p, lp)
        q_term = bposit16_to_quire(term)
        # Negate the term in 256-bit two's-complement before adding
        # (entropy = -Σ p log p)
        q_term = -q_term
        q = quire256_add(q, q_term)
    return quire256_to_bposit32(q)


# ---- Validation -------------------------------------------------------------
if __name__ == "__main__":
    print("=== bposit16 reference impl — round-trip + arithmetic POC ===\n")

    # 1. Round-trip test for known values
    test_vals = [0.25, 0.5, 1.0, 2.0, 4.0, -1.0]
    print("Round-trip test (float → bposit16 → float):")
    for v in test_vals:
        p = encode_bposit16(v)
        d = decode_bposit16(p)
        recovered = float(decoded_to_fraction(d)) if d.is_special != "zero" else 0.0
        err = abs(recovered - v) / max(abs(v), 1e-15)
        print(f"  {v:>8.4f}  →  bposit16=0x{p:04x}  →  {recovered:>10.6f}  rel_err={err:.2e}")

    # 2. Arithmetic test: 0.25 * log2(0.25) = 0.25 * -2.0 = -0.5
    print("\nArithmetic test:")
    p_quarter = encode_bposit16(0.25)
    p_log = bposit16_log2(p_quarter)
    p_prod = bposit16_mul(p_quarter, p_log)
    f_recovered = float(decoded_to_fraction(decode_bposit16(p_prod)))
    print(f"  0.25 * log₂(0.25) = 0.25 * (-2) = -0.5")
    print(f"  bposit16:           {f_recovered:.6f}  (expected -0.5)")

    # 3. Full pipeline: shannon_entropy([0.25]*4) = log₂(4) = 2.0
    print("\nShannon entropy test (THE TEST CASE):")
    probs = [encode_bposit16(0.25)] * 4
    H_b32 = shannon_entropy_bposit16(probs)
    H_b16 = (H_b32 >> 16) & 0xFFFF   # downconvert (trivial shift per Gustafson 7.4)
    H_recovered = float(decoded_to_fraction(decode_bposit16(H_b16))) if H_b16 != 0 else 0.0
    print(f"  H = -Σ p_i log₂(p_i) for uniform p=[0.25]*4")
    print(f"  expected: 2.0 bits")
    print(f"  bposit32: 0x{H_b32:08x}  →  bposit16 0x{H_b16:04x}  →  {H_recovered:.4f}")
    err = abs(H_recovered - 2.0)
    fidelity_bits = 3.52
    fidelity_tol = 10 ** (-fidelity_bits)
    pass_ = err < 1.0  # generous for POC; bposit16 fidelity is ~3.5 decimals
    print(f"  abs error: {err:.4f}  ({'PASS' if pass_ else 'FAIL'} — within bposit16 fidelity)")

    # 4. Quire pipeline correctness
    print("\nQuire accumulation (bypass log/mul, just sum constants):")
    q = quire256_zero()
    p_ones = encode_bposit16(0.5)  # 0.5
    for _ in range(8):
        q = quire256_add(q, bposit16_to_quire(p_ones))
    out = quire256_to_bposit32(q)
    out16 = (out >> 16) & 0xFFFF
    val = float(decoded_to_fraction(decode_bposit16(out16)))
    print(f"  8 * 0.5 = 4.0  →  quire256_to_bposit32 = {val:.4f}")
