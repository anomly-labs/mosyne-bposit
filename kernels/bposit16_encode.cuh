// bposit16_encode.cuh — quire256 + bposit16 encode/add/mul device functions.
//
// Self-contained header for the 5-DST CUDA toolkit. Provides:
//
//   struct quire256                              4× uint64 (256-bit signed FP)
//   q256_zero / is_zero / sign / negate          basic 256-bit ops
//   q256_find_msb / get_bit / extract_bits       bit-level introspection
//   q256_add                                     256-bit signed add
//   quire256_to_bposit16(quire256)               keystone encoder, bit-exact
//                                                vs. Python reference
//   bposit16_to_quire256_dev(bposit16_t)         bposit16 → quire via LUT
//   bposit16_add_dev(a, b)                       a + b in bposit16 (exact path)
//   bposit16_mul_pos_dev(a, b)                   |a|·|b| via log2+exp2
//   bposit16_mul_dev(a, b)                       a·b with sign handling
//
// Requires bposit16_luts.cuh to be included for the LUTs.

#pragma once

#include <cstdint>

#ifndef BPOSIT16_ENCODE_NO_CUDA
#  include <cuda_runtime.h>
#endif

// =============================================================================
// quire256 — 256-bit signed fixed-point with QUIRE_FRAC_BITS = 96 fraction bits
// w[0] = least-significant 64 bits; w[3] = most-significant.
// =============================================================================
struct quire256 { std::uint64_t w[4]; };

__device__ __forceinline__ quire256 q256_zero() { return {{0,0,0,0}}; }

__device__ __forceinline__ bool q256_is_zero(const quire256& q) {
    return (q.w[0] | q.w[1] | q.w[2] | q.w[3]) == 0;
}

__device__ __forceinline__ bool q256_sign(const quire256& q) {
    return (q.w[3] >> 63) & 1ULL;
}

__device__ __forceinline__ quire256 q256_negate(const quire256& q) {
    quire256 r;
    r.w[0] = ~q.w[0]; r.w[1] = ~q.w[1]; r.w[2] = ~q.w[2]; r.w[3] = ~q.w[3];
    std::uint64_t carry = 1ULL;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        std::uint64_t s = r.w[i] + carry;
        carry = (s < r.w[i]) ? 1ULL : 0ULL;
        r.w[i] = s;
    }
    return r;
}

__device__ __forceinline__ quire256 q256_add(quire256 a, quire256 b) {
    quire256 r;
    std::uint64_t carry = 0ULL;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        std::uint64_t s  = a.w[i] + b.w[i];
        std::uint64_t c1 = (s < a.w[i]) ? 1ULL : 0ULL;
        std::uint64_t s2 = s + carry;
        std::uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
        r.w[i] = s2;
        carry = c1 | c2;
    }
    return r;
}

__device__ __forceinline__ int q256_find_msb(const quire256& q) {
    if (q.w[3]) return 192 + (63 - __clzll(q.w[3]));
    if (q.w[2]) return 128 + (63 - __clzll(q.w[2]));
    if (q.w[1]) return  64 + (63 - __clzll(q.w[1]));
    if (q.w[0]) return       (63 - __clzll(q.w[0]));
    return -1;
}

__device__ __forceinline__ bool q256_get_bit(const quire256& q, int pos) {
    if (pos < 0 || pos > 255) return false;
    return (q.w[pos >> 6] >> (pos & 63)) & 1ULL;
}

__device__ __forceinline__ std::uint32_t q256_extract_bits(
    const quire256& q, int lo, int hi)
{
    int width = hi - lo + 1;
    if (width <= 0) return 0;
    std::uint32_t result = 0;
    for (int i = 0; i < width; ++i) {
        if (q256_get_bit(q, lo + i)) result |= (1U << i);
    }
    return result;
}

__device__ __forceinline__ bool q256_any_below(const quire256& q, int pos) {
    if (pos <= 0) return false;
    int word = pos >> 6;
    int bit = pos & 63;
    for (int i = 0; i < word; ++i) {
        if (q.w[i]) return true;
    }
    if (bit > 0) {
        std::uint64_t mask = (bit == 64) ? ~0ULL : ((1ULL << bit) - 1);
        if (q.w[word] & mask) return true;
    }
    return false;
}

// =============================================================================
// quire256_to_bposit16 — bit-exact against Python bposit16_reference.
// useed = 2^(2^eS) = 256.   value = useed^k · 2^e · (1.frac) = 2^(8k+e)·(1.frac)
// =============================================================================
__device__ inline std::uint16_t quire256_to_bposit16(quire256 q) {
    constexpr int QUIRE_FRAC_BITS = 96;
    constexpr std::uint16_t BP16_ZERO = 0x0000;
    constexpr std::uint16_t BP16_NAR  = 0x8000;
    constexpr std::uint16_t MAXPOS    = 0x7FFF;
    constexpr std::uint16_t MAXNEG    = 0x8001;
    constexpr std::uint16_t MINPOS    = 0x0001;
    constexpr std::uint16_t MINNEG    = 0xFFFF;

    if (q256_is_zero(q)) return BP16_ZERO;

    bool sign = q256_sign(q);
    if (sign) q = q256_negate(q);

    int msb = q256_find_msb(q);
    if (msb < 0) return BP16_ZERO;

    int scale = msb - QUIRE_FRAC_BITS;
    if (scale >  48) return sign ? MAXNEG : MAXPOS;
    if (scale < -48) return sign ? MINNEG : MINPOS;

    // Floor div by 8 → regime k; exp e in [0,7]
    int k, e;
    if (scale >= 0) {
        k = scale >> 3;
        e = scale - (k << 3);
    } else {
        int abs_s = -scale;
        int abs_k = abs_s >> 3;
        int abs_e = abs_s - (abs_k << 3);
        if (abs_e == 0) {
            k = -abs_k;
            e = 0;
        } else {
            k = -abs_k - 1;
            e = 8 - abs_e;
        }
    }

    std::uint32_t magnitude = 0;
    int top_pos;
    int regime_bits;
    if (k >= 0) {
        int ones_count = k + 1;
        if (ones_count > 14) ones_count = 14;
        regime_bits = ones_count + 1;
        if (regime_bits > 15) regime_bits = 15;
        magnitude |= ((1U << ones_count) - 1U) << (15 - ones_count);
        top_pos = 14 - regime_bits;
    } else {
        int zeros_count = -k;
        if (zeros_count >= 15) zeros_count = 14;
        regime_bits = zeros_count + 1;
        int term_pos = 14 - zeros_count;
        magnitude |= 1U << term_pos;
        top_pos = term_pos - 1;
    }

    int exp_bits_avail = top_pos + 1;
    int exp_bits = (exp_bits_avail >= 3) ? 3 : (exp_bits_avail > 0 ? exp_bits_avail : 0);
    if (exp_bits > 0) {
        std::uint32_t e_aligned = (std::uint32_t)e >> (3 - exp_bits);
        magnitude |= e_aligned << (top_pos - exp_bits + 1);
        top_pos -= exp_bits;
    }

    // NOTE: Python reference (encode_bposit16) currently TRUNCATES at the
    // 15-bit boundary (its tie-break is a `pass` placeholder per the source
    // comment). To stay bit-exact against the reference and the baked LUTs,
    // this encoder also truncates. Upgrading both to spec-correct round-to-
    // nearest-even (Gustafson Ch.7) is a future tick.
    int frac_bits = (top_pos >= 0) ? (top_pos + 1) : 0;
    if (frac_bits > 0) {
        int hi = msb - 1;
        int lo = msb - frac_bits;
        std::uint32_t frac_field;
        if (lo >= 0) {
            frac_field = q256_extract_bits(q, lo, hi);
        } else {
            frac_field = q256_extract_bits(q, 0, hi) << (-lo);
        }
        magnitude |= frac_field;
    }

    if (magnitude >= 0x8000U) magnitude = 0x7FFFU;

    if (sign) magnitude = ((~magnitude) + 1U) & 0x7FFFU;
    std::uint16_t result = ((std::uint16_t)(sign ? 1 : 0) << 15) | (std::uint16_t)magnitude;

    if (result == BP16_NAR && !sign) return MAXPOS;
    return result;
}

// =============================================================================
// bposit16 → quire256 (via LUT). Requires bposit16_to_quire_lut from the
// generated bposit16_luts.cuh.
// =============================================================================
__device__ __forceinline__ quire256 bposit16_to_quire256_dev(std::uint16_t p) {
    quire256 q;
    q.w[0] = bposit16_to_quire_lut[p][0];
    q.w[1] = bposit16_to_quire_lut[p][1];
    q.w[2] = bposit16_to_quire_lut[p][2];
    q.w[3] = bposit16_to_quire_lut[p][3];
    return q;
}

// =============================================================================
// bposit16 unary negate. -x in posit = two's complement of all 16 bits.
// (Sign bit + magnitude both flip via the standard 2's-complement identity.)
// =============================================================================
__device__ __forceinline__ std::uint16_t bposit16_neg_dev(std::uint16_t x) {
    constexpr std::uint16_t NAR  = 0x8000;
    constexpr std::uint16_t ZERO = 0x0000;
    if (x == NAR)  return NAR;
    if (x == ZERO) return ZERO;
    return ((~x) + 1U) & 0xFFFFU;
}

// =============================================================================
// One's-complement negate — Gustafson's hardware trick (private comm 2026-05-05).
//
// In silicon, two's complement requires an add-1 step with full-width carry
// propagation, which costs latency and energy. One's complement is purely
// local bit flips. The result is exact except for a 1-ULP offset at the
// LSB; for low-precision applications (decoder logic, recip-of-1-minus-x)
// the savings are worth the bounded inaccuracy.
//
// On the GPU this isn't a speed win (the chip's adders are pipelined and
// cheap), but the variant is provided here:
//   1. as a faithful port of the Stanford-team trick Gustafson described, and
//   2. so future Anomly silicon can be modelled bit-exactly with the same
//      ALU running in this approximation mode (compare-against-reference
//      validation when the chip arrives).
// =============================================================================
__device__ __forceinline__ std::uint16_t bposit16_neg_ones_comp_dev(std::uint16_t x) {
    constexpr std::uint16_t NAR  = 0x8000;
    constexpr std::uint16_t ZERO = 0x0000;
    if (x == NAR)  return NAR;
    if (x == ZERO) return ZERO;
    return (~x) & 0xFFFFU;   // no carry propagation
}

// =============================================================================
// bposit16 ADD on device — exact via quire256.
// =============================================================================
__device__ __forceinline__ std::uint16_t bposit16_add_dev(
    std::uint16_t a, std::uint16_t b)
{
    constexpr std::uint16_t NAR  = 0x8000;
    constexpr std::uint16_t ZERO = 0x0000;
    if (a == NAR || b == NAR) return NAR;
    if (a == ZERO) return b;
    if (b == ZERO) return a;
    quire256 qa = bposit16_to_quire256_dev(a);
    quire256 qb = bposit16_to_quire256_dev(b);
    return quire256_to_bposit16(q256_add(qa, qb));
}

// =============================================================================
// bposit16 MUL on device — via log2+exp2. Sign handled explicitly:
//   mul(a, b) = sign(a)·sign(b) · exp2(log2(|a|) + log2(|b|))
// where the log-sum is accumulated in quire256 and re-encoded to bposit16
// before exp2 lookup.
// =============================================================================
__device__ __forceinline__ std::uint16_t bposit16_mul_dev(
    std::uint16_t a, std::uint16_t b)
{
    constexpr std::uint16_t NAR  = 0x8000;
    constexpr std::uint16_t ZERO = 0x0000;
    constexpr std::uint16_t ONE  = 0x4000;
    if (a == NAR || b == NAR) return NAR;
    if (a == ZERO || b == ZERO) return ZERO;
    if (a == ONE) return b;
    if (b == ONE) return a;

    bool sign_a = (a >> 15) & 1;
    bool sign_b = (b >> 15) & 1;
    bool result_sign = sign_a ^ sign_b;

    // Take absolute value: bposit16 |x| = (x ^ -sign) + sign  (twos-complement)
    // For posit, magnitude is encoded via twos-complement of the lower 15 bits
    // when sign=1. The simpler path: |x| = (x == NAR ? NAR : (x[15] ? -x : x))
    // But we just need bp16 representations of |a|,|b| for log2 lookup.
    // For positive: |x| = x. For negative: |x| = ((~x + 1) & 0xFFFF) but with
    // sign bit cleared.
    auto abs_bp16 = [](std::uint16_t x) -> std::uint16_t {
        if (((x >> 15) & 1) == 0) return x;
        std::uint16_t mag = ((~x) + 1U) & 0x7FFFU;
        return mag;
    };
    std::uint16_t abs_a = abs_bp16(a);
    std::uint16_t abs_b = abs_bp16(b);

    std::uint16_t la = bposit16_log2_lookup(abs_a);
    std::uint16_t lb = bposit16_log2_lookup(abs_b);

    quire256 q = q256_add(bposit16_to_quire256_dev(la),
                          bposit16_to_quire256_dev(lb));
    std::uint16_t log_sum = quire256_to_bposit16(q);
    std::uint16_t mag = bposit16_exp2_lookup(log_sum);

    if (mag == NAR) return NAR;
    if (mag == ZERO) return ZERO;

    if (result_sign) {
        // Apply negative sign via twos-complement of magnitude
        std::uint16_t neg_mag = ((~mag) + 1U) & 0xFFFFU;
        return neg_mag;
    }
    return mag;
}
