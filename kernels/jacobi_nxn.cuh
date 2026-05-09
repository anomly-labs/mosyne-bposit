// jacobi_nxn.cuh — cyclic Jacobi eigendecomposition for small Hermitian
// bposit16 matrices. Lifted out of test_jacobi_nxn.cu in iter-40 so the
// von_neumann_entropy library kernel can dispatch n ≥ 3 through it.
//
// Mirrors bposit16_reference.jacobi_eigvals_nxn exactly: cyclic (p, q)
// order, numerically-stable no-trig rotation. Each call works on a
// single n×n matrix in registers / per-thread local memory.
//
// Requires bposit16_luts.cuh and bposit16_encode.cuh to be included
// first (for bposit16_log2_lookup, bposit16_recip_lookup,
// bposit16_sqrt_lookup, bposit16_add_dev, bposit16_mul_dev,
// bposit16_neg_dev).
//
// Helpers exposed (all `__device__ __forceinline__` so safe to include
// from multiple translation units): jac_bp16_is_neg, jac_bp16_abs,
// jac_bp16_sub, jac_bp16_square. Prefixed `jac_` to avoid colliding
// with any caller's local helpers of the same purpose.

#pragma once

#include <cstdint>

#ifndef BPOSIT16_ENCODE_NO_CUDA
#  include <cuda_runtime.h>
#endif

constexpr int JACOBI_MAX_N = 4;

__device__ __forceinline__ bool jac_bp16_is_neg(std::uint16_t x) {
    return ((x >> 15) & 1U) && x != 0x0000 && x != 0x8000;
}

__device__ __forceinline__ std::uint16_t jac_bp16_abs(std::uint16_t x) {
    if (!jac_bp16_is_neg(x)) return x;
    return ((~x) + 1U) & 0x7FFFU;
}

__device__ __forceinline__ std::uint16_t jac_bp16_sub(std::uint16_t a, std::uint16_t b) {
    return bposit16_add_dev(a, bposit16_neg_dev(b));
}

__device__ __forceinline__ std::uint16_t jac_bp16_square(std::uint16_t x) {
    return bposit16_mul_dev(x, x);
}

// Single Jacobi rotation in-place: zero A[p,q] = A[q,p] and update the
// rest of the matrix to preserve eigenvalues. Symmetric updates are
// applied to keep A Hermitian.
__device__ inline void jacobi_rotate_dev(std::uint16_t* A, int n, int p, int q) {
    constexpr std::uint16_t BP16_ZERO = 0x0000;
    constexpr std::uint16_t BP16_ONE  = 0x4000;
    constexpr std::uint16_t BP16_TWO  = 0x4400;

    std::uint16_t Apq = A[p * n + q];
    if (Apq == BP16_ZERO) return;

    std::uint16_t App = A[p * n + p];
    std::uint16_t Aqq = A[q * n + q];

    // θ = (A[q,q] − A[p,p]) / (2·A[p,q])
    std::uint16_t diff    = jac_bp16_sub(Aqq, App);
    std::uint16_t two_apq = bposit16_mul_dev(BP16_TWO, Apq);
    std::uint16_t theta   = bposit16_mul_dev(diff, bposit16_recip_lookup(two_apq));

    // t = sign(θ) / (|θ| + sqrt(θ² + 1))
    std::uint16_t abs_t  = jac_bp16_abs(theta);
    std::uint16_t th2    = jac_bp16_square(theta);
    std::uint16_t sqrt_t = bposit16_sqrt_lookup(bposit16_add_dev(BP16_ONE, th2));
    std::uint16_t inv_d  = bposit16_recip_lookup(bposit16_add_dev(abs_t, sqrt_t));
    std::uint16_t t = jac_bp16_is_neg(theta) ? bposit16_neg_dev(inv_d) : inv_d;

    // c = 1/sqrt(1 + t²), s = t·c
    std::uint16_t c = bposit16_recip_lookup(
        bposit16_sqrt_lookup(bposit16_add_dev(BP16_ONE, jac_bp16_square(t))));
    std::uint16_t s = bposit16_mul_dev(t, c);

    // Off-diagonal row/column updates (i ≠ p, q)
    for (int i = 0; i < n; ++i) {
        if (i == p || i == q) continue;
        std::uint16_t Aip = A[i * n + p];
        std::uint16_t Aiq = A[i * n + q];
        std::uint16_t new_ip = jac_bp16_sub(bposit16_mul_dev(c, Aip),
                                            bposit16_mul_dev(s, Aiq));
        std::uint16_t new_iq = bposit16_add_dev(bposit16_mul_dev(s, Aip),
                                                bposit16_mul_dev(c, Aiq));
        A[i * n + p] = new_ip;  A[p * n + i] = new_ip;
        A[i * n + q] = new_iq;  A[q * n + i] = new_iq;
    }

    // Diagonal updates
    std::uint16_t cs = bposit16_mul_dev(c, s);
    std::uint16_t cc = jac_bp16_square(c);
    std::uint16_t ss = jac_bp16_square(s);
    std::uint16_t two_cs_apq = bposit16_mul_dev(bposit16_mul_dev(BP16_TWO, cs), Apq);
    A[p * n + p] = bposit16_add_dev(
        jac_bp16_sub(bposit16_mul_dev(cc, App), two_cs_apq),
        bposit16_mul_dev(ss, Aqq));
    A[q * n + q] = bposit16_add_dev(
        bposit16_add_dev(bposit16_mul_dev(ss, App), two_cs_apq),
        bposit16_mul_dev(cc, Aqq));
    A[p * n + q] = BP16_ZERO;
    A[q * n + p] = BP16_ZERO;
}

// Compute eigenvalues of a single n×n Hermitian bposit16 matrix in
// place (cyclic Jacobi, fixed sweep count). Diagonal of the input is
// the eigenvalue array on return; off-diagonals are zeroed. Caller
// owns A and must pass n satisfying n <= JACOBI_MAX_N.
__device__ inline void jacobi_eigvals_dev(std::uint16_t* A, int n, int n_sweeps) {
    for (int sw = 0; sw < n_sweeps; ++sw) {
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                jacobi_rotate_dev(A, n, p, q);
            }
        }
    }
}
