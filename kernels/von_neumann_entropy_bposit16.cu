// von_neumann_entropy_bposit16.cu — second 5-DST → RTX 5090 kernel.
//
// Spec: 5DST_v2_API_SPEC.h, layer 1 (quantum information):
//     bposit32_t von_neumann_entropy(const bposit16_t* rho, int n);
//
//     S(ρ) = -Tr(ρ log₂ ρ) = -Σ_i λ_i log₂(λ_i)
//
// where λ_i are the eigenvalues of the n×n Hermitian density matrix ρ.
// Architectural compliance: integer-only over bposit16 bit patterns, no
// IEEE float in the device path. Eigenvalue computation is closed-form
// for n=2 (the exact same numerically-stable form used in
// test_von_neumann_2x2.cu, which is already validated against the
// canonical 2×2 cases — Bell-state reduced ρ, mixed states).
//
// Coverage status:
//   n = 1:    trivial — pure 1-state ρ has λ = {1}, S = 0.
//   n = 2:    closed-form eigenvalues + shannon-style accumulation.
//             Closes single-qubit reduced density matrices.
//   n ∈ 3, 4: cyclic Jacobi via jacobi_nxn.cuh (lifted from
//             test_jacobi_nxn.cu in iter-40), then shannon-style
//             accumulation of -Σ λ log₂ λ. JACOBI_MAX_N = 4.
//   n ≥ 5:    returns -2 (would need Jacobi MAX_N raised + revalidation).
//
// The diagonal case (any n) is already covered by shannon_entropy on
// the diagonal entries; callers with already-diagonalised ρ should call
// shannon_entropy_bposit16_cuda directly. This entry point is for
// "I have a Hermitian density matrix; give me its entropy" — in which
// case the eigendecomposition is part of the computation.
//
// Build:
//     nvcc -O3 -std=c++17 -arch=sm_120 \
//          -shared -Xcompiler -fPIC \
//          von_neumann_entropy_bposit16.cu \
//              -o libvon_neumann_entropy_bposit.so

#include "bposit16_luts.cuh"
#include "bposit16_encode.cuh"
#include "jacobi_nxn.cuh"   // jacobi_eigvals_dev, JACOBI_MAX_N

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;
using bposit32_t = std::uint32_t;

constexpr bposit16_t BP16_ZERO = 0x0000;
constexpr bposit16_t BP16_HALF = 0x3c00;
constexpr bposit16_t BP16_NAR  = 0x8000;
constexpr bposit32_t BP32_ZERO = 0x00000000U;
constexpr bposit32_t BP32_NAR  = 0x80000000U;

__device__ __forceinline__ bposit16_t bp16_square(bposit16_t x) {
    return bposit16_mul_dev(x, x);
}

// Trivial: ρ = [1] is a pure state, S = 0 by definition.
__global__ void vn_n1_kernel(bposit32_t* __restrict__ out) {
    if (threadIdx.x == 0) *out = BP32_ZERO;
}

// 2×2 case. Closed-form eigenvalues:
//     ρ = [[a, b],
//          [b, d]]
//     λ_± = (a+d)/2 ± sqrt(((a-d)/2)^2 + b^2)
//
// then S = -Σ_± λ_± log₂(λ_±) accumulated in quire256, encoded to bposit32
// via the native quire256_to_bposit32 from bposit16_encode.cuh.
__global__ void vn_n2_kernel(
    bposit16_t a, bposit16_t b, bposit16_t d,
    bposit32_t* __restrict__ out)
{
    if (threadIdx.x != 0) return;

    bposit16_t mid       = bposit16_mul_dev(BP16_HALF, bposit16_add_dev(a, d));
    bposit16_t diff_half = bposit16_mul_dev(BP16_HALF,
                              bposit16_add_dev(a, bposit16_neg_dev(d)));
    bposit16_t disc_sq   = bposit16_add_dev(bp16_square(diff_half), bp16_square(b));
    bposit16_t disc      = bposit16_sqrt_lookup(disc_sq);
    bposit16_t lp        = bposit16_add_dev(mid, disc);
    bposit16_t lm        = bposit16_add_dev(mid, bposit16_neg_dev(disc));

    quire256 q = q256_zero();
    bposit16_t lams[2] = {lp, lm};
    bool nar = false;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        bposit16_t lam = lams[j];
        if (lam == BP16_ZERO) continue;
        if (lam == BP16_NAR) { nar = true; break; }
        // Negative eigenvalue means the input ρ wasn't PSD → mark NaR.
        if ((lam >> 15) & 1U) { nar = true; break; }
        bposit16_t l2    = bposit16_log2_lookup(lam);
        bposit16_t term  = bposit16_mul_dev(lam, l2);   // λ · log₂(λ)  (≤ 0)
        bposit16_t nterm = bposit16_neg_dev(term);      // -(λ · log₂(λ)) (≥ 0)
        q = q256_add(q, bposit16_to_quire256_dev(nterm));
    }
    *out = nar ? BP32_NAR : quire256_to_bposit32(q);
}

// Jacobi path for n ∈ {3, 4}. One thread does the whole computation:
// copy ρ into local n×n storage, Jacobi-rotate to a diagonal, then
// accumulate -Σ λ log₂ λ into a quire256 and encode to bposit32.
__global__ void vn_jacobi_kernel(
    const bposit16_t* __restrict__ d_rho, int n, int n_sweeps,
    bposit32_t* __restrict__ out)
{
    if (threadIdx.x != 0) return;

    bposit16_t A[JACOBI_MAX_N * JACOBI_MAX_N];
    for (int i = 0; i < n * n; ++i) A[i] = d_rho[i];

    jacobi_eigvals_dev(A, n, n_sweeps);

    quire256 q = q256_zero();
    bool nar = false;
    for (int i = 0; i < n; ++i) {
        bposit16_t lam = A[i * n + i];
        if (lam == BP16_ZERO) continue;
        if (lam == BP16_NAR) { nar = true; break; }
        if ((lam >> 15) & 1U) { nar = true; break; }
        bposit16_t l2    = bposit16_log2_lookup(lam);
        bposit16_t term  = bposit16_mul_dev(lam, l2);   // ≤ 0
        bposit16_t nterm = bposit16_neg_dev(term);      // ≥ 0
        q = q256_add(q, bposit16_to_quire256_dev(nterm));
    }
    *out = nar ? BP32_NAR : quire256_to_bposit32(q);
}

// Public extern "C" entry point.
//
// rho is a flat row-major n×n array of bposit16 values. For n=2 only
// the [0,0], [0,1], and [1,1] entries are read (we assume Hermiticity);
// for n ≥ 3 the full n² array is consumed as Hermitian.
// Returns 0 on ok, non-zero on error:
//   -1  n < 1
//   -2  n > JACOBI_MAX_N (currently 4)
//   -3  cuda kernel launch error
//  -11  cudaMemcpy failed (n=2 path only)
extern "C" int von_neumann_entropy_bposit16_cuda(
    const bposit16_t* d_rho, int n, bposit32_t* d_out)
{
    if (n < 1) return -1;
    if (n > JACOBI_MAX_N) return -2;

    if (n == 1) {
        vn_n1_kernel<<<1, 1>>>(d_out);
        return cudaGetLastError() == cudaSuccess ? 0 : -3;
    }

    if (n == 2) {
        // Pull the three independent entries from the device-side
        // matrix and launch the closed-form kernel.
        bposit16_t h_rho[4];
        cudaError_t err = cudaMemcpy(h_rho, d_rho, 4 * sizeof(bposit16_t),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -11;
        vn_n2_kernel<<<1, 32>>>(h_rho[0], h_rho[1], h_rho[3], d_out);
        return cudaGetLastError() == cudaSuccess ? 0 : -3;
    }

    // n ∈ {3, 4}: dispatch through Jacobi.
    constexpr int N_SWEEPS = 10;  // matches test_jacobi_nxn.cu default
    vn_jacobi_kernel<<<1, 32>>>(d_rho, n, N_SWEEPS, d_out);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
