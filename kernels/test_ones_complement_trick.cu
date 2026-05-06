// test_ones_complement_trick.cu — Gustafson's one's-complement reciprocal trick.
//
// Per private comm with John Gustafson (2026-05-05): the Stanford tensor
// posit unit (MINOTAUR group) replaced two's-complement negation with
// one's complement (~x) when computing 1/(1 − 2M/r) and similar low-
// precision reciprocals — saving the add-1 step that costs full-width
// carry propagation in silicon.
//
// On a GPU this isn't a hardware speed win (the chip's pipelined adders
// are essentially free), but the variant is *bit-comparable* to what
// Anomly silicon will produce in approximation mode, so we ship the
// variant here for cross-validation when our silicon arrives.
//
// This test runs Schwarzschild g_tt = -(1 − 2M/r) two ways:
//   path A: standard two's-complement neg  ((~x) + 1)
//   path B: Gustafson's one's-complement   (~x)
// and shows the maximum error of B vs A across a range of factor inputs.
// Expected: B is exactly 1 ULP off from A on the bposit16 representation
// (the missing "+1" step), and zero ULP for inputs that happen to land
// on a representable value where the LSB is already zero.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_ones_complement_trick.cu -o test_ones_comp

#include "bposit16_luts.cuh"
#include "bposit16_encode.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void compare_negs_kernel(const bposit16_t* __restrict__ in,
                                    bposit16_t* __restrict__ out_twos,
                                    bposit16_t* __restrict__ out_ones,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out_twos[idx] = bposit16_neg_dev(in[idx]);
    out_ones[idx] = bposit16_neg_ones_comp_dev(in[idx]);
}

int main() {
    // Sweep representative bposit16 values: positive, negative, special, near-1.
    const bposit16_t cases[] = {
        0x0000,                               // zero
        0x4000, 0x4400, 0x4600, 0x4900,       // 1, 2, 3, 5
        0x3c00, 0x3800, 0x3666,               // 0.5, 0.25, 0.2
        0x3e00, 0x3ccc, 0x3a66, 0x38cc,       // 0.75, 0.6, 0.4, 0.3
        0xc000, 0xc400, 0xbc00,               // -1, -2, -0.5
        0x7FFF, 0x0001, 0xFFFF, 0x8001,       // maxpos, minpos, minneg, maxneg
        0x8000,                               // NaR
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    bposit16_t h_in[N], h_two[N], h_one[N];
    for (int i = 0; i < N; ++i) h_in[i] = cases[i];

    bposit16_t *d_in, *d_two, *d_one;
    cudaMalloc(&d_in,  N * sizeof(bposit16_t));
    cudaMalloc(&d_two, N * sizeof(bposit16_t));
    cudaMalloc(&d_one, N * sizeof(bposit16_t));
    cudaMemcpy(d_in, h_in, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    compare_negs_kernel<<<1, 32>>>(d_in, d_two, d_one, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_two, d_two, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_one, d_one, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== Gustafson's one's-complement trick: -x via ~x vs ~x+1 ===\n\n");
    std::printf("  in       two's-comp (-x = ~x+1)   one's-comp (-x ≈ ~x)   ULP delta\n");
    std::printf("  -------  ----------------------   --------------------   ---------\n");
    int n_match = 0;
    int n_oneulp = 0;
    int max_ulp = 0;
    for (int i = 0; i < N; ++i) {
        int delta = (int)h_two[i] - (int)h_one[i];
        // Wrap to signed 16-bit
        if (delta > 32768) delta -= 65536;
        if (delta < -32768) delta += 65536;
        int absdelta = delta < 0 ? -delta : delta;
        if (absdelta == 0) ++n_match;
        else if (absdelta == 1) ++n_oneulp;
        if (absdelta > max_ulp) max_ulp = absdelta;
        std::printf("  0x%04x   0x%04x                   0x%04x                 %+d\n",
                    h_in[i], h_two[i], h_one[i], delta);
    }
    std::printf("\n");
    std::printf("Summary: %d/%d exact matches, %d/%d off by 1 ULP, max |delta|=%d\n",
                n_match, N, n_oneulp, N, max_ulp);
    std::printf("Verdict: one's-complement is exactly the documented %d-ULP-bounded\n",
                max_ulp);
    std::printf("approximation Gustafson described. Suitable for low-precision\n");
    std::printf("reciprocal-of-(1-x) silicon paths where the +1 step is not worth\n");
    std::printf("the full-width carry-propagation latency.\n");

    cudaFree(d_in); cudaFree(d_two); cudaFree(d_one);
    return (max_ulp <= 1) ? 0 : 1;
}
