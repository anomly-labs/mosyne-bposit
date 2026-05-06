// Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

// test_bposit16_mul.cu — Validate bposit16_mul_dev (a · b via log+exp2).
//
// Path: mul(a, b) = sign(a)·sign(b) · exp2(log2(|a|) + log2(|b|))
// Log-sum is accumulated in quire256, re-encoded to bposit16, then exp2.
// This is NOT bit-exact against decode→Fraction-mul→encode (1 ULP error
// possible from log/exp2 round-trip), but IS bit-exact against the same
// log+exp2 path implemented in Python — which is what we test here.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_bposit16_mul.cu -o test_mul

#include "bposit16_luts.cuh"
#include "bposit16_encode.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void mul_kernel(const bposit16_t* __restrict__ a_in,
                           const bposit16_t* __restrict__ b_in,
                           bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bposit16_mul_dev(a_in[idx], b_in[idx]);
}

int main() {
    using uint16_t = std::uint16_t;
    #include "test_cases_mul_inline.h"
    constexpr int N = sizeof(mul_cases) / sizeof(mul_cases[0]);

    bposit16_t h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = mul_cases[i].a;
        h_b[i] = mul_cases[i].b;
    }

    bposit16_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(bposit16_t));
    cudaMalloc(&d_b, N * sizeof(bposit16_t));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_a, h_a, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    mul_kernel<<<1, 32>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== bposit16_mul device function (log2 + quire + exp2) ===\n\n");
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        bool ok = (h_out[i] == mul_cases[i].expected);
        if (ok) ++pass;
        std::printf("  [%s] 0x%04x * 0x%04x = 0x%04x  (expected 0x%04x)  %s\n",
                    ok ? "PASS" : "FAIL",
                    mul_cases[i].a, mul_cases[i].b, h_out[i],
                    mul_cases[i].expected, mul_cases[i].note);
    }
    std::printf("\nSummary: %d/%d pass\n", pass, N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return (pass == N) ? 0 : 1;
}