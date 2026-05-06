// Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

// test_exp2.cu — validate the GPU bposit16 exp2 (2^x) LUT.
//
// Companion to log2_lut. Together they enable POSITIVE-OPERAND multiplication
// via mul(a, b) = exp2(log2(a) + log2(b)). Combined with sign handling on
// inputs, this gives us a full bposit16 multiply on the device — without
// needing the 8 GB direct mul LUT.
//
// 65 K × 2 bytes baked at host build time from the Python reference.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_exp2.cu -o test_exp2

#include "bposit16_luts.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void exp2_kernel(const bposit16_t* __restrict__ in,
                            bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bposit16_exp2_lookup(in[idx]);
}

int main() {
    // Test cases from the Python reference: encode_bposit16(...) inputs and
    // bposit16_exp2(...) outputs. No hand math.
    struct Tcase { bposit16_t in; bposit16_t expected; const char* note; };
    Tcase cases[] = {
        {0x0000, 0x4000, "exp2(0) = 1"},
        {0x4000, 0x4400, "exp2(1) = 2"},
        {0x4400, 0x4800, "exp2(2) = 4"},
        {0xc000, 0x3c00, "exp2(-1) = 0.5"},
        {0x3c00, 0x41a8, "exp2(0.5) ≈ 1.4140  (sqrt(2))"},
        {0x4c00, 0x6000, "exp2(8) = 256"},
        {0xb400, 0x2000, "exp2(-8) ≈ 0.0039"},
        {0x5600, 0x7f00, "exp2(48) = 2^48 (top of bposit16 range)"},
        {0x8000, 0x8000, "exp2(NaR) = NaR"},
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    bposit16_t h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = cases[i].in;

    bposit16_t* d_in;
    bposit16_t* d_out;
    cudaMalloc(&d_in, N * sizeof(bposit16_t));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_in, h_in, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    exp2_kernel<<<1, 32>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== bposit16 exp2 LUT validation ===\n\n");
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        bool ok = (h_out[i] == cases[i].expected);
        if (ok) ++pass;
        std::printf("  [%s] in=0x%04x  got=0x%04x  expected=0x%04x  %s\n",
                    ok ? "PASS" : "FAIL", cases[i].in, h_out[i], cases[i].expected, cases[i].note);
    }
    std::printf("\nSummary: %d/%d pass\n", pass, N);

    cudaFree(d_in); cudaFree(d_out);
    return (pass == N) ? 0 : 1;
}