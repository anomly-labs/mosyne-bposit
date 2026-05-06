// test_recip.cu — validate the GPU bposit16 reciprocal LUT.
//
// 65 K × 2-byte __device__ const table baked at build time from the Python
// reference (Fraction-arithmetic 1/x; no float ever reaches the device).
// This is the foundational op for bposit DIVISION and the Layer-4
// Schwarzschild g_rr = 1/(1 - 2M/r) computation.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_recip.cu -o test_recip

#include "bposit16_luts.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void recip_kernel(const bposit16_t* __restrict__ in,
                             bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bposit16_recip_lookup(in[idx]);
}

int main() {
    // Test cases. "in" / "expected" generated from the Python reference
    // (encode_bposit16 + bposit16_reciprocal) — no hand math.
    struct Tcase { bposit16_t in; bposit16_t expected; const char* note; };
    Tcase cases[] = {
        {0x4000, 0x4000, "1/1.0   = 1.0"},
        {0x4400, 0x3c00, "1/2.0   = 0.5"},
        {0x3c00, 0x4400, "1/0.5   = 2.0"},
        {0x4800, 0x3800, "1/4.0   = 0.25"},
        {0x3800, 0x4800, "1/0.25  = 4.0"},
        {0xc400, 0xbc00, "1/-2.0  = -0.5"},
        {0x3266, 0x4d00, "1/bposit16(0.1)    ≈ 10.5    (recip of bposit16-rounded 0.1)"},
        {0x251e, 0x5a40, "1/bposit16(0.01)   ≈ 100.0+  (recip of bposit16-rounded 0.01)"},
        {0x0000, 0x8000, "1/0     = NaR"},
        {0x8000, 0x8000, "1/NaR   = NaR"},
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    bposit16_t h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = cases[i].in;

    bposit16_t* d_in;
    bposit16_t* d_out;
    cudaMalloc(&d_in, N * sizeof(bposit16_t));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_in, h_in, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    recip_kernel<<<1, 32>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== bposit16 reciprocal LUT validation ===\n\n");
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
