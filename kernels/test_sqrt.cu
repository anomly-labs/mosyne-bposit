// test_sqrt.cu — validate the GPU bposit16 sqrt LUT.
//
// Provides bposit16_sqrt(p) for any positive p, as a 65 K × 2 byte LUT in
// __device__ const memory. The LUT is baked at host build time from the
// validated Python reference (which uses math.sqrt at LUT-build time only;
// no float ever reaches the device).
//
// Test: each input → expected output from Python reference.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_sqrt.cu -o test_sqrt

#include "bposit16_luts.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void sqrt_kernel(const bposit16_t* __restrict__ in,
                            bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bposit16_sqrt_lookup(in[idx]);
}

int main() {
    // Test cases from the Python reference. Each: input bposit16, expected output.
    struct Tcase { bposit16_t in; bposit16_t expected; const char* note; };
    // All "in" and "expected" values regenerated from the Python reference
    // (encode_bposit16 + bposit16_sqrt) — no hand math.
    Tcase cases[] = {
        {0x4800, 0x4400, "sqrt(4.0) = 2.0"},
        {0x3800, 0x3c00, "sqrt(0.25) = 0.5"},
        {0x4400, 0x41a8, "sqrt(2.0) ≈ 1.4141"},
        {0x5000, 0x4800, "sqrt(16.0) = 4.0"},
        {0x4000, 0x4000, "sqrt(1.0) = 1.0"},
        {0x3e00, 0x3eed, "sqrt(0.75) ≈ 0.8660  (LQG j=1/2 case: j(j+1)=3/4)"},
        {0x0000, 0x0000, "sqrt(0) = 0"},
        {0xc000, 0x8000, "sqrt(-1) = NaR"},
        {0x4900, 0x4478, "sqrt(5.0) ≈ 2.2360"},
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    bposit16_t h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = cases[i].in;

    bposit16_t* d_in;
    bposit16_t* d_out;
    cudaMalloc(&d_in, N * sizeof(bposit16_t));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_in, h_in, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    sqrt_kernel<<<1, 32>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== bposit16 sqrt LUT validation ===\n\n");
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
