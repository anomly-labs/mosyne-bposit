// test_quire_to_bp16.cu — Validate the CUDA quire256_to_bposit16 device function.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_quire_to_bp16.cu -o test_q2bp16

#include "bposit16_luts.cuh"
#include "bposit16_encode.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void encode_kernel(const quire256* __restrict__ in,
                              bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = quire256_to_bposit16(in[idx]);
}

int main() {
    using uint64_t = std::uint64_t;
    using uint16_t = std::uint16_t;
    #include "test_cases_quire_inline.h"
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    quire256 h_in[N];
    bposit16_t h_out[N];
    for (int i = 0; i < N; ++i) {
        h_in[i].w[0] = cases[i].w[0];
        h_in[i].w[1] = cases[i].w[1];
        h_in[i].w[2] = cases[i].w[2];
        h_in[i].w[3] = cases[i].w[3];
    }

    quire256* d_in;
    bposit16_t* d_out;
    cudaMalloc(&d_in, N * sizeof(quire256));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_in, h_in, N * sizeof(quire256), cudaMemcpyHostToDevice);
    encode_kernel<<<1, 32>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== quire256 → bposit16 device encoder ===\n\n");
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        bool ok = (h_out[i] == cases[i].expected);
        if (ok) ++pass;
        std::printf("  [%s] got=0x%04x  expected=0x%04x  %s\n",
                    ok ? "PASS" : "FAIL", h_out[i], cases[i].expected, cases[i].note);
    }
    std::printf("\nSummary: %d/%d pass\n", pass, N);

    cudaFree(d_in); cudaFree(d_out);
    return (pass == N) ? 0 : 1;
}
