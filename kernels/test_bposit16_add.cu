// test_bposit16_add.cu — Validate bposit16_add_dev (a + b via quire256).
//
// The exact path: a + b = quire256_to_bposit16(quire(a) + quire(b)).
// With this primitive, posit's quire-exact accumulation property gives us
// addition with NO intermediate rounding error — the only rounding happens
// at the final encode step.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          test_bposit16_add.cu -o test_add

#include "bposit16_luts.cuh"
#include "bposit16_encode.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

using bposit16_t = std::uint16_t;

__global__ void add_kernel(const bposit16_t* __restrict__ a_in,
                           const bposit16_t* __restrict__ b_in,
                           bposit16_t* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = bposit16_add_dev(a_in[idx], b_in[idx]);
}

int main() {
    using uint16_t = std::uint16_t;
    #include "test_cases_add_inline.h"
    constexpr int N = sizeof(add_cases) / sizeof(add_cases[0]);

    bposit16_t h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = add_cases[i].a;
        h_b[i] = add_cases[i].b;
    }

    bposit16_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(bposit16_t));
    cudaMalloc(&d_b, N * sizeof(bposit16_t));
    cudaMalloc(&d_out, N * sizeof(bposit16_t));
    cudaMemcpy(d_a, h_a, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(bposit16_t), cudaMemcpyHostToDevice);
    add_kernel<<<1, 32>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(bposit16_t), cudaMemcpyDeviceToHost);

    std::printf("=== bposit16_add device function (exact via quire256) ===\n\n");
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        bool ok = (h_out[i] == add_cases[i].expected);
        if (ok) ++pass;
        std::printf("  [%s] 0x%04x + 0x%04x = 0x%04x  (expected 0x%04x)  %s\n",
                    ok ? "PASS" : "FAIL",
                    add_cases[i].a, add_cases[i].b, h_out[i],
                    add_cases[i].expected, add_cases[i].note);
    }
    std::printf("\nSummary: %d/%d pass\n", pass, N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return (pass == N) ? 0 : 1;
}
