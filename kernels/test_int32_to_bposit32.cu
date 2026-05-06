// Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

// test_int32_to_bposit32.cu — CUDA-native bposit32 encoder.
//
// Takes int32 fixed-point values (semantic = int32 / 2^scale_shift) and
// emits the corresponding bposit32 encoding. This is the post-decode
// stage that closes the bposit-in / bposit-out loop on the IMMA-MERA
// kernel: GEMM produces int32, this encoder turns it into bposit32 per
// the v2 spec.
//
// Build:
//     nvcc -O3 -std=c++17 -arch=sm_120 test_int32_to_bposit32.cu -o test_int32
//     ./test_int32
//
// Validation set (matches the Python reference outputs):
//
//   semantic value  scale_shift  int32 input    expected bposit32
//   ─────────────────────────────────────────────────────────────
//   4.0             12           16384          0x48000000
//   2.0             12            8192          0x44000000
//   1.0             12            4096          0x40000000
//   0.5             12            2048          0x3c000000
//   0.25            12            1024          0x38000000
//   16.0            12           65536          0x50000000

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int ES_32 = 3;

// =============================================================================
// CUDA-native bposit32 encoder.
// Input:  abs_int (uint32_t, magnitude only) + scale_shift (frac bits)
// Output: 31-bit unsigned bposit32 field (sign applied separately)
// =============================================================================
__device__ __forceinline__ uint32_t encode_unsigned_32(
    uint32_t mantissa_bits,    // top 31 bits hold frac (bit 30 = 0.5, bit 29 = 0.25, ...)
    int total_e)               // binary exponent, range [-48, 48]
{
    if (total_e > 48)  return 0x7FFFFFFFu;  // +maxPos
    if (total_e < -48) return 0x00000001u;  // +minPos

    int k = total_e >> 3;   // arithmetic shift (signed). useed = 2^(2^3) = 256, log_useed = 3
    int e = total_e & 7;    // 3 exp bits

    uint32_t out = 0;
    int n_bits = 0;         // bits packed into `out` so far (from MSB)

    // Regime
    if (k >= 0) {
        int ones_to_add = k + 1;
        for (int i = 0; i < ones_to_add && n_bits < 31; ++i) {
            out = (out << 1) | 1u;
            ++n_bits;
        }
        if (n_bits < 31) {  // terminator 0
            out = (out << 1);
            ++n_bits;
        }
    } else {
        int zeros_to_add = -k;
        for (int i = 0; i < zeros_to_add && n_bits < 31; ++i) {
            out = (out << 1);
            ++n_bits;
        }
        if (n_bits < 31) {  // terminator 1
            out = (out << 1) | 1u;
            ++n_bits;
        }
    }

    // Exponent: ES_32 = 3 bits, MSB-first
    #pragma unroll
    for (int i = ES_32 - 1; i >= 0; --i) {
        if (n_bits >= 31) break;
        out = (out << 1) | ((e >> i) & 1u);
        ++n_bits;
    }

    // Fraction bits from mantissa_bits, bit 30 first
    int frac_pos = 30;
    while (n_bits < 31 && frac_pos >= 0) {
        out = (out << 1) | ((mantissa_bits >> frac_pos) & 1u);
        ++n_bits;
        --frac_pos;
    }

    out <<= (31 - n_bits);
    return out & 0x7FFFFFFFu;
}

__device__ __forceinline__ uint32_t int_to_bposit32(int32_t I, int scale_shift) {
    if (I == 0) return 0;
    int sign = (I < 0) ? 1 : 0;
    uint32_t abs_I = sign ? (uint32_t)(-I) : (uint32_t)I;
    // Find msb position (0..31)
    int nlz = __clz(abs_I);
    int msb_pos = 31 - nlz;
    int total_e = msb_pos - scale_shift;
    // Mantissa bits in [0, 1) form, 31-bit field with bit 30 = ½, bit 29 = ¼, …
    // abs_I = 2^msb_pos + frac_part, where 0 ≤ frac_part < 2^msb_pos.
    // shift left by (31 - msb_pos) brings frac_part into bits 30..0 with the
    // implicit leading 1 in bit 31.
    uint32_t shifted = (msb_pos >= 31) ? abs_I : (abs_I << (31 - msb_pos));
    uint32_t mantissa_bits = shifted & 0x7FFFFFFFu;

    uint32_t field = encode_unsigned_32(mantissa_bits, total_e);
    if (sign) field = ((~field) + 1) & 0x7FFFFFFFu;
    return ((uint32_t(sign) << 31) | field) & 0xFFFFFFFFu;
}

// =============================================================================
// Kernel
// =============================================================================
__global__ void int32_to_bposit32_kernel(
    const int32_t* __restrict__ in,
    uint32_t*       __restrict__ out,
    int n,
    int scale_shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = int_to_bposit32(in[idx], scale_shift);
}

// =============================================================================
// Host driver: validates against the Python reference values.
// =============================================================================
int main() {
    struct Tcase { int32_t input; int scale_shift; uint32_t expected; const char* note; };
    // To regenerate expected values for arbitrary inputs, run:
    //   .venv/bin/python -c "
    //   import sys; sys.path.insert(0, 'deploy/5dst_cuda')
    //   from bposit16_reference import encode_bposit32
    //   from fractions import Fraction
    //   x = INT32_INPUT
    //   shift = SCALE_SHIFT
    //   print(hex(encode_bposit32(Fraction(x, 1 << shift))))
    Tcase cases[] = {
        {16384, 12, 0x48000000, "4.0    = bposit32(4.0)"},
        { 8192, 12, 0x44000000, "2.0    = bposit32(2.0)"},
        { 4096, 12, 0x40000000, "1.0    = bposit32(1.0)"},
        { 2048, 12, 0x3c000000, "0.5    = bposit32(0.5)"},
        { 1024, 12, 0x38000000, "0.25   = bposit32(0.25)"},
        {65536, 12, 0x50000000, "16.0   = bposit32(16.0)"},
        {-16384, 12, 0xb8000000, "-4.0   = bposit32(-4.0)"},
        {     0, 12, 0x00000000, "0.0    = bposit32 ZERO"},
        // Non-power-of-2 cases — exercise fraction-bit packing
        { 6144, 12, 0x42000000, "1.5    = bposit32(1.5)   (1.5 = 1·useed^0·2^0·1.5)"},
        {12288, 12, 0x46000000, "3.0    = bposit32(3.0)"},
        { 5120, 12, 0x41000000, "1.25   = bposit32(1.25)"},
    };
    constexpr int N = sizeof(cases) / sizeof(cases[0]);

    int32_t  h_in[N];
    uint32_t h_out[N];
    int      h_scale[N];  // not used by kernel directly; one global per launch
    for (int i = 0; i < N; ++i) {
        h_in[i] = cases[i].input;
        h_scale[i] = cases[i].scale_shift;
    }

    int32_t* d_in;
    uint32_t* d_out;
    cudaMalloc(&d_in,  N * sizeof(int32_t));
    cudaMalloc(&d_out, N * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, N * sizeof(int32_t), cudaMemcpyHostToDevice);

    // All inputs share the same scale_shift in our test; one launch per group.
    int32_to_bposit32_kernel<<<1, 32>>>(d_in, d_out, N, /*scale_shift*/ 12);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    cudaMemcpy(h_out, d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::printf("=== int32 → bposit32 encoder validation ===\n\n");
    int pass = 0;
    for (int i = 0; i < N; ++i) {
        bool ok = (h_out[i] == cases[i].expected);
        if (ok) ++pass;
        std::printf("  [%s] in=%-10d (shift=%d)  got=0x%08x  expected=0x%08x  %s\n",
                    ok ? "PASS" : "FAIL",
                    cases[i].input, cases[i].scale_shift,
                    h_out[i], cases[i].expected, cases[i].note);
    }
    std::printf("\nSummary: %d/%d cases match Python reference\n", pass, N);

    cudaFree(d_in); cudaFree(d_out);
    return (pass == N) ? 0 : 1;
}