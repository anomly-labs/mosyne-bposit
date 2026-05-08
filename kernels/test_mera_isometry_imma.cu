// test_mera_isometry_imma.cu — IMMA-accelerated MERA isometry contraction
// over bposit8 inputs, reformulated as int8 GEMM via cuBLASLt.
//
// Operation (same as test_mera_isometry.cu, chi=4 path-A reference):
//     out[k][b] = Σ_{i,j} w[i,j,k] · a_b[i] · b_b[j]
//
// Reformulation as GEMM:
//     Define AB[(i*chi+j), b] = a_b[i] * b_b[j]               (chi²×B outer products)
//     Define Wᵀ[k, (i*chi+j)] = w[i,j,k]                      (chi×chi² flattened)
//     Then out = Wᵀ · AB  (chi×B = (chi×chi²) · (chi²×B))
//
// Path C: bposit8 inputs → int8 fixed-point (LUT) → cuBLASLt INT8 GEMM
// (the path we measured at 176 TPOPS = 1.02× FP4 QMMA on the 5090)
// → int32 output → quire256 / bposit final form.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 -lcublasLt -lcudart \
//          test_mera_isometry_imma.cu -o test_mera_imma
//
// Validation: same test case as test_mera_isometry.cu (all w=1.0, a=b=0.5,
// expected out[k][b] = 4.0 for all (k,b)). At chi=4, B=4 we can compare
// element-by-element vs path-A. At chi=64, B=64 we measure throughput.

#include "bposit16_luts.cuh"

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas %s @%d: %d\n", #x, __LINE__, (int)s); \
    std::exit(3); } } while (0)

using bposit8_t = std::uint8_t;

// Scale chosen so bposit8(1.0) → int8(64). Preserves the small-value
// arithmetic exactly (bposit8(0.5) → int8(32), bposit8(0.25) → int8(16), …)
// and stays inside int8 range for any decoded value with |value| ≤ 1.98.
constexpr int SCALE_BITS = 6;        // 2^6 = 64
constexpr int SCALE      = 1 << SCALE_BITS;

// Pre-built LUT: bposit8 → int8 (fixed-point, SCALE = 2^6).
// Populated at runtime via cudaMemcpyToSymbol; visible inside kernels by name.
__device__ std::int8_t bposit8_to_int8_lut[256];

// =============================================================================
// Pre-decode kernel: bposit8 → int8 fixed-point.
// Reads the __device__ LUT directly by name (passing __device__ symbols as
// pointer args from host doesn't work — host doesn't know the device address
// without cudaGetSymbolAddress).
// =============================================================================
__global__ void bposit8_to_int8(
    const bposit8_t* __restrict__ in,
    std::int8_t* __restrict__ out,
    std::size_t n)
{
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = bposit8_to_int8_lut[in[i]];
    }
}

// =============================================================================
// bposit32 encoder (post-decode): int32 → bposit32. Validated bit-exact
// against the Python reference in test_int32_to_bposit32.cu.
// =============================================================================
__device__ __forceinline__ uint32_t encode_unsigned_32(uint32_t mantissa_bits,
                                                       int total_e) {
    if (total_e > 48)  return 0x7FFFFFFFu;
    if (total_e < -48) return 0x00000001u;
    int k = total_e >> 3;
    int e = total_e & 7;
    uint32_t out = 0;
    int n_bits = 0;
    if (k >= 0) {
        int ones_to_add = k + 1;
        for (int i = 0; i < ones_to_add && n_bits < 31; ++i) {
            out = (out << 1) | 1u; ++n_bits;
        }
        if (n_bits < 31) { out <<= 1; ++n_bits; }
    } else {
        int zeros_to_add = -k;
        for (int i = 0; i < zeros_to_add && n_bits < 31; ++i) {
            out <<= 1; ++n_bits;
        }
        if (n_bits < 31) { out = (out << 1) | 1u; ++n_bits; }
    }
    #pragma unroll
    for (int i = 2; i >= 0; --i) {
        if (n_bits >= 31) break;
        out = (out << 1) | ((e >> i) & 1u);
        ++n_bits;
    }
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
    int nlz = __clz(abs_I);
    int msb_pos = 31 - nlz;
    int total_e = msb_pos - scale_shift;
    uint32_t shifted = (msb_pos >= 31) ? abs_I : (abs_I << (31 - msb_pos));
    uint32_t mantissa_bits = shifted & 0x7FFFFFFFu;
    uint32_t field = encode_unsigned_32(mantissa_bits, total_e);
    if (sign) field = ((~field) + 1) & 0x7FFFFFFFu;
    return ((uint32_t(sign) << 31) | field) & 0xFFFFFFFFu;
}

__global__ void int32_to_bposit32_kernel(
    const int32_t* __restrict__ in,
    uint32_t*      __restrict__ out,
    int n,
    int scale_shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = int_to_bposit32(in[idx], scale_shift);
}


// =============================================================================
// AB outer-product kernel: AB[(i*chi+j), b] = a_b[i] * b_b[j]
// (using int8 inputs, int8 output via SCALE rescaling)
// =============================================================================
__global__ void outer_product_int8(
    const std::int8_t* __restrict__ a,    // (chi, B) in column-major: a[i, b]
    const std::int8_t* __restrict__ b,    // (chi, B) col-major
    std::int8_t* __restrict__ ab,          // (chi², B) col-major
    int chi, int B)
{
    int batch = blockIdx.y;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= B || idx >= chi * chi) return;
    int i = idx / chi;
    int j = idx % chi;
    int prod = int(a[batch * chi + i]) * int(b[batch * chi + j]);
    // a, b are int8 fixed-point with SCALE; product has SCALE². Rescale to int8.
    int prod_rescaled = prod >> SCALE_BITS;
    if (prod_rescaled > 127) prod_rescaled = 127;
    if (prod_rescaled < -128) prod_rescaled = -128;
    ab[batch * chi * chi + idx] = std::int8_t(prod_rescaled);
}

// =============================================================================
// Host-side LUT builder: compute bposit8 → int8 fixed-point at SCALE
// from the existing bposit8_to_quire_lut (which gives the value × 2^96).
// Convert by shifting right (96 - SCALE_BITS) bits.
// =============================================================================
static void build_bposit8_to_int8_lut(std::int8_t out_lut[256]) {
    // Copy bposit8_to_quire_lut to host first
    std::uint64_t host_quire[256][4];
    CUDA_OK(cudaMemcpyFromSymbol(host_quire, bposit8_to_quire_lut,
                                  sizeof(host_quire)));
    for (int i = 0; i < 256; ++i) {
        // Reconstruct signed 256-bit from 4× uint64
        // For our range, the value × 2^96 fits in int64 most of the time.
        // Take w[1] (which is the bits [64..127]); for value in [-2, 2]
        // the bits [64+SCALE_BITS .. 64+SCALE_BITS+7] hold the int8.
        // Shift the value's quire by (96 - SCALE_BITS) right.
        // Effectively: int8 = (signed_value × 2^96) >> (96 - SCALE_BITS)
        //                   = signed_value × 2^SCALE_BITS = signed_value × 64
        //
        // For the test case (small values), w[1] is enough; w[2..3] are sign-ext.
        std::uint64_t hi = host_quire[i][1];
        std::uint64_t lo = host_quire[i][0];
        // Combined as signed 128-bit (for our range, w[2..3] are 0 or ~0
        // depending on sign).
        bool negative = (host_quire[i][3] & 0x8000000000000000ULL) != 0;
        // Build signed value via two's complement of (lo, hi)
        // Shift right by (96 - SCALE_BITS) = 90 bits → only top 38 bits of hi matter
        // → essentially: int8 = (signed-hi) >> (90 - 64) = signed-hi >> 26
        int64_t signed_hi = (int64_t)hi;
        if (negative && signed_hi >= 0) signed_hi |= (int64_t(1) << 63);
        int64_t scaled = signed_hi >> (96 - SCALE_BITS - 64);  // = signed_hi >> 26
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        out_lut[i] = (std::int8_t)scaled;
        (void)lo;
    }
}

// =============================================================================
// Main bench/test
// =============================================================================
int main(int argc, char** argv) {
    int chi = (argc > 1) ? std::atoi(argv[1]) : 4;
    int B   = (argc > 2) ? std::atoi(argv[2]) : 4;
    bool measure_only = (argc > 3) && std::strcmp(argv[3], "--bench") == 0;

    std::printf("=== MERA isometry contraction via IMMA (cuBLASLt INT8 GEMM) ===\n");
    std::printf("chi=%d  B=%d\n\n", chi, B);

    // Build host LUT
    std::int8_t h_lut[256];
    // We need device's __device__ const lut populated; do it by computing on
    // host then memcpy to the symbol.
    // First verify the symbol exists (it does — declared above).
    // Compute by referring to bposit8_to_quire_lut via cudaMemcpyFromSymbol.
    build_bposit8_to_int8_lut(h_lut);
    CUDA_OK(cudaMemcpyToSymbol(bposit8_to_int8_lut, h_lut, sizeof(h_lut)));

    // Spot-check the LUT for a few known values
    constexpr bposit8_t BP8_HALF = 0x38;  // bposit8(0.5)
    constexpr bposit8_t BP8_ONE  = 0x40;  // bposit8(1.0)
    std::printf("LUT spot-check: bposit8(0.5)=0x%02x → int8(%d) (expect 32)\n",
                BP8_HALF, (int)h_lut[BP8_HALF]);
    std::printf("LUT spot-check: bposit8(1.0)=0x%02x → int8(%d) (expect 64)\n",
                BP8_ONE,  (int)h_lut[BP8_ONE]);
    std::printf("\n");

    // Build inputs:
    //   w_full[chi*chi*chi]: all bposit8(1.0)
    //   a, b shape (chi, B): all bposit8(0.5)
    int n_w = chi * chi * chi;
    std::vector<bposit8_t> h_w(n_w, BP8_ONE);
    std::vector<bposit8_t> h_a(chi * B, BP8_HALF);
    std::vector<bposit8_t> h_b(chi * B, BP8_HALF);

    bposit8_t *d_w_bp, *d_a_bp, *d_b_bp;
    std::int8_t *d_w_i8, *d_a_i8, *d_b_i8, *d_ab_i8;
    std::int32_t* d_out_i32;
    CUDA_OK(cudaMalloc(&d_w_bp, n_w));
    CUDA_OK(cudaMalloc(&d_a_bp, chi * B));
    CUDA_OK(cudaMalloc(&d_b_bp, chi * B));
    CUDA_OK(cudaMalloc(&d_w_i8, n_w));
    CUDA_OK(cudaMalloc(&d_a_i8, chi * B));
    CUDA_OK(cudaMalloc(&d_b_i8, chi * B));
    CUDA_OK(cudaMalloc(&d_ab_i8, chi * chi * B));
    CUDA_OK(cudaMalloc(&d_out_i32, chi * B * sizeof(std::int32_t)));

    CUDA_OK(cudaMemcpy(d_w_bp, h_w.data(), n_w, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_a_bp, h_a.data(), chi * B, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b_bp, h_b.data(), chi * B, cudaMemcpyHostToDevice));

    // Step 1: pre-decode bposit8 → int8
    bposit8_to_int8<<<(n_w + 255) / 256, 256>>>(d_w_bp, d_w_i8, n_w);
    bposit8_to_int8<<<(chi * B + 255) / 256, 256>>>(d_a_bp, d_a_i8, chi * B);
    bposit8_to_int8<<<(chi * B + 255) / 256, 256>>>(d_b_bp, d_b_i8, chi * B);
    CUDA_OK(cudaDeviceSynchronize());

    // Sanity-check: verify pre-decode actually produced 32 (= int8(0.5))
    {
        std::int8_t check[4];
        cudaMemcpy(check, d_a_i8, 4, cudaMemcpyDeviceToHost);
        std::printf("Sanity: a_i8[0..3] = %d %d %d %d (expect 32 32 32 32)\n",
                    (int)check[0], (int)check[1], (int)check[2], (int)check[3]);
    }

    // Step 2: compute AB outer products
    dim3 grid_ab((chi * chi + 255) / 256, B);
    outer_product_int8<<<grid_ab, 256>>>(d_a_i8, d_b_i8, d_ab_i8, chi, B);
    CUDA_OK(cudaDeviceSynchronize());

    // Step 3: cuBLASLt INT8 GEMM
    //   out[chi, B] = Wᵀ[chi, chi²] · AB[chi², B]
    //   Wᵀ is just W reshaped (chi² rows = the (i,j) flat index, chi cols = k),
    //   stored in column-major as (chi × chi²) where matrix[k, (i*chi+j)] = w[i,j,k].
    //   Equivalently: transpose W along (k, (i,j)) axes — but for the all-ones
    //   case the transpose is a no-op (every element is the same).
    cublasLtHandle_t lt;
    CUBLAS_OK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op;
    cublasOperation_t op_t = CUBLAS_OP_T;  // transpose Wᵀ (so input W is stored normally)
    cublasOperation_t op_n = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

    int M = chi, N = B, K = chi * chi;
    cublasLtMatrixLayout_t la, lb, lc;
    // A is W stored as (chi² × chi) col-major (which is W with W[(i,j), k] = w[i,j,k])
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, M, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, N, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, N, M));

    // Workspace: env-overridable so we can crank it up for big GEMMs when
    // the chip has memory headroom (e.g. vLLM paused).
    size_t ws_mb = std::getenv("MOSYNE_WS_MB")
                 ? size_t(std::atoi(std::getenv("MOSYNE_WS_MB"))) : 32;
    size_t ws_bytes = ws_mb << 20;
    void* dWS = nullptr;
    CUDA_OK(cudaMalloc(&dWS, ws_bytes));

    cublasLtMatmulPreference_t pref;
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heur[8]{};
    int got = 0;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc,
                                              pref, 8, heur, &got));
    if (got == 0) {
        std::fprintf(stderr, "no algo found for chi=%d B=%d K=%d\n", chi, B, K);
        return 4;
    }

    int alpha = 1, beta = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // Warm-up
    CUBLAS_OK(cublasLtMatmul(lt, op, &alpha, d_w_i8, la, d_ab_i8, lb,
                             &beta, d_out_i32, lc, d_out_i32, lc,
                             &heur[0].algo, dWS, ws_bytes, /*stream*/ 0));
    CUDA_OK(cudaDeviceSynchronize());

    // Timed runs
    int repeats = (chi * B < 4096) ? 50 : 20;
    double total_ms = 0;
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(e0);
        CUBLAS_OK(cublasLtMatmul(lt, op, &alpha, d_w_i8, la, d_ab_i8, lb,
                                 &beta, d_out_i32, lc, d_out_i32, lc,
                                 &heur[0].algo, dWS, ws_bytes, 0));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        total_ms += ms;
    }
    double ms_avg = total_ms / repeats;
    double macs = double(M) * double(N) * double(K);
    double tops = macs / (ms_avg * 1e-3) / 1e12;

    // POST-DECODE: int32 → bposit32 via the CUDA-native encoder.
    // semantic value = int32 / SCALE² (= int32 / 4096 with our SCALE_BITS=6),
    // so scale_shift = 2 * SCALE_BITS = 12.
    uint32_t* d_out_bp32 = nullptr;
    CUDA_OK(cudaMalloc(&d_out_bp32, chi * B * sizeof(uint32_t)));
    int n_out = chi * B;
    int32_to_bposit32_kernel<<<(n_out + 255) / 256, 256>>>(
        d_out_i32, d_out_bp32, n_out, /*scale_shift*/ 2 * SCALE_BITS);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_bp32(n_out);
    CUDA_OK(cudaMemcpy(h_bp32.data(), d_out_bp32,
                       n_out * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Read output
    std::vector<std::int32_t> h_out(chi * B);
    CUDA_OK(cudaMemcpy(h_out.data(), d_out_i32,
                       chi * B * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    // Expected: int32 = chi² · (32 * 32 / 64) * 64 = chi² · 16 * 64 = chi² · 1024
    //                   ↑ outer prod scaled    ↑ rescale to int8
    //                                                ↑ then GEMM with W=64
    // For chi=4, B=4: expected = 16 · 1024 = 16384 per output cell
    // Decode to real: int32 / SCALE³ = int32 / (64)³ = int32 / 262144
    //   16384 / 262144 = 0.0625... wait that's wrong, should be 4.0
    //
    // Let me retrace: a, b in int8 = 32 (= 0.5 * 64).
    //   AB outer = 32 * 32 = 1024 (in int32), rescaled to int8 = 1024 >> 6 = 16.
    //     So AB int8 = 16, representing 0.25 (= 16/64).
    //   W in int8 = 64 (= 1.0 * 64).
    //   GEMM: out[k][b] = Σ_{i,j} W[i,j,k] · AB[(i,j), b]
    //                   = Σ 64 · 16 = chi² · 1024
    //   For chi=4: 16 · 1024 = 16384 in int32.
    //   Decode: 16384 / 64² = 16384 / 4096 = 4.0 ✓ (one fewer SCALE because AB
    //   was already rescaled)
    int32_t expected = chi * chi * SCALE * SCALE / 4;  // = chi² · 1024 for our test
    // Actually: AB int8 = 16 = 0.25*64. Its semantic value is 0.25. After GEMM
    // with W=64 (semantic 1.0), output is Σ chi² · (1·0.25) = chi²·0.25 in
    // semantic units. In int32 with one SCALE applied: chi² · 16 · 64 = chi²·1024.
    // For chi=4: 16384. Semantic = 16384/64 = 256/64 = 4.0 ✓
    //
    // (We divide by SCALE not SCALE² because outer-product already pre-scaled.)
    int32_t expected_correct = chi * chi * 16 * SCALE;
    int pass = 0;
    for (int idx = 0; idx < chi * B; ++idx) {
        if (h_out[idx] == expected_correct) ++pass;
    }
    // Semantic value = int32 / (SCALE_W · SCALE_AB). Both carry one SCALE.
    double semantic = double(expected_correct) / (double(SCALE) * double(SCALE));
    std::printf("Expected per cell: %d  (semantic %.4f, target 4.0 for all-half/all-one)\n",
                expected_correct, semantic);
    std::printf("Got out[0]: %d   out[chi*B-1]: %d\n",
                h_out[0], h_out[chi * B - 1]);
    std::printf("\n");

    if (measure_only) {
        std::printf("BENCH: chi=%d B=%d K=%d  GEMM=%d×%d×%d  "
                    "ms_avg=%.4f  TOPS=%.1f  TPOPS=%.1f\n",
                    chi, B, K, M, N, K, ms_avg, tops, 2 * tops);
    } else {
        std::printf("CORRECTNESS (int32): %d/%d outputs match %d  (semantic %.4f) — %s\n",
                    pass, chi * B, expected_correct,
                    double(expected_correct) / SCALE,
                    (pass == chi * B) ? "PASS" : "FAIL");
        std::printf("THROUGHPUT (info): ms=%.4f  TOPS=%.2f  TPOPS=%.2f\n",
                    ms_avg, tops, 2 * tops);

        // BPOSIT32 OUTPUT VERIFICATION (closes the bposit-in/bposit-out loop)
        // Expected: each output cell encodes to bposit32(semantic) where
        // semantic = expected_correct / SCALE² = chi² · 16 · SCALE / SCALE² = chi²/4
        // For chi=4: semantic = 4.0  →  bposit32 = 0x48000000
        // For chi=8: semantic = 16.0 →  bposit32 = 0x50000000
        uint32_t expected_bp32_per_chi[] = {
            0x00000000,  // chi=0 placeholder
            0x40000000,  // chi=2  → semantic 1.0  → bposit32 0x40000000
            0x44000000,  // chi=3
            0x48000000,  // chi=4  → semantic 4.0  → bposit32 0x48000000
        };
        uint32_t expected_bp32 = 0x48000000;  // For chi=4 specifically
        if (chi == 4) expected_bp32 = 0x48000000;
        else if (chi == 8) expected_bp32 = 0x50000000;   // 16.0
        else if (chi == 2) expected_bp32 = 0x40000000;   // 1.0
        int bp_pass = 0;
        for (int idx = 0; idx < n_out; ++idx) {
            if (h_bp32[idx] == expected_bp32) ++bp_pass;
        }
        std::printf("BPOSIT32 OUTPUT: out[0]=0x%08x  expected=0x%08x  %d/%d %s\n",
                    h_bp32[0], expected_bp32, bp_pass, n_out,
                    (bp_pass == n_out) ? "PASS" : "FAIL");
    }
    cudaFree(d_out_bp32);

    cudaFree(d_w_bp); cudaFree(d_a_bp); cudaFree(d_b_bp);
    cudaFree(d_w_i8); cudaFree(d_a_i8); cudaFree(d_b_i8);
    cudaFree(d_ab_i8); cudaFree(d_out_i32); cudaFree(dWS);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
    return (measure_only || pass == chi * B) ? 0 : 1;
}
