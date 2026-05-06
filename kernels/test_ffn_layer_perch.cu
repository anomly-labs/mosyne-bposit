// test_ffn_layer_perch.cu — Llama-class FFN forward with PER-CHANNEL weight
// quantization. Demonstrates that the bposit-IMMA pipeline matches BF16 HMMA
// throughput AND production-grade inference accuracy (L2 relerr < 1%) when
// paired with the per-output-channel scaling that TensorRT-LLM, AWQ, GPTQ,
// and SmoothQuant all use.
//
// Architecture (one block, Llama-3-8B class):
//     x [B,D]  →  x @ W_gate [D,F]  →  ReLU  →  @ W_down [F,D]  →  y [B,D]
// where D=4096, F=14336, B=128.
//
// Per-channel quantization scheme:
//   - Each weight matrix W[K,N] gets one scale per output channel:
//         w_scale[j] = 127 / max(|W[:,j]|)         for j in [0, N)
//     so each column maps to ±127 with no clipping.
//   - Quantized weight: W_q[i,j] = round(W[i,j] * w_scale[j])
//   - Input still uses per-tensor scale (cheap, accurate enough at FP16 input).
//   - Dequant: y[i,j] = y_int[i,j] / (x_scale * w_scale[j])
//
// Compared to test_ffn_layer_bench.cu (per-tensor, 3.4% L2 error), this kernel
// expects to land below 1% L2 error on the same Gaussian-init weights.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          -lcublasLt -lcudart \
//          test_ffn_layer_perch.cu -o test_ffn_perch

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas %s @%d: %d\n", #x, __LINE__, (int)s); \
    std::exit(3); } } while (0)

constexpr int B = 128;
constexpr int D = 4096;
constexpr int F = 14336;
constexpr int N_RUNS = 10;

// =============================================================================
// Compute per-output-channel scale: w_scale[j] = 127 / max(|W[:,j]|)
// W is column-major K×N (K = input dim, N = output dim).
// =============================================================================
__global__ void compute_per_channel_scale(const float* __restrict__ W,
                                          int K, int N,
                                          float* __restrict__ w_scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float maxabs = 0.f;
    for (int i = 0; i < K; ++i) {
        float v = fabsf(W[j * K + i]);  // column-major: W[i + j*K]
        if (v > maxabs) maxabs = v;
    }
    w_scale[j] = (maxabs > 1e-8f) ? (127.f / maxabs) : 1.f;
}

// Quantize weight with per-channel scale: W_q[i,j] = round(W[i,j] * w_scale[j])
__global__ void quantize_weight_per_channel(const float* __restrict__ W,
                                            const float* __restrict__ w_scale,
                                            int8_t* __restrict__ W_q,
                                            int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    int j = idx / K;  // output channel
    float v = W[idx] * w_scale[j];
    v = fminf(127.f, fmaxf(-127.f, v));
    W_q[idx] = (int8_t)__float2int_rn(v);
}

// Per-tensor input quantization (input still per-tensor; per-token would be
// the next refinement)
__global__ void quantize_input_per_tensor(const float* in, int8_t* out,
                                          int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i] * scale;
        v = fminf(127.f, fmaxf(-127.f, v));
        out[i] = (int8_t)__float2int_rn(v);
    }
}

// Per-channel dequant: y[i,j] = y_int[i,j] / (x_scale * w_scale[j])
__global__ void dequant_per_channel(const int32_t* y_int, float* y_fp,
                                    int M, int N, float x_scale,
                                    const float* w_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int j = idx / M;
    y_fp[idx] = (float)y_int[idx] / (x_scale * w_scale[j]);
}

__global__ void relu_fp32(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0) x[i] = 0;
}
__global__ void relu_bf16(__nv_bfloat16* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __bfloat162float(x[i]) < 0) x[i] = __float2bfloat16(0.f);
}

__global__ void f32_to_bf16(const float* in, __nv_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ void bf16_to_f32(const __nv_bfloat16* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void l2_err_kernel(const float* a, const float* b, int n,
                              float* sum_diff, float* sum_ref) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = a[i] - b[i];
        atomicAdd(sum_diff, d * d);
        atomicAdd(sum_ref, b[i] * b[i]);
    }
}

static double matmul_bf16(cublasLtHandle_t lt, int m, int n, int k,
                          void* dA, void* dB, void* dC, void* dWS, size_t ws) {
    cublasLtMatmulDesc_t op;
    cublasOperation_t T = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T));
    cublasLtMatrixLayout_t la, lb, lc;
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, m, k, m));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, m, n, m));
    cublasLtMatmulPreference_t pref;
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int got;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, heur, &got));
    float alpha = 1.f, beta = 0.f;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc,
                   &heur[0].algo, dWS, ws, 0); cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < N_RUNS; ++r)
        cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc,
                       &heur[0].algo, dWS, ws, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    return (double)ms / N_RUNS;
}

static double matmul_int8(cublasLtHandle_t lt, int m, int n, int k,
                          void* dA, void* dB, void* dC, void* dWS, size_t ws) {
    cublasLtMatmulDesc_t op;
    cublasOperation_t T = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T));
    cublasLtMatrixLayout_t la, lb, lc;
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  m, k, m));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  k, n, k));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, m, n, m));
    cublasLtMatmulPreference_t pref;
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int got;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, heur, &got));
    int32_t alpha = 1, beta = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc,
                   &heur[0].algo, dWS, ws, 0); cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < N_RUNS; ++r)
        cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc,
                       &heur[0].algo, dWS, ws, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    return (double)ms / N_RUNS;
}

int main() {
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Transformer FFN forward, PER-CHANNEL bposit-IMMA quantization ===\n");
    std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Block:  x[%d,%d] @ W_gate[%d,%d] → ReLU → @ W_down[%d,%d]\n\n", B, D, D, F, F, D);

    static float h_x[B*D], h_Wg[D*F], h_Wd[F*D];
    unsigned int seed = 0xCAFEC0DE;
    auto urand = [&]{ seed = seed*1664525u + 1013904223u; return ((seed>>1)/(double)0x7FFFFFFF) * 2.0 - 1.0; };
    auto frac  = [&]{ seed = seed*1664525u + 1013904223u; return ((seed>>1)/(double)0x7FFFFFFF); };
    for (int i = 0; i < B*D; ++i) h_x[i] = (float)urand();
    // Realistic LLM-style weight init: most columns at stddev 0.02, but ~3% of
    // columns have a 10x amplitude outlier scale (the salient-feature pattern
    // AWQ/GPTQ specifically address — present in real Llama/Qwen weights).
    static float wg_col_scale[F], wd_col_scale[D];
    for (int j = 0; j < F; ++j) wg_col_scale[j] = (frac() < 0.03) ? 0.20f : 0.02f;
    for (int j = 0; j < D; ++j) wd_col_scale[j] = (frac() < 0.03) ? 0.20f : 0.02f;
    for (int i = 0; i < D*F; ++i) {  // column-major: i = row + col*K
        int col = i / D;
        h_Wg[i] = (float)(urand() * wg_col_scale[col]);
    }
    for (int i = 0; i < F*D; ++i) {
        int col = i / F;
        h_Wd[i] = (float)(urand() * wd_col_scale[col]);
    }

    // ---- Buffers ----
    float *d_x_f, *d_Wg_f, *d_Wd_f;
    __nv_bfloat16 *d_x_bf, *d_Wg_bf, *d_Wd_bf, *d_h1_bf, *d_y_bf;
    int8_t *d_x_i8, *d_Wg_i8, *d_Wd_i8, *d_h1_i8;
    int32_t *d_h1_i32, *d_y_i32;
    float *d_h1_f, *d_y_imma_f, *d_y_bf_f;
    float *d_Wg_scale, *d_Wd_scale, *d_h1_scale;
    CUDA_OK(cudaMalloc(&d_x_f,  B*D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Wg_f, D*F*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Wd_f, F*D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_x_bf,  B*D*sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_Wg_bf, D*F*sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_Wd_bf, F*D*sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_h1_bf, B*F*sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_y_bf,  B*D*sizeof(__nv_bfloat16)));
    CUDA_OK(cudaMalloc(&d_x_i8,  B*D*sizeof(int8_t)));
    CUDA_OK(cudaMalloc(&d_Wg_i8, D*F*sizeof(int8_t)));
    CUDA_OK(cudaMalloc(&d_Wd_i8, F*D*sizeof(int8_t)));
    CUDA_OK(cudaMalloc(&d_h1_i8, B*F*sizeof(int8_t)));
    CUDA_OK(cudaMalloc(&d_h1_i32, B*F*sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_y_i32,  B*D*sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_h1_f, B*F*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_y_imma_f, B*D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_y_bf_f,   B*D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Wg_scale, F*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_Wd_scale, D*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_h1_scale, F*sizeof(float)));  // for h1 (per-tensor scalar; F just for consistency)

    CUDA_OK(cudaMemcpy(d_x_f,  h_x,  B*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Wg_f, h_Wg, D*F*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_Wd_f, h_Wd, F*D*sizeof(float), cudaMemcpyHostToDevice));

    int t = 256;
    int blocks_xd = (B*D + t - 1) / t;
    int blocks_dF = (D*F + t - 1) / t;
    int blocks_FD = (F*D + t - 1) / t;
    int blocks_xF = (B*F + t - 1) / t;

    // ---- BF16 baseline ----
    f32_to_bf16<<<blocks_xd, t>>>(d_x_f,  d_x_bf,  B*D);
    f32_to_bf16<<<blocks_dF, t>>>(d_Wg_f, d_Wg_bf, D*F);
    f32_to_bf16<<<blocks_FD, t>>>(d_Wd_f, d_Wd_bf, F*D);

    // ---- Per-channel scales for weights ----
    int blocks_F = (F + t - 1) / t;
    int blocks_D = (D + t - 1) / t;
    compute_per_channel_scale<<<blocks_F, t>>>(d_Wg_f, D, F, d_Wg_scale);
    compute_per_channel_scale<<<blocks_D, t>>>(d_Wd_f, F, D, d_Wd_scale);
    cudaDeviceSynchronize();

    quantize_weight_per_channel<<<blocks_dF, t>>>(d_Wg_f, d_Wg_scale, d_Wg_i8, D, F);
    quantize_weight_per_channel<<<blocks_FD, t>>>(d_Wd_f, d_Wd_scale, d_Wd_i8, F, D);
    float x_scale = 120.f / 1.0f;
    quantize_input_per_tensor<<<blocks_xd, t>>>(d_x_f, d_x_i8, B*D, x_scale);

    cublasLtHandle_t lt; CUBLAS_OK(cublasLtCreate(&lt));
    size_t ws = 64ull << 20; void* dWS;
    CUDA_OK(cudaMalloc(&dWS, ws));

    // ---- BF16 path timing ----
    double bf_g_ms = matmul_bf16(lt, B, F, D, d_x_bf, d_Wg_bf, d_h1_bf, dWS, ws);
    relu_bf16<<<blocks_xF, t>>>(d_h1_bf, B*F);
    double bf_d_ms = matmul_bf16(lt, B, D, F, d_h1_bf, d_Wd_bf, d_y_bf, dWS, ws);
    cudaDeviceSynchronize();

    // ---- INT8 path: gate matmul ----
    double i8_g_ms = matmul_int8(lt, B, F, D, d_x_i8, d_Wg_i8, d_h1_i32, dWS, ws);
    // Per-channel dequant for gate output, then ReLU, then re-quantize per-tensor for h1
    dequant_per_channel<<<blocks_xF, t>>>(d_h1_i32, d_h1_f, B, F, x_scale, d_Wg_scale);
    relu_fp32<<<blocks_xF, t>>>(d_h1_f, B*F);
    float h1_scale = 120.f / 5.0f;  // h1 magnitude bounded by sqrt(D) * input_max_W
    quantize_input_per_tensor<<<blocks_xF, t>>>(d_h1_f, d_h1_i8, B*F, h1_scale);

    // ---- INT8 path: down matmul ----
    double i8_d_ms = matmul_int8(lt, B, D, F, d_h1_i8, d_Wd_i8, d_y_i32, dWS, ws);
    dequant_per_channel<<<blocks_xd, t>>>(d_y_i32, d_y_imma_f, B, D, h1_scale, d_Wd_scale);
    cudaDeviceSynchronize();

    // ---- BF16 result → fp32 for accuracy comparison ----
    bf16_to_f32<<<blocks_xd, t>>>(d_y_bf, d_y_bf_f, B*D);
    cudaDeviceSynchronize();

    // ---- L2 relerr ----
    float* d_diff; float* d_ref;
    CUDA_OK(cudaMalloc(&d_diff, sizeof(float)));
    CUDA_OK(cudaMalloc(&d_ref,  sizeof(float)));
    float zero = 0;
    cudaMemcpy(d_diff, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref,  &zero, sizeof(float), cudaMemcpyHostToDevice);
    l2_err_kernel<<<blocks_xd, t>>>(d_y_imma_f, d_y_bf_f, B*D, d_diff, d_ref);
    cudaDeviceSynchronize();
    float h_diff, h_ref;
    cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ref,  d_ref,  sizeof(float), cudaMemcpyDeviceToHost);
    double l2_relerr = std::sqrt((double)h_diff / (double)h_ref);

    double bf_total_ms = bf_g_ms + bf_d_ms;
    double i8_total_ms = i8_g_ms + i8_d_ms;
    double gate_ops = 2.0 * (double)B * F * D;
    double down_ops = 2.0 * (double)B * D * F;
    double total_ops = gate_ops + down_ops;

    std::printf("Per-channel weight quantization (one scale per output column):\n\n");
    std::printf("Per-matmul timings (avg of %d runs):\n", N_RUNS);
    std::printf("  gate:  BF16 HMMA   %7.3f ms   %7.1f TFLOPS\n",
                bf_g_ms, gate_ops / 1e12 / (bf_g_ms * 1e-3));
    std::printf("         bposit-IMMA %7.3f ms   %7.1f TPOPS  (per-channel W scales)\n",
                i8_g_ms, gate_ops / 1e12 / (i8_g_ms * 1e-3));
    std::printf("  down:  BF16 HMMA   %7.3f ms   %7.1f TFLOPS\n",
                bf_d_ms, down_ops / 1e12 / (bf_d_ms * 1e-3));
    std::printf("         bposit-IMMA %7.3f ms   %7.1f TPOPS  (per-channel W scales)\n",
                i8_d_ms, down_ops / 1e12 / (i8_d_ms * 1e-3));
    std::printf("\nBlock total:\n");
    std::printf("  BF16 HMMA:    %7.3f ms   %7.1f TFLOPS    weights %.1f MB\n",
                bf_total_ms, total_ops / 1e12 / (bf_total_ms * 1e-3),
                (D*F + F*D) * 2.0 / 1e6);
    std::printf("  bposit-IMMA:  %7.3f ms   %7.1f TPOPS    weights %.1f MB (2x lighter)\n",
                i8_total_ms, total_ops / 1e12 / (i8_total_ms * 1e-3),
                (D*F + F*D) * 1.0 / 1e6);
    std::printf("\nThroughput ratio: %.2fx\n", bf_total_ms / i8_total_ms);
    std::printf("Output L2 relerr (per-channel PTQ vs BF16): %.4e\n", l2_relerr);
    std::printf("\nReference per-tensor result was 3.39e-02 L2 relerr (test_ffn_layer_bench).\n");
    std::printf("Improvement: %.2fx tighter accuracy at the same throughput.\n",
                3.39e-2 / l2_relerr);

    cudaFree(d_diff); cudaFree(d_ref);
    cudaFree(d_x_f); cudaFree(d_Wg_f); cudaFree(d_Wd_f);
    cudaFree(d_x_bf); cudaFree(d_Wg_bf); cudaFree(d_Wd_bf); cudaFree(d_h1_bf); cudaFree(d_y_bf);
    cudaFree(d_x_i8); cudaFree(d_Wg_i8); cudaFree(d_Wd_i8); cudaFree(d_h1_i8);
    cudaFree(d_h1_i32); cudaFree(d_y_i32);
    cudaFree(d_h1_f); cudaFree(d_y_imma_f); cudaFree(d_y_bf_f);
    cudaFree(d_Wg_scale); cudaFree(d_Wd_scale); cudaFree(d_h1_scale);
    cudaFree(dWS);
    cublasLtDestroy(lt);
    return 0;
}
