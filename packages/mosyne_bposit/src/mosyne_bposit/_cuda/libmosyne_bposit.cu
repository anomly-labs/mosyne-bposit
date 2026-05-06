// libmosyne_bposit.cu — shared library exposing the bposit-IMMA W8A8 pipeline
// via a clean C ABI. Designed to be loaded from any language via dlopen /
// ctypes / cffi / etc. — no PyTorch dependency, no per-use rebuild.
//
// Public API (all symbols extern "C"):
//
//   int  mosyne_bposit_init();                    // call once; returns 0 on ok
//   void mosyne_bposit_shutdown();
//
//   // Quantize a column-major float32 weight [K, N] to int8 with per-output-
//   // channel scales. w_scale_out is [N] floats.
//   int  mosyne_bposit_quantize_weight_per_channel(
//          const float* w_fp32, int K, int N,
//          int8_t* w_i8_out, float* w_scale_out);
//
//   // W8A8 linear: y[M,N] = x[M,K] @ w_i8[K,N] / (x_scale[M] * w_scale[N]).
//   // Quantizes input per-token internally. Allocates one cuBLASLt workspace
//   // on first call. All buffers must be device pointers.
//   int  mosyne_bposit_linear_w8a8(
//          const float* d_x_fp32, int M, int K,
//          const int8_t* d_w_i8, const float* d_w_scale, int N,
//          float* d_y_fp32);
//
//   // Convenience: full-fp32 round trip for testing without dealing with
//   // cuda allocation. host_x[M,K], host_w[K,N], host_y[M,N], all column-major.
//   int  mosyne_bposit_linear_w8a8_host(
//          const float* host_x, int M, int K,
//          const float* host_w, int N,
//          float* host_y);
//
// Build:
//     nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
//          -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          -lcublasLt -lcudart \
//          libmosyne_bposit.cu -o libmosyne_bposit.so

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- Internal state ---------------------------------------------------------
static cublasLtHandle_t g_lt = nullptr;
static void*  g_ws = nullptr;
static size_t g_ws_bytes = 0;

extern "C" int mosyne_bposit_init() {
    if (g_lt) return 0;
    if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
    g_ws_bytes = (size_t)64 << 20;  // 64 MB default workspace
    if (cudaMalloc(&g_ws, g_ws_bytes) != cudaSuccess) {
        cublasLtDestroy(g_lt); g_lt = nullptr; return -2;
    }
    return 0;
}

extern "C" void mosyne_bposit_shutdown() {
    if (g_ws) { cudaFree(g_ws); g_ws = nullptr; }
    if (g_lt) { cublasLtDestroy(g_lt); g_lt = nullptr; }
}

// ---- Per-channel weight quantization ---------------------------------------
__global__ void k_per_channel_scale(const float* W, int K, int N, float* w_scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float maxabs = 0.f;
    for (int i = 0; i < K; ++i) {
        float v = fabsf(W[j * K + i]);
        if (v > maxabs) maxabs = v;
    }
    w_scale[j] = (maxabs > 1e-8f) ? (127.f / maxabs) : 1.f;
}

__global__ void k_quantize_weight_per_channel(const float* W, const float* w_scale,
                                              int8_t* W_q, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    int j = idx / K;
    float v = W[idx] * w_scale[j];
    v = fminf(127.f, fmaxf(-127.f, v));
    W_q[idx] = (int8_t)__float2int_rn(v);
}

extern "C" int mosyne_bposit_quantize_weight_per_channel(
    const float* w_fp32, int K, int N,
    int8_t* w_i8_out, float* w_scale_out)
{
    int t = 256;
    int blocks_n = (N + t - 1) / t;
    int blocks_kn = (K * N + t - 1) / t;
    k_per_channel_scale<<<blocks_n, t>>>(w_fp32, K, N, w_scale_out);
    k_quantize_weight_per_channel<<<blocks_kn, t>>>(w_fp32, w_scale_out, w_i8_out, K, N);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---- Per-token input quantization ------------------------------------------
__global__ void k_per_token_scale(const float* X, int M, int K, float* x_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float maxabs = 0.f;
    for (int k = 0; k < K; ++k) {
        float v = fabsf(X[i + k * M]);
        if (v > maxabs) maxabs = v;
    }
    x_scale[i] = (maxabs > 1e-8f) ? (127.f / maxabs) : 1.f;
}

__global__ void k_quantize_input_per_token(const float* X, const float* x_scale,
                                           int8_t* X_q, int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    int i = idx % M;
    float v = X[idx] * x_scale[i];
    v = fminf(127.f, fmaxf(-127.f, v));
    X_q[idx] = (int8_t)__float2int_rn(v);
}

// ---- Per-token × per-channel dequant ---------------------------------------
__global__ void k_dequant(const int32_t* y_int, float* y_fp,
                          int M, int N,
                          const float* x_scale, const float* w_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx % M;
    int j = idx / M;
    y_fp[idx] = (float)y_int[idx] / (x_scale[i] * w_scale[j]);
}

// ---- INT8 IMMA matmul ------------------------------------------------------
static int matmul_int8_inplace(int M, int N, int K,
                               const int8_t* dA, const int8_t* dB,
                               int32_t* dC) {
    cublasLtMatmulDesc_t op;
    cublasOperation_t T = CUBLAS_OP_N;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I) != CUBLAS_STATUS_SUCCESS) return -1;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T);
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  M, K, M);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  K, N, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, N, M);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &g_ws_bytes, sizeof g_ws_bytes);
    cublasLtMatmulHeuristicResult_t heur[1]; int got = 0;
    cublasLtMatmulAlgoGetHeuristic(g_lt, op, la, lb, lc, lc, pref, 1, heur, &got);
    int rc = -2;
    if (got > 0) {
        int32_t alpha = 1, beta = 0;
        if (cublasLtMatmul(g_lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc,
                           &heur[0].algo, g_ws, g_ws_bytes, 0) == CUBLAS_STATUS_SUCCESS)
            rc = 0;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    return rc;
}

// ---- Public: device-pointer linear -----------------------------------------
extern "C" int mosyne_bposit_linear_w8a8(
    const float* d_x_fp32, int M, int K,
    const int8_t* d_w_i8, const float* d_w_scale, int N,
    float* d_y_fp32)
{
    if (!g_lt) return -10;
    // Allocate intermediates
    float*  d_x_scale = nullptr;
    int8_t* d_x_i8    = nullptr;
    int32_t* d_y_i32  = nullptr;
    cudaMalloc(&d_x_scale, M * sizeof(float));
    cudaMalloc(&d_x_i8,    M * K * sizeof(int8_t));
    cudaMalloc(&d_y_i32,   M * N * sizeof(int32_t));
    if (!d_x_scale || !d_x_i8 || !d_y_i32) {
        cudaFree(d_x_scale); cudaFree(d_x_i8); cudaFree(d_y_i32);
        return -11;
    }

    int t = 256;
    k_per_token_scale<<<(M + t - 1) / t, t>>>(d_x_fp32, M, K, d_x_scale);
    k_quantize_input_per_token<<<(M*K + t - 1) / t, t>>>(d_x_fp32, d_x_scale, d_x_i8, M, K);
    int rc = matmul_int8_inplace(M, N, K, d_x_i8, d_w_i8, d_y_i32);
    if (rc == 0) {
        k_dequant<<<(M*N + t - 1) / t, t>>>(d_y_i32, d_y_fp32, M, N, d_x_scale, d_w_scale);
    }
    cudaDeviceSynchronize();
    cudaFree(d_x_scale); cudaFree(d_x_i8); cudaFree(d_y_i32);
    return rc;
}

// ---- Convenience: host-side full round trip --------------------------------
extern "C" int mosyne_bposit_linear_w8a8_host(
    const float* host_x, int M, int K,
    const float* host_w, int N,
    float* host_y)
{
    if (!g_lt) {
        int rc = mosyne_bposit_init();
        if (rc != 0) return rc;
    }

    float*  d_x = nullptr; float* d_w = nullptr; float* d_y = nullptr;
    int8_t* d_wi8 = nullptr; float* d_wscale = nullptr;
    cudaMalloc(&d_x, M*K*sizeof(float));
    cudaMalloc(&d_w, K*N*sizeof(float));
    cudaMalloc(&d_y, M*N*sizeof(float));
    cudaMalloc(&d_wi8, K*N*sizeof(int8_t));
    cudaMalloc(&d_wscale, N*sizeof(float));
    cudaMemcpy(d_x, host_x, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, host_w, K*N*sizeof(float), cudaMemcpyHostToDevice);

    int rc = mosyne_bposit_quantize_weight_per_channel(d_w, K, N, d_wi8, d_wscale);
    if (rc == 0) {
        rc = mosyne_bposit_linear_w8a8(d_x, M, K, d_wi8, d_wscale, N, d_y);
    }
    if (rc == 0) {
        cudaMemcpy(host_y, d_y, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y); cudaFree(d_wi8); cudaFree(d_wscale);
    return rc;
}
