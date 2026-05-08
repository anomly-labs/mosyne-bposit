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
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

// ---- Internal state ---------------------------------------------------------
static cublasLtHandle_t g_lt = nullptr;
static void*  g_ws = nullptr;        // cuBLASLt algorithm workspace
static size_t g_ws_bytes = 0;

// Per-forward intermediates cached across calls: grow on demand, free in
// shutdown. Eliminates 3× cudaMalloc + 3× cudaFree per forward (each ~10–50 µs)
// without changing semantics.
static float*   g_d_x_scale         = nullptr;
static size_t   g_d_x_scale_bytes   = 0;
static int8_t*  g_d_x_i8            = nullptr;
static size_t   g_d_x_i8_bytes      = 0;
static int32_t* g_d_y_i32           = nullptr;
static size_t   g_d_y_i32_bytes     = 0;

// Shape-independent cuBLASLt descriptors: created once at init.
static cublasLtMatmulDesc_t       g_op   = nullptr;
static cublasLtMatmulPreference_t g_pref = nullptr;

// Per-shape cached layouts + heuristic algorithm — populated on first use
// of a (M, N, K) tuple, reused on subsequent calls. A typical transformer
// has only a handful of distinct linear shapes, so this stays tiny.
struct ShapeCache {
    cublasLtMatrixLayout_t la = nullptr;
    cublasLtMatrixLayout_t lb = nullptr;
    cublasLtMatrixLayout_t lc = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    bool has_algo = false;
};
static std::unordered_map<uint64_t, ShapeCache> g_shape_cache;

static inline uint64_t pack_shape(int M, int N, int K) {
    // M, N, K each fit in 21 bits comfortably for transformer-class shapes
    // (max ~2M per dim, vs. typical ~16K).
    return ((uint64_t)(uint32_t)M << 42)
         | ((uint64_t)(uint32_t)N << 21)
         | (uint64_t)(uint32_t)K;
}

static int ensure_buffer(void** ptr, size_t* current, size_t required) {
    if (*current >= required) return 0;
    if (*ptr) { cudaFree(*ptr); *ptr = nullptr; *current = 0; }
    if (cudaMalloc(ptr, required) != cudaSuccess) return -1;
    *current = required;
    return 0;
}

extern "C" int mosyne_bposit_init() {
    if (g_lt) return 0;
    if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
    g_ws_bytes = (size_t)64 << 20;  // 64 MB default workspace
    if (cudaMalloc(&g_ws, g_ws_bytes) != cudaSuccess) {
        cublasLtDestroy(g_lt); g_lt = nullptr; return -2;
    }
    // Shape-independent descriptors — created once.
    if (cublasLtMatmulDescCreate(&g_op, CUBLAS_COMPUTE_32I, CUDA_R_32I)
            != CUBLAS_STATUS_SUCCESS) {
        cudaFree(g_ws); g_ws = nullptr;
        cublasLtDestroy(g_lt); g_lt = nullptr;
        return -3;
    }
    cublasOperation_t T = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(g_op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
    cublasLtMatmulDescSetAttribute(g_op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T);
    if (cublasLtMatmulPreferenceCreate(&g_pref) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(g_op); g_op = nullptr;
        cudaFree(g_ws); g_ws = nullptr;
        cublasLtDestroy(g_lt); g_lt = nullptr;
        return -4;
    }
    cublasLtMatmulPreferenceSetAttribute(
        g_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_ws_bytes, sizeof g_ws_bytes);
    return 0;
}

extern "C" void mosyne_bposit_shutdown() {
    if (g_ws)         { cudaFree(g_ws);         g_ws = nullptr; }
    if (g_d_x_scale)  { cudaFree(g_d_x_scale);  g_d_x_scale = nullptr; g_d_x_scale_bytes = 0; }
    if (g_d_x_i8)     { cudaFree(g_d_x_i8);     g_d_x_i8 = nullptr;    g_d_x_i8_bytes = 0; }
    if (g_d_y_i32)    { cudaFree(g_d_y_i32);    g_d_y_i32 = nullptr;   g_d_y_i32_bytes = 0; }
    for (auto& kv : g_shape_cache) {
        if (kv.second.la) cublasLtMatrixLayoutDestroy(kv.second.la);
        if (kv.second.lb) cublasLtMatrixLayoutDestroy(kv.second.lb);
        if (kv.second.lc) cublasLtMatrixLayoutDestroy(kv.second.lc);
    }
    g_shape_cache.clear();
    if (g_pref) { cublasLtMatmulPreferenceDestroy(g_pref); g_pref = nullptr; }
    if (g_op)   { cublasLtMatmulDescDestroy(g_op);          g_op = nullptr; }
    if (g_lt)   { cublasLtDestroy(g_lt);                    g_lt = nullptr; }
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

// ---- Per-token input scale + quantize (fused) ------------------------------
// One block per token. Threads cooperate via shared memory to find the row's
// max-abs in a single pass, then re-read the row to write quantized int8s.
// Replaces the previous two-kernel sequence (k_per_token_scale +
// k_quantize_input_per_token) with one launch and saves the per-launch
// latency that dominated decode-shape (small-M) overhead.
template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template<> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<int THREADS, typename T>
__global__ void k_per_token_quantize_fused(
    const T* __restrict__ X, int M, int K,
    int8_t* __restrict__ X_q, float* __restrict__ x_scale_out)
{
    int i = blockIdx.x;
    if (i >= M) return;
    __shared__ float smax[THREADS];

    // Stage 1 — each thread computes a partial maxabs over its K-stride.
    float t_max = 0.f;
    for (int k = threadIdx.x; k < K; k += THREADS) {
        float v = fabsf(to_float<T>(X[i + k * M]));
        if (v > t_max) t_max = v;
    }
    smax[threadIdx.x] = t_max;
    __syncthreads();

    // Tree reduction across the block.
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float o = smax[threadIdx.x + s];
            if (o > smax[threadIdx.x]) smax[threadIdx.x] = o;
        }
        __syncthreads();
    }
    float row_max = smax[0];
    float scale = (row_max > 1e-8f) ? (127.f / row_max) : 1.f;
    if (threadIdx.x == 0) x_scale_out[i] = scale;

    // Stage 2 — quantize using the row scale (no need to broadcast: every
    // thread already saw `row_max` after the syncthreads above).
    for (int k = threadIdx.x; k < K; k += THREADS) {
        float v = to_float<T>(X[i + k * M]) * scale;
        v = fminf(127.f, fmaxf(-127.f, v));
        X_q[i + k * M] = (int8_t)__float2int_rn(v);
    }
}

// ---- Per-token × per-channel dequant ---------------------------------------
template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<typename T>
__global__ void k_dequant(const int32_t* y_int, T* y_out,
                          int M, int N,
                          const float* x_scale, const float* w_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int i = idx % M;
    int j = idx / M;
    float v = (float)y_int[idx] / (x_scale[i] * w_scale[j]);
    y_out[idx] = from_float<T>(v);
}

// ---- INT8 IMMA matmul ------------------------------------------------------
// Looks up (M, N, K) in g_shape_cache. On a hit, reuses the cached layouts
// and heuristic algorithm. On a miss, creates and caches them — a transformer
// only sees a handful of distinct linear shapes so the cache stays small.
//
// Eliminates per-call: 1× DescCreate, 3× LayoutCreate, 1× PreferenceCreate,
// 1× AlgoGetHeuristic, plus their matching Destroys. Heuristic lookup
// alone is documented as "microseconds"; in practice on consumer Blackwell
// the whole setup runs ~30–80 µs depending on shape — that's the residual
// small-shape overhead we measured at +65–75 µs in the previous iteration.
static int matmul_int8_inplace(int M, int N, int K,
                               const int8_t* dA, const int8_t* dB,
                               int32_t* dC) {
    if (!g_lt || !g_op || !g_pref) return -1;

    uint64_t key = pack_shape(M, N, K);
    auto it = g_shape_cache.find(key);
    ShapeCache* sc;
    if (it == g_shape_cache.end()) {
        ShapeCache fresh;
        if (cublasLtMatrixLayoutCreate(&fresh.la, CUDA_R_8I, M, K, M)
                != CUBLAS_STATUS_SUCCESS) return -1;
        if (cublasLtMatrixLayoutCreate(&fresh.lb, CUDA_R_8I, K, N, K)
                != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(fresh.la);
            return -1;
        }
        if (cublasLtMatrixLayoutCreate(&fresh.lc, CUDA_R_32I, M, N, M)
                != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(fresh.la);
            cublasLtMatrixLayoutDestroy(fresh.lb);
            return -1;
        }
        int got = 0;
        if (cublasLtMatmulAlgoGetHeuristic(
                g_lt, g_op, fresh.la, fresh.lb, fresh.lc, fresh.lc,
                g_pref, 1, &fresh.heur, &got) == CUBLAS_STATUS_SUCCESS
            && got > 0) {
            fresh.has_algo = true;
        }
        auto inserted = g_shape_cache.emplace(key, fresh);
        sc = &inserted.first->second;
    } else {
        sc = &it->second;
    }
    if (!sc->has_algo) return -2;

    int32_t alpha = 1, beta = 0;
    if (cublasLtMatmul(g_lt, g_op, &alpha, dA, sc->la, dB, sc->lb, &beta,
                       dC, sc->lc, dC, sc->lc,
                       &sc->heur.algo, g_ws, g_ws_bytes, 0)
            == CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    return -2;
}

// ---- Public: device-pointer linear -----------------------------------------
extern "C" int mosyne_bposit_linear_w8a8(
    const float* d_x_fp32, int M, int K,
    const int8_t* d_w_i8, const float* d_w_scale, int N,
    float* d_y_fp32)
{
    if (!g_lt) return -10;
    // Cached intermediates — grow on demand, never shrink, freed in shutdown.
    if (ensure_buffer((void**)&g_d_x_scale, &g_d_x_scale_bytes,
                      (size_t)M * sizeof(float))) return -11;
    if (ensure_buffer((void**)&g_d_x_i8, &g_d_x_i8_bytes,
                      (size_t)M * (size_t)K * sizeof(int8_t))) return -11;
    if (ensure_buffer((void**)&g_d_y_i32, &g_d_y_i32_bytes,
                      (size_t)M * (size_t)N * sizeof(int32_t))) return -11;

    constexpr int FUSED_THREADS = 256;
    k_per_token_quantize_fused<FUSED_THREADS, float><<<M, FUSED_THREADS>>>(
        d_x_fp32, M, K, g_d_x_i8, g_d_x_scale);
    int t = 256;
    int rc = matmul_int8_inplace(M, N, K, g_d_x_i8, d_w_i8, g_d_y_i32);
    if (rc == 0) {
        k_dequant<float><<<(M*N + t - 1) / t, t>>>(
            g_d_y_i32, d_y_fp32, M, N, g_d_x_scale, d_w_scale);
    }
    // No sync — callers order: host wrapper via cudaMemcpy(D2H) (sync,
    // stream-ordered); PyTorch via stream ordering of subsequent ops.
    return rc;
}

// ---- Public: device-pointer linear, bf16 input ----------------------------
// Same as mosyne_bposit_linear_w8a8 but accepts column-major bf16 input
// directly. Saves a bf16→fp32 cast for callers like PyTorch wrappers whose
// activation tensors are bf16 anyway. Output is still fp32 (column-major).
extern "C" int mosyne_bposit_linear_w8a8_bf16(
    const __nv_bfloat16* d_x_bf16, int M, int K,
    const int8_t* d_w_i8, const float* d_w_scale, int N,
    float* d_y_fp32)
{
    if (!g_lt) return -10;
    if (ensure_buffer((void**)&g_d_x_scale, &g_d_x_scale_bytes,
                      (size_t)M * sizeof(float))) return -11;
    if (ensure_buffer((void**)&g_d_x_i8, &g_d_x_i8_bytes,
                      (size_t)M * (size_t)K * sizeof(int8_t))) return -11;
    if (ensure_buffer((void**)&g_d_y_i32, &g_d_y_i32_bytes,
                      (size_t)M * (size_t)N * sizeof(int32_t))) return -11;

    constexpr int FUSED_THREADS = 256;
    k_per_token_quantize_fused<FUSED_THREADS, __nv_bfloat16><<<M, FUSED_THREADS>>>(
        d_x_bf16, M, K, g_d_x_i8, g_d_x_scale);
    int t = 256;
    int rc = matmul_int8_inplace(M, N, K, g_d_x_i8, d_w_i8, g_d_y_i32);
    if (rc == 0) {
        k_dequant<float><<<(M*N + t - 1) / t, t>>>(
            g_d_y_i32, d_y_fp32, M, N, g_d_x_scale, d_w_scale);
    }
    return rc;
}

// ---- Public: device-pointer linear, bf16 input AND bf16 output -------------
// Fully bf16-native hot path: skips both the bf16→fp32 input cast and the
// fp32→bf16 output cast. For PyTorch wrappers whose activations are bf16
// end-to-end this is the no-cast path.
extern "C" int mosyne_bposit_linear_w8a8_bf16_io(
    const __nv_bfloat16* d_x_bf16, int M, int K,
    const int8_t* d_w_i8, const float* d_w_scale, int N,
    __nv_bfloat16* d_y_bf16)
{
    if (!g_lt) return -10;
    if (ensure_buffer((void**)&g_d_x_scale, &g_d_x_scale_bytes,
                      (size_t)M * sizeof(float))) return -11;
    if (ensure_buffer((void**)&g_d_x_i8, &g_d_x_i8_bytes,
                      (size_t)M * (size_t)K * sizeof(int8_t))) return -11;
    if (ensure_buffer((void**)&g_d_y_i32, &g_d_y_i32_bytes,
                      (size_t)M * (size_t)N * sizeof(int32_t))) return -11;

    constexpr int FUSED_THREADS = 256;
    k_per_token_quantize_fused<FUSED_THREADS, __nv_bfloat16><<<M, FUSED_THREADS>>>(
        d_x_bf16, M, K, g_d_x_i8, g_d_x_scale);
    int t = 256;
    int rc = matmul_int8_inplace(M, N, K, g_d_x_i8, d_w_i8, g_d_y_i32);
    if (rc == 0) {
        k_dequant<__nv_bfloat16><<<(M*N + t - 1) / t, t>>>(
            g_d_y_i32, d_y_bf16, M, N, g_d_x_scale, d_w_scale);
    }
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
