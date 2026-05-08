// bench_robust.cu — rigorous matmul bench with trial variance + cuBLASLt
// algorithm sweep. Replaces single-shot bench_imma_vs_qmma for whitepaper
// numbers that need to survive scrutiny.
//
// What it does, per (M, N, K) shape:
//   1. Runs both BF16 HMMA and INT8 IMMA through cuBLASLt heuristic algo[0],
//      doing N_TRIALS independent trials of N_REPS_PER_TRIAL launches each.
//      Reports mean, stddev, min, max in milliseconds and TFLOPS/TPOPS.
//   2. Optionally enumerates ALL cuBLASLt algorithms via algoGetIds for the
//      INT8 path (env MOSYNE_BENCH_ALGO_SWEEP=1) and reports the FASTEST one.
//      This catches cases where the heuristic mis-picks for a given shape.
//
// Usage:
//   ./bench_robust 2048 2048 2048           # single shape, default 5 trials
//   ./bench_robust 128 4096 14336 8 50      # 8 trials, 50 reps each
//   MOSYNE_BENCH_ALGO_SWEEP=1 ./bench_robust 128 4096 14336
//
// Build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//        -gencode arch=compute_120,code=sm_120 \
//        -lcublasLt -lcudart bench_robust.cu -o bench_robust

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas %s @%d: %d\n", #x, __LINE__, (int)s); \
    std::exit(3); } } while (0)

struct Stats {
    double mean_ms = 0, std_ms = 0, min_ms = 0, max_ms = 0;
    double mean_tflops = 0, min_tflops = 0, max_tflops = 0;
    int n_trials = 0;
    bool ok = false;
};

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}
static double stddev(const std::vector<double>& v) {
    double m = mean(v); double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / std::max<size_t>(1, v.size() - 1));
}

// One trial = N_REPS launches; returns ms_avg per launch over the trial.
static double time_one_trial(cublasLtHandle_t lt, int M, int N, int K,
                             cudaDataType_t in_t, cudaDataType_t out_t,
                             cublasComputeType_t comp_t, cudaDataType_t scale_t,
                             void* dA, void* dB, void* dC,
                             void* dWS, size_t ws,
                             const cublasLtMatmulAlgo_t* algo,
                             int n_reps)
{
    cublasLtMatmulDesc_t op;
    cublasOperation_t T = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, comp_t, scale_t));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T));
    cublasLtMatrixLayout_t la, lb, lc;
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&la, in_t,  M, K, M));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lb, in_t,  K, N, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lc, out_t, M, N, M));

    union { float f; int32_t i; } alpha_v, beta_v;
    if (scale_t == CUDA_R_32I) { alpha_v.i = 1; beta_v.i = 0; }
    else                        { alpha_v.f = 1.0f; beta_v.f = 0.0f; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Warmup
    CUBLAS_OK(cublasLtMatmul(lt, op, &alpha_v, dA, la, dB, lb, &beta_v,
                             dC, lc, dC, lc, algo, dWS, ws, 0));
    CUDA_OK(cudaDeviceSynchronize());

    cudaEventRecord(e0);
    for (int r = 0; r < n_reps; ++r) {
        CUBLAS_OK(cublasLtMatmul(lt, op, &alpha_v, dA, la, dB, lb, &beta_v,
                                 dC, lc, dC, lc, algo, dWS, ws, 0));
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float total_ms; cudaEventElapsedTime(&total_ms, e0, e1);

    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)total_ms / n_reps;
}

static cublasLtMatmulAlgo_t get_heuristic_algo(
    cublasLtHandle_t lt, int M, int N, int K,
    cudaDataType_t in_t, cudaDataType_t out_t,
    cublasComputeType_t comp_t, cudaDataType_t scale_t,
    size_t ws, bool* ok_out)
{
    cublasLtMatmulDesc_t op;
    cublasOperation_t T = CUBLAS_OP_N;
    cublasLtMatmulDescCreate(&op, comp_t, scale_t);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T);
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, in_t,  M, K, M);
    cublasLtMatrixLayoutCreate(&lb, in_t,  K, N, K);
    cublasLtMatrixLayoutCreate(&lc, out_t, M, N, M);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    cublasLtMatmulHeuristicResult_t heur[1]; int got = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, heur, &got);
    cublasLtMatmulAlgo_t out{};
    *ok_out = (got > 0);
    if (got > 0) out = heur[0].algo;
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);
    return out;
}

static Stats bench(cublasLtHandle_t lt, int M, int N, int K,
                   cudaDataType_t in_t, cudaDataType_t out_t,
                   cublasComputeType_t comp_t, cudaDataType_t scale_t,
                   void* dA, void* dB, void* dC,
                   void* dWS, size_t ws,
                   const cublasLtMatmulAlgo_t* algo,
                   int n_trials, int n_reps)
{
    Stats s;
    s.n_trials = n_trials;
    std::vector<double> times;
    for (int t = 0; t < n_trials; ++t) {
        double ms = time_one_trial(lt, M, N, K, in_t, out_t, comp_t, scale_t,
                                   dA, dB, dC, dWS, ws, algo, n_reps);
        times.push_back(ms);
    }
    s.mean_ms = mean(times);
    s.std_ms  = stddev(times);
    s.min_ms  = *std::min_element(times.begin(), times.end());
    s.max_ms  = *std::max_element(times.begin(), times.end());
    double macs = double(M) * double(N) * double(K);
    auto tflops_at = [macs](double ms){ return 2.0 * macs / 1e12 / (ms * 1e-3); };
    s.mean_tflops = tflops_at(s.mean_ms);
    s.min_tflops  = tflops_at(s.max_ms);
    s.max_tflops  = tflops_at(s.min_ms);
    s.ok = true;
    return s;
}

// Sweep all available cuBLASLt algorithms for the INT8 path. Returns the
// stats for the best one + how many algorithms were tried.
struct AlgoSweepResult { Stats best; int n_tried = 0; int n_ok = 0; };

static AlgoSweepResult sweep_int8_algos(
    cublasLtHandle_t lt, int M, int N, int K,
    void* dA, void* dB, void* dC,
    void* dWS, size_t ws,
    int n_trials_per_algo, int n_reps_per_trial)
{
    AlgoSweepResult result;
    result.best.mean_ms = 1e30;
    int algo_ids[64]; int n_algos = 0;
    cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32I, CUDA_R_32I,
                             CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
                             64, algo_ids, &n_algos);
    result.n_tried = n_algos;
    for (int i = 0; i < n_algos; ++i) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32I, CUDA_R_32I,
                                   CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
                                   algo_ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        // Quick capability check: try one matmul; skip on error
        cublasLtMatmulDesc_t op;
        cublasOperation_t T = CUBLAS_OP_N;
        if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I)
                != CUBLAS_STATUS_SUCCESS) continue;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T);
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof T);
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8I,  M, K, M);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I,  K, N, K);
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, N, M);
        int32_t alpha = 1, beta = 0;
        cublasStatus_t rc = cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb,
                                           &beta, dC, lc, dC, lc, &algo,
                                           dWS, ws, 0);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatmulDescDestroy(op);
        if (rc != CUBLAS_STATUS_SUCCESS) continue;
        ++result.n_ok;
        Stats s = bench(lt, M, N, K, CUDA_R_8I, CUDA_R_32I,
                        CUBLAS_COMPUTE_32I, CUDA_R_32I,
                        dA, dB, dC, dWS, ws, &algo,
                        n_trials_per_algo, n_reps_per_trial);
        if (s.ok && s.mean_ms < result.best.mean_ms) {
            result.best = s;
        }
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s M N K [n_trials=5] [n_reps=20]\n", argv[0]);
        return 1;
    }
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    int n_trials = (argc > 4) ? std::atoi(argv[4]) : 5;
    int n_reps   = (argc > 5) ? std::atoi(argv[5]) : 20;
    bool sweep   = std::getenv("MOSYNE_BENCH_ALGO_SWEEP") != nullptr;

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("shape:  M=%d N=%d K=%d → %.2f GMACs/call\n",
                M, N, K, double(M)*N*K / 1e9);
    std::printf("trials: %d × %d reps each\n", n_trials, n_reps);
    std::printf("\n");

    cublasLtHandle_t lt;
    CUBLAS_OK(cublasLtCreate(&lt));
    size_t ws = (size_t)64 << 20;
    void* dWS; CUDA_OK(cudaMalloc(&dWS, ws));

    void *dA_bf, *dB_bf, *dC_bf;
    CUDA_OK(cudaMalloc(&dA_bf, size_t(M)*K*2));
    CUDA_OK(cudaMalloc(&dB_bf, size_t(K)*N*2));
    CUDA_OK(cudaMalloc(&dC_bf, size_t(M)*N*2));
    CUDA_OK(cudaMemset(dA_bf, 0x3F, size_t(M)*K*2));
    CUDA_OK(cudaMemset(dB_bf, 0x3F, size_t(K)*N*2));

    void *dA_i8, *dB_i8, *dC_i32;
    CUDA_OK(cudaMalloc(&dA_i8, size_t(M)*K));
    CUDA_OK(cudaMalloc(&dB_i8, size_t(K)*N));
    CUDA_OK(cudaMalloc(&dC_i32, size_t(M)*N*4));
    CUDA_OK(cudaMemset(dA_i8, 0x01, size_t(M)*K));
    CUDA_OK(cudaMemset(dB_i8, 0x01, size_t(K)*N));

    bool bf_ok, i8_ok;
    cublasLtMatmulAlgo_t algo_bf = get_heuristic_algo(
        lt, M, N, K, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, ws, &bf_ok);
    cublasLtMatmulAlgo_t algo_i8 = get_heuristic_algo(
        lt, M, N, K, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, ws, &i8_ok);

    Stats bf{}, i8{};
    if (bf_ok) bf = bench(lt, M, N, K, CUDA_R_16BF, CUDA_R_16BF,
                          CUBLAS_COMPUTE_32F, CUDA_R_32F,
                          dA_bf, dB_bf, dC_bf, dWS, ws, &algo_bf, n_trials, n_reps);
    if (i8_ok) i8 = bench(lt, M, N, K, CUDA_R_8I, CUDA_R_32I,
                          CUBLAS_COMPUTE_32I, CUDA_R_32I,
                          dA_i8, dB_i8, dC_i32, dWS, ws, &algo_i8, n_trials, n_reps);

    auto print = [](const char* label, const Stats& s, const char* unit) {
        if (!s.ok) { std::printf("  %-44s [no algo]\n", label); return; }
        std::printf("  %-44s ms=%7.4f ± %5.4f  (min %.4f / max %.4f)  %s=%6.1f (range %.1f–%.1f)\n",
                    label, s.mean_ms, s.std_ms, s.min_ms, s.max_ms,
                    unit, s.mean_tflops, s.min_tflops, s.max_tflops);
    };
    std::printf("=== heuristic algo[0], %d trials × %d reps ===\n", n_trials, n_reps);
    print("BF16 in/out, F32 accum (HMMA path)", bf, "TFLOPS");
    print("INT8 in, INT32 out (IMMA path)",     i8, "TPOPS");

    if (bf.ok && i8.ok) {
        double ratio = bf.mean_ms / i8.mean_ms;   // ratio > 1 → INT8 faster
        std::printf("\n  ratio (INT8 / BF16)     : %.3fx %s\n",
                    ratio,
                    ratio >= 1.0 ? "(INT8 wins)" :
                    ratio >= 0.95 ? "(parity)" : "(BF16 wins)");
    }

    if (sweep && i8_ok) {
        std::printf("\n=== algorithm sweep for INT8 path (env MOSYNE_BENCH_ALGO_SWEEP=1) ===\n");
        auto r = sweep_int8_algos(lt, M, N, K, dA_i8, dB_i8, dC_i32,
                                   dWS, ws, std::max(2, n_trials/2), n_reps);
        std::printf("  enumerated %d algos, %d compatible with this shape\n",
                    r.n_tried, r.n_ok);
        if (r.best.ok) {
            print("BEST INT8 algo across sweep", r.best);
            if (bf.ok) {
                double ratio = bf.mean_ms / r.best.mean_ms;
                std::printf("\n  ratio (best-INT8 / BF16): %.3fx\n", ratio);
            }
        }
    }

    cudaFree(dA_bf); cudaFree(dB_bf); cudaFree(dC_bf);
    cudaFree(dA_i8); cudaFree(dB_i8); cudaFree(dC_i32);
    cudaFree(dWS);
    cublasLtDestroy(lt);
    return 0;
}
