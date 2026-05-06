// bench_imma_vs_qmma.cu — measured throughput of INT8 IMMA vs BF16 (which
// cuBLASLt routes through FP4 QMMA on Blackwell, per the morning's SASS RE)
// at the same GEMM shape, on the same GPU. The headline question:
//
//     If INT8 IMMA is within ~2× of FP4 QMMA on the 5090, then Mosyne's
//     Path-C b-posit-via-IMMA path gives near-tensor-core-peak throughput
//     while keeping the spec's "no IEEE float" rule intact.
//
// Build:
//     nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//          -gencode arch=compute_120,code=sm_120 \
//          -lcublasLt -lcudart \
//          bench_imma_vs_qmma.cu -o bench_imma_vs_qmma
//
// Run:
//     ./bench_imma_vs_qmma                       # default M=N=K=2048
//     ./bench_imma_vs_qmma 4096                  # M=N=K=4096
//     CUDA_VISIBLE_DEVICES=1 ./bench_imma_vs_qmma 4096   # 3090 instead

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

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


struct BenchResult {
    const char* label;
    double ms_avg;
    double tops;      // tera-MAC-ops per second (1 MAC = 2 ops, but we report MACs not ops)
    double tflops;    // 2 × MACs (the FLOPS-equivalent count)
    int n_repeats;
    bool ok;
};


static BenchResult bench_one(
    cublasLtHandle_t lt,
    int M, int N, int K,
    cudaDataType_t in_type,
    cudaDataType_t out_type,
    cublasComputeType_t compute_type,
    cudaDataType_t scale_type,
    void* dA, void* dB, void* dC, void* dWS, size_t ws_bytes,
    const char* label)
{
    cublasLtMatmulDesc_t op = nullptr;
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescCreate(&op, compute_type, scale_type));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    cublasLtMatrixLayout_t la, lb, lc;
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&la, in_type, M, K, M));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lb, in_type, K, N, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&lc, out_type, M, N, M));

    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heur[8]{};
    int got = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        lt, op, la, lb, lc, lc, pref, 8, heur, &got);
    if (hs != CUBLAS_STATUS_SUCCESS || got == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatmulDescDestroy(op);
        return {label, 0, 0, 0, 0, false};
    }

    // alpha/beta in scale_type
    union { float f; int32_t i; __nv_bfloat16 bf; } alpha_v, beta_v;
    alpha_v.f = 1.0f; beta_v.f = 0.0f;

    // Warm-up + timing
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    int reps = 20;
    // Warm-up
    CUBLAS_OK(cublasLtMatmul(lt, op, &alpha_v, dA, la, dB, lb, &beta_v, dC, lc, dC, lc,
                             &heur[0].algo, dWS, ws_bytes, /*stream*/ 0));
    CUDA_OK(cudaDeviceSynchronize());

    double total_ms = 0;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0);
        CUBLAS_OK(cublasLtMatmul(lt, op, &alpha_v, dA, la, dB, lb, &beta_v, dC, lc, dC, lc,
                                 &heur[0].algo, dWS, ws_bytes, /*stream*/ 0));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        total_ms += ms;
    }
    double ms_avg = total_ms / reps;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(op);

    double macs = double(M) * double(N) * double(K);
    double mac_per_s = macs / (ms_avg * 1e-3);
    double tops = mac_per_s / 1e12;
    double tflops = 2.0 * tops;
    return {label, ms_avg, tops, tflops, reps, true};
}


int main(int argc, char** argv) {
    // Usage:
    //   ./bench_imma_vs_qmma                  # M=N=K=2048
    //   ./bench_imma_vs_qmma 4096             # M=N=K=4096
    //   ./bench_imma_vs_qmma 1024 4096 4096   # explicit M N K
    int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
    int N = (argc > 2) ? std::atoi(argv[2]) : M;
    int K = (argc > 3) ? std::atoi(argv[3]) : M;

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("shape:  M=N=K=%d  →  %.2f GMACs/call\n",
                M, double(M)*N*K / 1e9);
    std::printf("\n");

    cublasLtHandle_t lt;
    CUBLAS_OK(cublasLtCreate(&lt));

    // Workspace — small to fit alongside running services on the 5090
    size_t ws_bytes = size_t(32) << 20;  // 32 MB
    void* dWS = nullptr;
    CUDA_OK(cudaMalloc(&dWS, ws_bytes));

    // Allocate A, B, C in different sizes per test below.
    // For BF16 case: in/out 2 bytes/elem, A=B=M*K*2, C=M*N*2
    void* dA_bf = nullptr; void* dB_bf = nullptr; void* dC_bf = nullptr;
    CUDA_OK(cudaMalloc(&dA_bf, size_t(M)*K*2));
    CUDA_OK(cudaMalloc(&dB_bf, size_t(K)*N*2));
    CUDA_OK(cudaMalloc(&dC_bf, size_t(M)*N*2));
    CUDA_OK(cudaMemset(dA_bf, 0x3F, size_t(M)*K*2));  // ~0.5 in bf16
    CUDA_OK(cudaMemset(dB_bf, 0x3F, size_t(K)*N*2));

    // INT8 case: in 1 byte/elem, out 4 bytes (S32)
    void* dA_i8 = nullptr; void* dB_i8 = nullptr; void* dC_i32 = nullptr;
    CUDA_OK(cudaMalloc(&dA_i8, size_t(M)*K*1));
    CUDA_OK(cudaMalloc(&dB_i8, size_t(K)*N*1));
    CUDA_OK(cudaMalloc(&dC_i32, size_t(M)*N*4));
    CUDA_OK(cudaMemset(dA_i8, 0x01, size_t(M)*K*1));
    CUDA_OK(cudaMemset(dB_i8, 0x01, size_t(K)*N*1));

    // ---- BF16 GEMM (Blackwell consumer routes to FP4 QMMA per the morning RE) ----
    BenchResult r_bf = bench_one(
        lt, M, N, K,
        /*in*/ CUDA_R_16BF, /*out*/ CUDA_R_16BF,
        /*compute*/ CUBLAS_COMPUTE_32F, /*scale*/ CUDA_R_32F,
        dA_bf, dB_bf, dC_bf, dWS, ws_bytes,
        "BF16 in/out, F32 accum (→ FP4 QMMA on sm_120)");

    // ---- INT8 GEMM (the IMMA path — what bposit8-via-IMMA would use) ----
    BenchResult r_i8 = bench_one(
        lt, M, N, K,
        /*in*/ CUDA_R_8I, /*out*/ CUDA_R_32I,
        /*compute*/ CUBLAS_COMPUTE_32I, /*scale*/ CUDA_R_32I,
        dA_i8, dB_i8, dC_i32, dWS, ws_bytes,
        "INT8 in, INT32 out (= IMMA path)");

    // ---- Report ----
    std::printf("=== %d×%d×%d GEMM throughput, cuBLASLt heuristic algo[0] ===\n", M, N, K);
    auto print = [&](const BenchResult& r) {
        if (!r.ok) {
            std::printf("  %-50s  %s\n", r.label, "[no algo selected]");
            return;
        }
        std::printf("  %-50s  ms=%7.3f  TOPS=%6.1f  TFLOPS=%6.1f  (reps=%d)\n",
                    r.label, r.ms_avg, r.tops, r.tflops, r.n_repeats);
    };
    print(r_bf);
    print(r_i8);

    if (r_bf.ok && r_i8.ok) {
        double ratio = r_i8.tops / r_bf.tops;
        std::printf("\nIMMA / QMMA throughput ratio: %.2fx\n", ratio);
        if (ratio > 0.5)
            std::printf("→ INT8 IMMA achieves >=50%% of BF16/FP4 throughput.\n"
                        "  Path C (bposit8 via IMMA) is competitive with FP4 QMMA on this chip.\n");
        else
            std::printf("→ INT8 IMMA is significantly slower than BF16/FP4 here.\n"
                        "  cuBLASLt may be using sm_80 forwardCompat for INT8 (not sm_120 native).\n");
    }

    cudaFree(dA_bf); cudaFree(dB_bf); cudaFree(dC_bf);
    cudaFree(dA_i8); cudaFree(dB_i8); cudaFree(dC_i32);
    cudaFree(dWS);
    cublasLtDestroy(lt);
    return 0;
}
