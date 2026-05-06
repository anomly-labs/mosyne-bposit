#!/usr/bin/env bash
# Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0

# run_public_tests.sh — single-command verification of the bposit ALU
# primitives + AI/ML demos described in the whitepaper.

set -uo pipefail
cd "$(dirname "$0")"

DEVICE=${CUDA_VISIBLE_DEVICES:-0}
NVCC_FLAGS="-O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -diag-suppress 186"

if [[ ! -f bposit16_luts.cuh ]] || [[ build_bposit_luts.py -nt bposit16_luts.cuh ]]; then
    echo "[build] regenerating bposit16_luts.cuh from Python reference..."
    PYTHON=${PYTHON:-python3}
    $PYTHON build_bposit_luts.py | tail -1
fi

PASS=0; FAIL=0
declare -a RESULTS=()

build_and_run() {
    local label="$1" cu="$2" bin="$3" libs="$4"
    shift 4
    if [[ ! -x "$bin" ]] || [[ "$cu" -nt "$bin" ]] || [[ bposit16_luts.cuh -nt "$bin" ]]; then
        if ! nvcc $NVCC_FLAGS $libs -lcudart "$cu" -o "$bin" 2>build.err; then
            RESULTS+=("FAIL  $label  (build)"); ((FAIL++)); return
        fi
    fi
    out=$(CUDA_VISIBLE_DEVICES=$DEVICE ./"$bin" "$@" 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]] && ! echo "$out" | grep -qE '\bFAIL\b'; then
        bench=$(echo "$out" | grep -E '^BENCH' | head -1)
        if [[ -n "$bench" ]]; then RESULTS+=("PASS  $label  $bench"); else RESULTS+=("PASS  $label"); fi
        ((PASS++))
    else
        RESULTS+=("FAIL  $label  (rc=$rc)"); ((FAIL++))
    fi
}

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  mosyne-bposit public test suite"
echo "  device:  CUDA_VISIBLE_DEVICES=$DEVICE"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i $DEVICE 2>/dev/null \
  | sed 's/^/  gpu:     /'
echo "═══════════════════════════════════════════════════════════════════════════"
echo
echo "─ ALU primitive validation (bit-exact vs Python reference) ───────────────"
build_and_run "bposit16_sqrt_lut"              test_sqrt.cu                  test_sqrt          ""
build_and_run "bposit16_recip_lut"             test_recip.cu                 test_recip         ""
build_and_run "bposit16_exp2_lut"              test_exp2.cu                  test_exp2          ""
build_and_run "quire256_to_bposit16"           test_quire_to_bp16.cu         test_q2bp16        ""
build_and_run "bposit16_add(exact via quire)"  test_bposit16_add.cu          test_add           ""
build_and_run "bposit16_mul(log+exp2)"         test_bposit16_mul.cu          test_mul           ""
build_and_run "int32_to_bposit32"              test_int32_to_bposit32.cu     test_int32         ""
build_and_run "ones_complement_trick"          test_ones_complement_trick.cu test_ones_comp     ""

echo
echo "─ AI/ML demonstrations (the headline results) ────────────────────────────"
build_and_run "determinism_headline"            test_determinism_headline.cu test_determinism   ""
build_and_run "ffn_layer_bench(per-tensor PTQ)" test_ffn_layer_bench.cu      test_ffn_layer     "-lcublasLt"
build_and_run "ffn_layer_perchannel(W8 PTQ)"    test_ffn_layer_perch.cu      test_ffn_perch     "-lcublasLt"
build_and_run "ffn_layer_pertoken(W8A8 PTQ)"    test_ffn_layer_pertoken.cu   test_ffn_pt        "-lcublasLt"

echo
echo "─ Throughput benchmarks ──────────────────────────────────────────────────"
build_and_run "bench_imma_vs_qmma(M=2048)"     bench_imma_vs_qmma.cu         bench_imma_vs_qmma "-lcublasLt"   2048

echo
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  SUMMARY: $PASS PASS, $FAIL FAIL"
echo "═══════════════════════════════════════════════════════════════════════════"
echo
echo "Multi-trial mean ± stddev sweep across 11 transformer matmul shapes:"
echo "  bash bench_robust_sweep.sh"
exit $FAIL