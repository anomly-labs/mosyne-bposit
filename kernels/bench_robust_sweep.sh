#!/usr/bin/env bash
# Copyright (c) 2026 Ry Bruscoe and Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0

# bench_robust_sweep.sh — multi-trial mean±stddev sweep + algorithm enumeration
# for the suspect skinny-K shapes. Replaces single-shot bench_transformer_shapes.sh
# for whitepaper-grade numbers.

set -uo pipefail
cd "$(dirname "$0")"

if [[ ! -x bench_robust ]] || [[ bench_robust.cu -nt bench_robust ]]; then
    echo "[build] bench_robust"
    nvcc -O3 -std=c++17 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_120,code=sm_120 \
         -lcublasLt -lcudart -diag-suppress 186 \
         bench_robust.cu -o bench_robust >/dev/null 2>build.err || { cat build.err; exit 1; }
fi

DEVICE=${CUDA_VISIBLE_DEVICES:-0}
N_TRIALS=${MOSYNE_BENCH_TRIALS:-8}
N_REPS=${MOSYNE_BENCH_REPS:-30}

declare -a SHAPES=(
    "tiny_square                              512  512    512"
    "medium_square                           2048 2048   2048"
    "large_square                            4096 4096   4096"
    "decode_qkv_proj_b128                     128 4096   4096"
    "decode_o_proj_b128                       128 4096   4096"
    "decode_ffn_gate_b128                     128 14336  4096"
    "decode_ffn_down_b128                     128 4096  14336"
    "prefill_qkv_proj_seq2048                2048 4096   4096"
    "prefill_ffn_gate_seq2048                2048 14336  4096"
    "prefill_ffn_down_seq2048                2048 4096  14336"
    "attn_qk_seq2048_h128                    2048 2048    128"
)

declare -A SUSPECT_SHAPES=(
    [decode_qkv_proj_b128]=1
    [decode_o_proj_b128]=1
    [decode_ffn_down_b128]=1
)

echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "  robust matmul sweep — $N_TRIALS trials × $N_REPS reps each"
echo "  device: CUDA_VISIBLE_DEVICES=$DEVICE"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i $DEVICE 2>/dev/null \
  | sed 's/^/  gpu:    /'
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo

run_shape() {
    local name="$1"; local M="$2"; local N="$3"; local K="$4"
    echo "─── $name  ${M}×${N}×${K} ───"
    CUDA_VISIBLE_DEVICES=$DEVICE ./bench_robust "$M" "$N" "$K" "$N_TRIALS" "$N_REPS" 2>&1 \
        | grep -vE "^(device|trials|shape):" | tail -n +2
    if [[ -n "${SUSPECT_SHAPES[$name]:-}" ]]; then
        echo "  ─── algorithm sweep (suspect shape) ───"
        MOSYNE_BENCH_ALGO_SWEEP=1 CUDA_VISIBLE_DEVICES=$DEVICE \
            ./bench_robust "$M" "$N" "$K" "$N_TRIALS" "$N_REPS" 2>&1 \
            | sed -n '/algorithm sweep/,$p' | tail -n +2
    fi
    echo
}

for spec in "${SHAPES[@]}"; do
    read -ra parts <<< "$spec"
    run_shape "${parts[0]}" "${parts[1]}" "${parts[2]}" "${parts[3]}"
done

echo "─── done ───"