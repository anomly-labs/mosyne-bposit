#!/usr/bin/env bash
# bench_transformer_shapes.sh — sweep BF16 HMMA vs INT8 IMMA at LLM-relevant
# matmul shapes. Drives bench_imma_vs_qmma with the (M N K) triples that
# represent the actual hot paths in transformer training and inference.
#
# Reference shapes (Llama-3 / Qwen 7-8B class):
#   hidden = 4096, ffn = 14336, num_heads = 32, head_dim = 128
#
# Run:
#     bash bench_transformer_shapes.sh
# or with a specific GPU:
#     CUDA_VISIBLE_DEVICES=1 bash bench_transformer_shapes.sh

set -uo pipefail
cd "$(dirname "$0")"

# Build if needed
if [[ ! -x bench_imma_vs_qmma ]] || [[ bench_imma_vs_qmma.cu -nt bench_imma_vs_qmma ]]; then
    nvcc -O3 -std=c++17 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_120,code=sm_120 \
         -lcublasLt -lcudart -diag-suppress 186 \
         bench_imma_vs_qmma.cu -o bench_imma_vs_qmma >/dev/null 2>build.err || \
        { cat build.err; exit 1; }
fi

DEVICE=${CUDA_VISIBLE_DEVICES:-1}

declare -a SHAPES=(
    # name                           M     N      K
    "tiny_square                     512   512    512"
    "medium_square                  2048  2048   2048"
    "large_square                   4096  4096   4096"

    # Inference (batch=128 decode, head_dim=128, model_dim=4096, ffn=14336)
    "decode_qkv_proj_b128            128  4096   4096"
    "decode_o_proj_b128              128  4096   4096"
    "decode_ffn_gate_b128            128 14336   4096"
    "decode_ffn_down_b128            128  4096  14336"

    # Training prefill (seq=2048)
    "prefill_qkv_proj_seq2048       2048  4096   4096"
    "prefill_ffn_gate_seq2048       2048 14336   4096"
    "prefill_ffn_down_seq2048       2048  4096  14336"

    # Attention scores (B*H = 32, seq*seq)
    "attn_qk_seq2048_h128           2048  2048    128"
)

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Transformer matmul shapes: BF16 HMMA vs INT8 IMMA (bposit-via-IMMA path)"
echo "  device: CUDA_VISIBLE_DEVICES=$DEVICE"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i $DEVICE 2>/dev/null \
  | sed 's/^/  gpu:    /'
echo "═══════════════════════════════════════════════════════════════════════════"
printf "\n%-32s %-18s %10s %10s %8s %10s %10s %10s\n" \
    "name" "shape M×N×K" "BF16_ms" "BF16_TFs" "ratio" "INT8_ms" "INT8_TFs" "INT8/BF16"
printf '%.s─' {1..115}; echo

for spec in "${SHAPES[@]}"; do
    read -ra parts <<< "$spec"
    name=${parts[0]}; M=${parts[1]}; N=${parts[2]}; K=${parts[3]}
    out=$(CUDA_VISIBLE_DEVICES=$DEVICE ./bench_imma_vs_qmma "$M" "$N" "$K" 2>&1)
    rc=$?
    if [[ $rc -ne 0 ]]; then
        printf "%-32s %5d×%-5d×%-5d  [run failed rc=%d]\n" "$name" "$M" "$N" "$K" "$rc"
        continue
    fi
    bf_line=$(echo "$out" | grep "BF16 in/out")
    i8_line=$(echo "$out" | grep "INT8 in,")
    bf_ms=$(echo "$bf_line" | sed -E 's/.*ms= *([0-9.]+).*/\1/')
    bf_tf=$(echo "$bf_line" | sed -E 's/.*TFLOPS= *([0-9.]+).*/\1/')
    i8_ms=$(echo "$i8_line" | sed -E 's/.*ms= *([0-9.]+).*/\1/')
    i8_tf=$(echo "$i8_line" | sed -E 's/.*TFLOPS= *([0-9.]+).*/\1/')

    ratio_speed=$(awk -v a="$bf_ms" -v b="$i8_ms" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else print "—" }')
    ratio_tf=$(awk -v a="$i8_tf" -v b="$bf_tf" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else print "—" }')
    printf "%-32s %5d×%-5d×%-5d  %10s %10s %8s %10s %10s %10s\n" \
        "$name" "$M" "$N" "$K" "${bf_ms:-?}" "${bf_tf:-?}" "—" "${i8_ms:-?}" "${i8_tf:-?}" "$ratio_tf"
done

echo
echo "Notes:"
echo "  - BF16 path on Blackwell-consumer routes through HMMA (per cuBLASLt 12.8 algo probe);"
echo "    cuBLASLt does NOT expose FP4 QMMA via public API at this version."
echo "  - INT8 path is what bposit-via-IMMA uses (Path C: bposit8 → INT8 → IMMA → INT32 → bposit32)."
echo "  - INT8/BF16 ratio ~= 1.0 means parity. >1.0 means INT8 wins on this shape."
echo "  - Memory-bound (skinny) shapes favor INT8 for byte traffic."
echo "  - Compute-bound (large square) shapes favor whichever has higher tensor-core peak."
