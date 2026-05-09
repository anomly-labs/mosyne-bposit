#!/usr/bin/env bash
# demo.sh — self-contained reproducibility demo.
#
# Builds and runs deploy/5dst_cuda/test_determinism_headline.cu on every
# available NVIDIA GPU on this host, captures the bposit16+quire hash and
# the IEEE FP32+atomicAdd hashes from each, and prints a one-table
# comparison.
#
# Usage:
#     ./demo.sh                 # build (if missing) + run
#     ./demo.sh --rebuild       # force rebuild
#     ./demo.sh --gpus 0,1      # restrict to specific GPU indices
#
# Requires:
#     nvcc 12.x on PATH
#     nvidia-smi on PATH
#     bposit16_luts.cuh present in ../deploy/5dst_cuda/ (regenerated
#         by the existing build_bposit_luts.py if missing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
KERNEL_DIR="$REPO_ROOT/deploy/5dst_cuda"
BINARY="$KERNEL_DIR/test_determinism"

REBUILD=0
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=1; shift ;;
        --gpus)    GPUS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---- build if needed ------------------------------------------------------
if [[ "$REBUILD" -eq 1 ]] || [[ ! -x "$BINARY" ]]; then
    echo "[build] compiling test_determinism_headline.cu (sm_86 + sm_120) ..."
    [ -f "$KERNEL_DIR/bposit16_luts.cuh" ] || {
        echo "[build] regenerating bposit16_luts.cuh from Python reference"
        ( cd "$KERNEL_DIR" && "$REPO_ROOT/.venv/bin/python" build_bposit_luts.py | tail -1 )
    }
    nvcc -O3 -std=c++17 -diag-suppress 186 \
         -gencode arch=compute_86,code=sm_86 \
         -gencode arch=compute_120,code=sm_120 \
         "$KERNEL_DIR/test_determinism_headline.cu" -o "$BINARY"
fi

# ---- enumerate GPUs -------------------------------------------------------
if [[ -n "$GPUS" ]]; then
    GPU_IDS=(${GPUS//,/ })
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "no NVIDIA GPUs found" >&2
    exit 1
fi

# ---- run on each GPU ------------------------------------------------------
declare -A GPU_NAME GPU_BPOSIT_HASH GPU_FP32_HASHES
for idx in "${GPU_IDS[@]}"; do
    name=$(nvidia-smi -i "$idx" --query-gpu=name --format=csv,noheader)
    GPU_NAME[$idx]="$name"
    log="/tmp/repro_demo_gpu${idx}.log"
    CUDA_VISIBLE_DEVICES="$idx" "$BINARY" > "$log" 2>&1 || {
        echo "[run] FAILED on GPU $idx ($name) — see $log" >&2
        continue
    }
    # Extract: bposit16=0xXXXX (deduped) and the 5 FP32 bits=0xYYYY values
    GPU_BPOSIT_HASH[$idx]=$(grep -oE 'bposit16=0x[0-9a-f]{4}' "$log" | sort -u | head -1)
    GPU_FP32_HASHES[$idx]=$(grep -oE 'bits=0x[0-9a-f]{8}' "$log" | sort -u | tr '\n' ' ')
done

# ---- comparison table -----------------------------------------------------
echo
echo "================================================================"
echo "  Mosyne bposit reproducibility demo — sum of 65 K log-uniform "
echo "  values, IEEE FP32 atomicAdd vs bposit16 + quire256, 5 runs each."
echo "================================================================"
printf "\n%-3s  %-30s  %-22s  %s\n" "GPU" "Name" "bposit (5 runs)" "FP32 atomicAdd (5 runs, unique)"
printf "%-3s  %-30s  %-22s  %s\n"   "---" "----" "----------------" "-------------------------------"
for idx in "${GPU_IDS[@]}"; do
    bp=${GPU_BPOSIT_HASH[$idx]:-?}
    fp=${GPU_FP32_HASHES[$idx]:-?}
    n_fp=$(echo "$fp" | wc -w)
    printf "%-3s  %-30s  %-22s  %s (%d unique)\n" \
        "$idx" "${GPU_NAME[$idx]}" "$bp" "$fp" "$n_fp"
done

# ---- cross-GPU verdict ----------------------------------------------------
unique_bposit=$(printf '%s\n' "${GPU_BPOSIT_HASH[@]}" | sort -u | wc -l)
echo
echo "----------------------------------------------------------------"
if [[ "$unique_bposit" -eq 1 ]]; then
    echo "  ✓ bposit produced the SAME hash on all ${#GPU_IDS[@]} GPU(s) — "
    echo "    bit-exact reproducibility across hardware generations."
else
    echo "  ✗ bposit produced $unique_bposit different hashes across GPUs."
    echo "    Expected 1; this is a regression worth investigating."
fi

# Concatenate all FP32 hashes from all GPUs and dedupe.
all_fp32=""
for idx in "${GPU_IDS[@]}"; do
    all_fp32+="${GPU_FP32_HASHES[$idx]} "
done
unique_fp32=$(echo "$all_fp32" | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l)
total_fp32=$(echo "$all_fp32" | tr ' ' '\n' | grep -v '^$' | wc -l)
echo "  IEEE FP32 atomicAdd produced $unique_fp32 distinct hashes across"
echo "  ${total_fp32} runs total (5 per GPU × ${#GPU_IDS[@]} GPUs)."
echo "----------------------------------------------------------------"
