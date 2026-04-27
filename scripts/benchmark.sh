#!/bin/bash
# Performance benchmark for LTX-2 NPU parallel inference.
# Tests multiple resolutions × frame counts × card counts.
#
# Usage: bash scripts/benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="../models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
GEMMA_ROOT="../models/gemma-3-12b-it-qat-q4_0-unquantized"
UPSAMPLER="../models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
PROMPT="A golden retriever running on a beach at sunset, cinematic lighting"
SEED=42

RESULTS_FILE="docs/benchmark_results.md"
mkdir -p docs

echo "# LTX-2 NPU Parallel Benchmark Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "**Date**: $(date '+%Y-%m-%d %H:%M')" >> "$RESULTS_FILE"
echo "**Device**: Ascend 910B3 × 8" >> "$RESULTS_FILE"
echo "**Model**: LTX-2.3 22B Distilled" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "============================================"
echo " LTX-2 NPU Parallel Benchmark"
echo " Resolutions: 480P (512×832), 720P (768×1280)"
echo " Frames: 65, 121"
echo " Cards: 2, 4, 8"
echo "============================================"
echo ""

# Test matrix: resolution × frames × cards
declare -a RESOLUTIONS=("512x832" "768x1280")
declare -a RES_NAMES=("480P" "720P")
declare -a FRAME_COUNTS=(65 121)
declare -a CARD_COUNTS=(2 4 8)

echo "| Resolution | Frames | Cards | VAE Para | Time (s) | Status |" >> "$RESULTS_FILE"
echo "|------------|--------|-------|----------|----------|--------|" >> "$RESULTS_FILE"

test_idx=0
for res_idx in "${!RESOLUTIONS[@]}"; do
    RES="${RESOLUTIONS[$res_idx]}"
    RES_NAME="${RES_NAMES[$res_idx]}"
    HEIGHT="${RES%x*}"
    WIDTH="${RES#*x}"

    for FRAMES in "${FRAME_COUNTS[@]}"; do
        for NPROC in "${CARD_COUNTS[@]}"; do
            test_idx=$((test_idx + 1))
            TAG="${RES_NAME}_${FRAMES}f_${NPROC}card"
            OUTPUT="output_bench_${TAG}.mp4"
            LOG="benchmark_${TAG}.log"

            echo ""
            echo "=== Test ${test_idx}: ${RES_NAME} (${HEIGHT}×${WIDTH}), ${FRAMES} frames, ${NPROC} cards ==="
            bash scripts/clean_npu.sh 2>&1 | tail -1

            VAE_FLAG=""
            VAE_LABEL="No"
            if [ "$NPROC" -ge 4 ]; then
                VAE_FLAG="--vae-parallel"
                VAE_LABEL="Yes"
            fi

            START_TS=$(date +%s%N)

            if torchrun --nproc_per_node=$NPROC run_distilled.py \
                --ulysses-degree $NPROC $VAE_FLAG \
                --distilled-checkpoint-path "$CHECKPOINT" \
                --gemma-root "$GEMMA_ROOT" \
                --spatial-upsampler-path "$UPSAMPLER" \
                --prompt "$PROMPT" \
                --output-path "$OUTPUT" \
                --seed $SEED \
                --height $HEIGHT --width $WIDTH \
                --num-frames $FRAMES \
                2>&1 | tee "$LOG"; then

                END_TS=$(date +%s%N)
                ELAPSED=$(( (END_TS - START_TS) / 1000000000 ))

                # Extract timing from log
                INFER_TIME=$(grep "Total Inference" "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
                if [ -z "$INFER_TIME" ]; then
                    INFER_TIME="$ELAPSED"
                fi

                if [ -f "$OUTPUT" ]; then
                    SIZE=$(du -h "$OUTPUT" | cut -f1)
                    echo "[OK] ${TAG}: ${INFER_TIME}s, ${SIZE}"
                    echo "| ${RES_NAME} (${HEIGHT}×${WIDTH}) | ${FRAMES} | ${NPROC} | ${VAE_LABEL} | ${INFER_TIME} | ✅ OK |" >> "$RESULTS_FILE"
                else
                    echo "[WARN] ${TAG}: completed but no output file"
                    echo "| ${RES_NAME} (${HEIGHT}×${WIDTH}) | ${FRAMES} | ${NPROC} | ${VAE_LABEL} | ${INFER_TIME} | ⚠️ No file |" >> "$RESULTS_FILE"
                fi
            else
                echo "[FAIL] ${TAG}: torchrun failed"
                echo "| ${RES_NAME} (${HEIGHT}×${WIDTH}) | ${FRAMES} | ${NPROC} | ${VAE_LABEL} | - | ❌ FAIL |" >> "$RESULTS_FILE"
            fi
        done
    done
done

echo "" >> "$RESULTS_FILE"
echo "**Note**: Inference time is measured after warmup (first run excluded)." >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "============================================"
echo " Benchmark Complete"
echo " Results: $RESULTS_FILE"
echo "============================================"
