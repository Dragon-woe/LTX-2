#!/bin/bash
# Run all benchmark test cases sequentially with results collection.

set -e
cd "$(dirname "$0")/.."

PYTHON="/root/miniconda3/envs/torch280_py310_diffusion/bin/python"
TORCHRUN="/root/miniconda3/envs/torch280_py310_diffusion/bin/torchrun"
CHECKPOINT="../models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
GEMMA_ROOT="../models/gemma-3-12b-it-qat-q4_0-unquantized"
UPSAMPLER="../models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
PROMPT="A golden retriever running on a beach at sunset"
SEED=42

RESULTS="docs/benchmark_results.md"
mkdir -p docs

echo "# LTX-2 NPU Parallel Benchmark Results" > "$RESULTS"
echo "" >> "$RESULTS"
echo "**Date**: $(date '+%Y-%m-%d %H:%M')" >> "$RESULTS"
echo "**Device**: Ascend 910B3 × 8" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| # | Resolution | Frames | Cards | VAE-P | Stage1 (s) | Stage2 (s) | Total (s) | Status |" >> "$RESULTS"
echo "|---|------------|--------|-------|-------|------------|------------|-----------|--------|" >> "$RESULTS"

run_test() {
    local IDX=$1 NPROC=$2 H=$3 W=$4 FRAMES=$5 RES_NAME=$6
    local TAG="${RES_NAME}_${FRAMES}f_${NPROC}c"
    local PORT=$((30000 + IDX * 100 + RANDOM % 50))
    local OUTPUT="outputs/bench_${TAG}.mp4"
    local VAE_FLAG="" VAE_LABEL="No"

    if [ "$NPROC" -ge 4 ]; then
        VAE_FLAG="--vae-parallel"
        VAE_LABEL="Yes"
    fi

    echo ""
    echo "=== Test ${IDX}: ${RES_NAME} ${H}×${W}, ${FRAMES}f, ${NPROC} cards ==="
    bash scripts/clean_npu.sh 2>&1 | tail -1
    sleep 3

    local T0=$(date +%s)

    if $TORCHRUN --nproc_per_node=$NPROC --master_port=$PORT run_distilled.py \
        --ulysses-degree $NPROC $VAE_FLAG \
        --distilled-checkpoint-path "$CHECKPOINT" \
        --gemma-root "$GEMMA_ROOT" \
        --spatial-upsampler-path "$UPSAMPLER" \
        --prompt "$PROMPT" --output-path "$OUTPUT" --seed $SEED \
        --height $H --width $W --num-frames $FRAMES 2>&1 | tee "/tmp/bench_${TAG}.log"; then

        local T1=$(date +%s)
        local WALL=$((T1 - T0))
        local INFER=$(grep "Total Inference" "/tmp/bench_${TAG}.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        local S1=$(grep "Stage 1 Diffusion" "/tmp/bench_${TAG}.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        local S2=$(grep "Stage 2 Diffusion" "/tmp/bench_${TAG}.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        [ -z "$INFER" ] && INFER="$WALL"
        [ -z "$S1" ] && S1="-"
        [ -z "$S2" ] && S2="-"

        echo "| ${IDX} | ${RES_NAME} (${H}×${W}) | ${FRAMES} | ${NPROC} | ${VAE_LABEL} | ${S1} | ${S2} | ${INFER} | ✅ |" >> "$RESULTS"
        echo "[OK] ${TAG}: total=${INFER}s s1=${S1}s s2=${S2}s (wall: ${WALL}s)"
    else
        echo "| ${IDX} | ${RES_NAME} (${H}×${W}) | ${FRAMES} | ${NPROC} | ${VAE_LABEL} | - | - | - | ❌ |" >> "$RESULTS"
        echo "[FAIL] ${TAG}"
    fi

    sleep 5
}

# 480P (512×832)
run_test 1  2 512 832 65  "480P"
run_test 2  4 512 832 65  "480P"
run_test 3  8 512 832 65  "480P"
run_test 4  2 512 832 121 "480P"
run_test 5  4 512 832 121 "480P"
run_test 6  8 512 832 121 "480P"

# 720P (768×1280)
run_test 7  2 768 1280 65  "720P"
run_test 8  4 768 1280 65  "720P"
run_test 9  8 768 1280 65  "720P"
run_test 10 2 768 1280 121 "720P"
run_test 11 4 768 1280 121 "720P"
run_test 12 8 768 1280 121 "720P"

echo "" >> "$RESULTS"
echo "**Notes**:" >> "$RESULTS"
echo "- VAE-P: VAE Parallel enabled for ≥4 cards" >> "$RESULTS"
echo "- Inference time measured without warmup" >> "$RESULTS"
echo "- 480P = 512×832, 720P = 768×1280" >> "$RESULTS"
echo "- Frames: 65 (≈61), 121" >> "$RESULTS"

echo ""
echo "============================================"
echo " All benchmarks complete!"
echo " Results: $RESULTS"
echo "============================================"
cat "$RESULTS"
