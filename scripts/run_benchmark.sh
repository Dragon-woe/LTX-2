#!/bin/bash
# Run a single benchmark test case.
# Usage: bash scripts/run_benchmark.sh <NPROC> <HEIGHT> <WIDTH> <FRAMES> <TAG>

set -e
cd "$(dirname "$0")/.."

NPROC=$1
HEIGHT=$2
WIDTH=$3
FRAMES=$4
TAG=$5

PYTHON="/root/miniconda3/envs/torch280_py310_diffusion/bin/python"
TORCHRUN="/root/miniconda3/envs/torch280_py310_diffusion/bin/torchrun"
CHECKPOINT="../models/LTX-2.3/ltx-2.3-22b-distilled.safetensors"
GEMMA_ROOT="../models/gemma-3-12b-it-qat-q4_0-unquantized"
UPSAMPLER="../models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
PROMPT="A golden retriever running on a beach at sunset"
SEED=42

PORT=$((29500 + RANDOM % 1000))
OUTPUT="outputs/bench_${TAG}.mp4"

VAE_FLAG=""
if [ "$NPROC" -ge 4 ]; then
    VAE_FLAG="--vae-parallel"
fi

bash scripts/clean_npu.sh 2>&1 | tail -1
sleep 2

echo "[${TAG}] Starting: ${NPROC} cards, ${HEIGHT}x${WIDTH}, ${FRAMES} frames, port=${PORT}"

$TORCHRUN --nproc_per_node=$NPROC --master_port=$PORT run_distilled.py \
    --ulysses-degree $NPROC $VAE_FLAG \
    --distilled-checkpoint-path "$CHECKPOINT" \
    --gemma-root "$GEMMA_ROOT" \
    --spatial-upsampler-path "$UPSAMPLER" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT" \
    --seed $SEED \
    --height $HEIGHT --width $WIDTH \
    --num-frames $FRAMES \
    2>&1

echo "[${TAG}] Done."
