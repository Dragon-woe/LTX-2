#!/bin/bash
set -e

cd /home/zhoulong/LTX-2-feat-ascend

pkill -9 -f run_distilled.py || true
pkill -9 -f torchrun || true
sleep 3

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH
export ALGO=1
export FUSED_RMSNORM=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

unset LTX_ENABLE_RESIDENT_EMBEDDINGS_PROCESSOR
unset LTX_ENABLE_RESIDENT_IMAGE_ENCODER
unset LTX_ENABLE_RESIDENT_UPSAMPLER

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29506

mkdir -p output

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29506 \
  run_distilled.py \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A beautiful sunset over the ocean." \
  --seed 42 \
  --num-frames 17 \
  --height 512 \
  --width 768 \
  --ulysses-degree 8 \
  --no-warmup \
  --output-path output/test_ulysses8.mp4
