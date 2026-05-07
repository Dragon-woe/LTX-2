#!/bin/bash
set -e

PROMPT="A futuristic cyberpunk city with flying cars at night, neon lights reflecting on wet streets."
SEED=42
FRAMES=121

# 清理环境
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 2

echo "=========================================================="
echo "🎬 第一阶段：【画面生产】8卡火力全开 (预计耗时: ~15秒)"
echo "=========================================================="
# 关闭音频解码，专注画面
export LTX_ENABLE_AUDIO_ON_NPU=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=30188 run_distilled.py \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "$PROMPT" \
  --seed $SEED \
  --num-frames $FRAMES \
  --height 1024 \
  --width 1536 \
  --ulysses-degree 8 \
  --vae-parallel \
  --output-path ./output/step1_video_only.mp4

echo "=========================================================="
echo "🎵 第二阶段：【音频生产】单卡极速精算 (预计耗时: ~10秒)"
echo "=========================================================="
# 开启音频解码，强锁单卡
export LTX_ENABLE_AUDIO_ON_NPU=1
export ASCEND_RT_VISIBLE_DEVICES=0

# 注意：分辨率降至 512x512，大幅缩短 DiT 耗时，只为提取纯净音频
python run_distilled.py \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "$PROMPT" \
  --seed $SEED \
  --num-frames $FRAMES \
  --height 512 \
  --width 512 \
  --output-path ./output/step2_audio_only.mp4

echo "=========================================================="
echo "🎞️ 第三阶段：【后期封装】音画无损合成 (预计耗时: ~1秒)"
echo "=========================================================="
# 提取第一阶段的视频轨(v) 和 第二阶段的音频轨(a)，合并为最终文件
ffmpeg -y -i ./output/step1_video_only.mp4 -i ./output/step2_audio_only.mp4 \
  -c:v copy -map 0:v:0 -map 1:a:0 ./output/final_masterpiece.mp4 -loglevel error

echo "🎉 恭喜！终极成品已保存至: ./output/final_masterpiece.mp4"
