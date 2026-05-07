#!/bin/bash
set -e

# 清除僵尸进程
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 2


LAUNCHER_PATH=$(find . -name "ltx_ulysses_hccl_launcher_distilled_8npu.py" | head -n 1)

if [ -z "$LAUNCHER_PATH" ]; then
    echo "❌ 找不到 ltx_ulysses_hccl_launcher_distilled_8npu.py，请确认代码是否完整！"
    exit 1
fi

echo "✅ 成功找到华为原装 8卡 启动器: $LAUNCHER_PATH"
echo "=========================================================="
echo "🎬 第一阶段：【画面生产】原生 8卡 Ulysses 并行极速出图"
echo "=========================================================="

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 关闭音频在 NPU 上的解码，专注画面 (规避切分崩溃)
export LTX_ENABLE_AUDIO_ON_NPU=1

# 使用 torchrun 启动真正的 launcher
torchrun --nproc_per_node=8 --master_port=30222 $LAUNCHER_PATH \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A futuristic cyberpunk city with flying cars at night, neon lights reflecting on wet streets." \
  --seed 42 \
  --num-frames 121 \
  --height 1024 \
  --width 1536 \
  --ulysses-degree 8 \
  --vae-parallel \
  --output-path ./output/step1_video_only.mp4

echo "=========================================================="
echo "🎵 第二阶段：【音频生产】单卡原画音质补偿"
echo "=========================================================="
export ASCEND_RT_VISIBLE_DEVICES=0
export LTX_ENABLE_AUDIO_ON_NPU=1

# 降分辨率，单卡极速跑声音
python $LAUNCHER_PATH \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A futuristic cyberpunk city with flying cars at night, neon lights reflecting on wet streets." \
  --seed 42 \
  --num-frames 121 \
  --height 512 \
  --width 512 \
  --output-path ./output/step2_audio_only.mp4

echo "=========================================================="
echo "🎞️ 第三阶段：【无损封装】"
echo "=========================================================="
ffmpeg -y -i ./output/step1_video_only.mp4 -i ./output/step2_audio_only.mp4 \
  -c:v copy -map 0:v:0 -map 1:a:0 ./output/final_masterpiece.mp4 -loglevel error

echo "🎉 终极成品已保存至: ./output/final_masterpiece.mp4"
