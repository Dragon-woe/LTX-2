# LTX-2.3 Ascend NPU Benchmark Results

本文汇总当前 8 卡正式性能口径和早期 benchmark 结果。正式性能默认关闭 `LTX_PIPELINE_PROFILER`，并 unset tensor dump/probe 类调试环境变量。

## 当前正式口径

| 项目 | 值 |
|------|----|
| Device | Ascend 910B3 x 8 |
| Model | LTX-2.3 22B dev checkpoint + distilled LoRA 0.8 |
| Resolution | 1024 x 1536 |
| Frames | 121 |
| Prompt | `A beautiful sunset over the ocean.` |
| Seed | 42 |
| Parallel | `--ulysses-degree 8 --vae-parallel` |
| Audio | `LTX_ENABLE_AUDIO_ON_NPU=1`，默认 resident full-BWE |
| Profiler | `LTX_PIPELINE_PROFILER=0` |
| Progress | 最佳性能设置 `LTX_DISABLE_TQDM=1` |

正式命令参数：

```bash
torchrun --nproc_per_node=8 --master_port=29501 run_distilled.py \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A beautiful sunset over the ocean." \
  --seed 42 --num-frames 121 --height 1024 --width 1536 \
  --ulysses-degree 8 --vae-parallel \
  --output-path output/ltx_8npu_default.mp4
```

## 当前性能汇总

| 配置 | Total Inference | 说明 |
|------|-----------------|------|
| 单卡基线 | 64.62s | 历史端到端口径 |
| 8 卡基线 | 35.91s | 早期 8 卡端到端口径 |
| 8 卡 video-only no-profiler | 16.51s | 关闭音频路径，用于估算视频侧上限 |
| 8 卡 audio rank0-only no-profiler | 22.27s | resident audio 默认化之前 |
| 8 卡 audio all-ranks experiment | 21.15s | 所有 rank 重复 decode，不是真并行 |
| 8 卡 audio head parallel experiment | 22.33s | audio attention head parallel，更慢 |
| 8 卡 resident base-vocoder | 18.71s | `LTX_AUDIO_DISABLE_BWE=1`，更慢且降质量 |
| 8 卡 resident full-BWE 显式 env | 17.60s | 当前最佳实测之一 |
| 8 卡当前默认 resident full-BWE | 18.79s | 未显式关闭 tqdm |
| 8 卡 resident full-BWE + no tqdm | 16.61s | 当前带音频最佳实测，输出含 AAC 48kHz 双声道 |

当前带音频最佳相对单卡约 `3.89x`。当前带音频最佳相对早期 8 卡基线减少约 `19.30s`，约 `53.7%`。

## 当前默认与 video-only 对比

| 配置 | Total Inference | Audio Overhead |
|------|-----------------|----------------|
| 8 卡 video-only | 16.51s | — |
| 8 卡带音频最佳 | 16.61s | +0.10s |
| 8 卡当前默认 audio | 18.79s | +2.28s |
| 8 卡原始 rank0-only audio | 22.27s | +5.76s |

resident full-BWE audio 加上 `LTX_DISABLE_TQDM=1` 后，当前带音频 Total Inference 已接近 video-only 口径。注意 `Total Inference` 不包含最终 `encode_video` 写 MP4 阶段；本次输出经 `ffprobe` 确认包含 H.264 视频和 AAC 48kHz 双声道音频。

## Profiler 口径说明

`PipelineProfiler` 现在默认关闭。开启方式：

```bash
export LTX_PIPELINE_PROFILER=1
```

开启后会安装 hooks 并进行 NPU synchronize，适合模块耗时拆解，但不适合作为正式 Total Inference 口径。

早期 profiler-on audio enabled 结果参考：

| 配置 | Total Inference | Profiler TOTAL | 说明 |
|------|-----------------|----------------|------|
| 8 卡 audio enabled profiler-on | 21.06s | 23.032s | profiler 会引入同步和 hook 开销 |

## 早期 Benchmark 结果

以下数据来自早期批量 benchmark，时间为 2026-04-06 17:27。该表用于历史参考，不代表当前默认 resident full-BWE audio 正式口径。

| # | Resolution | Frames | Cards | VAE-P | Stage1 (s) | Stage2 (s) | Total (s) | Status |
|---|------------|--------|-------|-------|------------|------------|-----------|--------|
| 1 | 480P (512 x 832) | 65 | 2 | No | - | - | 36.11 | pass |
| 2 | 480P (512 x 832) | 65 | 4 | Yes | - | - | 42.11 | pass |
| 3 | 480P (512 x 832) | 65 | 8 | Yes | - | - | 45.46 | pass |
| 4 | 480P (512 x 832) | 121 | 2 | No | - | - | 38.25 | pass |
| 5 | 480P (512 x 832) | 121 | 4 | Yes | - | - | 41.51 | pass |
| 6 | 480P (512 x 832) | 121 | 8 | Yes | - | - | 46.89 | pass |
| 7 | 720P (768 x 1280) | 65 | 2 | No | - | - | 44.28 | pass |
| 8 | 720P (768 x 1280) | 65 | 4 | Yes | - | - | 44.26 | pass |
| 9 | 720P (768 x 1280) | 65 | 8 | Yes | - | - | 46.71 | pass |
| 10 | 720P (768 x 1280) | 121 | 2 | No | - | - | 49.88 | pass |
| 11 | 720P (768 x 1280) | 121 | 4 | Yes | - | - | 46.62 | pass |
| 12 | 720P (768 x 1280) | 121 | 8 | Yes | - | - | 47.85 | pass |

早期 benchmark 注意事项：

| 项目 | 说明 |
|------|------|
| VAE-P | VAE Parallel，早期脚本在 4 卡及以上启用 |
| 计时 | 当时记录为 without warmup，不等同当前正式 no-profiler 口径 |
| 分辨率 | 480P 为 512 x 832，720P 为 768 x 1280 |
| 帧数 | 65 和 121 |

## 复测建议

复测正式性能前执行：

```bash
KILL_TORCHRUN=1 bash scripts/clean_npu.sh && sleep 5
unset LTX_PIPELINE_PROFILER
unset LTX_AUDIO_TRACE
unset LTX_AUDIO_ATTN2_PROBE
unset LTX_ATTN2_TENSOR_DUMP
unset LTX_BLOCK0_MID_DUMP
unset LTX_FSDP_REPLICATE_AUDIO
export LTX_DISABLE_TQDM=1
```

如果复测结果显著慢于 `16.61s`，优先确认是否误开 profiler、tensor dump、`LTX_AUDIO_HEAD_PARALLEL`、`LTX_AUDIO_ALL_RANKS`，以及是否忘记设置 `LTX_DISABLE_TQDM=1`。
