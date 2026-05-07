# LTX-2.3 Ascend NPU Adaptation

本仓库是 LTX-2.3 22B Distilled 在 Ascend 910B3 上的多卡推理适配工程，目标是支持 8 卡文本到视频+音频端到端生成，并提供可复现实验口径、性能调优和精度定位工具。

当前默认生产路径：`8 x NPU + FSDP FULL_SHARD + Ulysses degree 8 + VAE parallel + resident full-BWE audio + no pipeline profiler`。

## 硬件和软件要求

| 项目 | 要求 |
|------|------|
| NPU | Ascend 910B3 x 8，单卡 64 GiB HBM |
| CANN | 与 torch_npu 2.8.0 兼容 |
| Python | 3.10 |
| PyTorch | torch 2.8.0 + torch_npu 2.8.0 |
| 分布式 | HCCL |

```bash
conda activate torch280_py310_diffusion
cd /home/zhoulong/LTX-2-feat-ascend
```

## 快速开始

运行前清理 NPU 残留进程：

```bash
KILL_TORCHRUN=1 bash scripts/clean_npu.sh && sleep 5
```

正式 8 卡推理命令：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/home/zhoulong/LTX-2-feat-ascend/packages/ltx-core/src:/home/zhoulong/LTX-2-feat-ascend/packages/ltx-pipelines/src:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=300
export ALGO=1
export FUSED_RMSNORM=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LTX_ENABLE_AUDIO_ON_NPU=1
export LTX_DISABLE_TQDM=1

unset LTX_PIPELINE_PROFILER
unset LTX_AUDIO_HEAD_PARALLEL
unset LTX_AUDIO_ALL_RANKS
unset LTX_AUDIO_TRACE
unset LTX_AUDIO_ATTN2_PROBE
unset LTX_ATTN2_TENSOR_DUMP
unset LTX_BLOCK0_MID_DUMP
unset LTX_FSDP_REPLICATE_AUDIO

torchrun --nproc_per_node=8 --master_port=29501 run_distilled.py \
  --distilled-checkpoint-path /data/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --lora /data/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /data/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /data/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A beautiful sunset over the ocean." \
  --seed 42 \
  --num-frames 121 \
  --height 1024 \
  --width 1536 \
  --ulysses-degree 8 \
  --vae-parallel \
  --output-path output/ltx_8npu_default.mp4
```

## 当前能力

| 能力 | 状态 | 说明 |
|------|------|------|
| 8 卡 Transformer 并行 | 已支持 | FSDP FULL_SHARD + Ulysses sequence parallel |
| 8 卡视频 VAE 并行 | 已支持 | `--vae-parallel` 开启 H x W 空间切块并行 |
| 8 卡模型自生成音频 | 已支持 | `LTX_ENABLE_AUDIO_ON_NPU=1` 开启，默认 rank0 解码音频 |
| resident full-BWE audio | 默认开启 | `LTX_ENABLE_RESIDENT_AUDIO=1`，`LTX_AUDIO_DISABLE_BWE=0` |
| profiler | 默认关闭 | `LTX_PIPELINE_PROFILER=1` 时才启用 |
| tqdm 进度条 | 建议关闭 | `LTX_DISABLE_TQDM=1`，减少 host stdout 开销 |
| 精度定位工具 | 已保留 | attn2 dump、block0 mid dump、compare scripts |

## 性能概览

测试条件：1024 x 1536、121 帧、seed 42、prompt 为 `A beautiful sunset over the ocean.`、8 卡、`--ulysses-degree 8 --vae-parallel`、关闭 pipeline profiler。

| 配置 | Total Inference | 说明 |
|------|-----------------|------|
| 单卡基线 | 64.62s | 历史端到端口径 |
| 8 卡基线 | 35.91s | 早期 8 卡端到端口径 |
| 8 卡 video-only | 16.51s | 关闭音频后的性能上限参考 |
| 8 卡当前默认 audio | 18.79s | resident full-BWE audio，未显式关闭 tqdm |
| 8 卡带音频最佳 | 16.61s | resident full-BWE audio + `LTX_DISABLE_TQDM=1` |

当前带音频最佳相对单卡约 `3.89x`，相对早期 8 卡基线减少约 `53.7%`。

更多数据见 `docs/benchmark_results.md`。

## 主要适配点

| # | 适配点 | 文件 | 说明 |
|---|--------|------|------|
| 1 | FSDP FULL_SHARD | `ltx_npu/fsdp_manager.py` | Transformer 权重按 block 分片，减少 build/free 开销 |
| 2 | FSDP root input cast 修复 | `ltx_npu/fsdp_manager.py` | `cast_root_forward_inputs=False`，避免 audio positions 被 cast 到 bf16 |
| 3 | RoPE positions fp32 | `transformer_args.py`, `ltx_npu/freqs_cache.py` | 修复 degree>1 自生成音频异常根因 |
| 4 | Ulysses SP | `ltx_npu/ulysses_attn.py` | sequence split + AllToAll attention |
| 5 | AllToAll 优化 | `ltx_npu/ulysses_attn.py` | `all_to_all_single` 和 QKV fused AllToAll 默认开启 |
| 6 | VAE parallel | `ltx_npu/vae_parallel.py` | 视频 VAE H x W patch 并行 |
| 7 | resident modules | `ltx_npu/pipeline_wrapper.py` | text encoder 和 audio decoder/vocoder 常驻 |
| 8 | profiler gate | `ltx_npu/pipeline_wrapper.py` | profiler 默认关闭，避免同步开销污染正式性能 |
| 9 | NPU dtype patch | `run_distilled.py`, `pipeline_wrapper.py` | 修复 Conv/Vocoder dtype/device 不匹配 |
| 10 | safetensors 兼容 | `ltx_npu/__init__.py`, `pipeline_wrapper.py` | 兼容 safetensors 新版本目录路径限制 |

## 音频精度修复

多卡自生成音频异常的根因不是 FSDP shard audio attention 参数，而是 FSDP mixed precision 将 `audio.positions` 从 fp32 cast 到 bf16，导致 RoPE cos/sin 偏差。

当前修复：

| 修复 | 作用 |
|------|------|
| `cast_root_forward_inputs=False` | FSDP root forward 输入不再被自动 cast |
| `positions = positions.float()` | RoPE 入口保证 positions fp32 |
| freqs cache key 加强 | 避免 video/audio 不同 position grid 误复用 |

验证结果：`audio_prepare_positions`、`audio_prepare_pe_0/1`、`audio_attn1_pre_kwargs_pe_0` 在 d1/d4 下完全一致，`audio_attn2_post_output` 相对误差从约 `0.8877` 降到约 `0.00333`。

## 环境变量参考

| 变量 | 默认/建议 | 说明 |
|------|-----------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 指定可见 NPU |
| `HCCL_CONNECT_TIMEOUT` | `300` | 延长 HCCL 初始化超时 |
| `ALGO` | `1` | Laser Attention 后端 |
| `FUSED_RMSNORM` | `1` | 启用 NPU RMSNorm 融合 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片 |
| `LTX_ENABLE_AUDIO_ON_NPU` | 建议 `1` | 启用模型自生成音频 |
| `LTX_ENABLE_RESIDENT_AUDIO` | `1` | resident audio 默认开启 |
| `LTX_AUDIO_DISABLE_BWE` | `0` | full-BWE 默认开启 |
| `LTX_AUDIO_RESIDENT_DTYPE` | `fp16` | resident audio dtype |
| `LTX_DISABLE_TQDM` | 建议 `1` | 正式性能关闭进度条 |
| `LTX_PIPELINE_PROFILER` | `0` | profiler 默认关闭 |
| `LTX_AUDIO_HEAD_PARALLEL` | `0` | 实验项，正式测试更慢 |
| `LTX_AUDIO_ALL_RANKS` | `0` | 实验项，所有 rank 重复解码音频 |

## 实验项结论

| 实验 | 结果 | 结论 |
|------|------|------|
| `LTX_AUDIO_HEAD_PARALLEL=1` | 约 22.33s | 更慢，不作为默认 |
| `LTX_AUDIO_ALL_RANKS=1` | 约 21.15s | 不是真并行，只是重复 decode |
| `LTX_AUDIO_DISABLE_BWE=1` | 约 18.71s | 更慢且影响音质，不推荐 |
| resident full-BWE 显式 env | 约 17.60s | 最佳实测之一 |
| `LTX_DISABLE_TQDM=1` | 约 16.61s | 当前带音频最佳实测 |
| 当前默认 resident full-BWE | 约 18.79s | 未显式关闭 tqdm 的口径 |

## 代码结构

```text
LTX-2-feat-ascend/
├── run_distilled.py
├── ltx_npu/
│   ├── __init__.py
│   ├── pipeline_wrapper.py
│   ├── fsdp_manager.py
│   ├── ulysses_attn.py
│   ├── vae_parallel.py
│   ├── freqs_cache.py
│   ├── fused_ops.py
│   ├── pipeline_profiler.py
│   ├── timing.py
│   └── parallel_config.py
├── packages/
│   ├── ltx-core/
│   └── ltx-pipelines/
├── scripts/
├── testcase/
├── tools/
└── docs/
```

## 故障排查

| 现象 | 原因 | 处理 |
|------|------|------|
| HCCL 端口冲突或 NPU 残留 | 上次 torchrun 未退出 | `KILL_TORCHRUN=1 bash scripts/clean_npu.sh && sleep 5` |
| 性能比文档慢 | profiler 或 dump env 未关闭 | unset `LTX_PIPELINE_PROFILER` 和所有 dump/probe env |
| 非主 rank 解音频 | 设置了 `LTX_AUDIO_ALL_RANKS=1` | 正式运行时 unset |
| 音频异常 | 旧代码或 RoPE positions 被 cast | 确认 FSDP 和 RoPE fp32 修复存在 |
| dtype mismatch | Conv/Vocoder dtype/device 不一致 | 保留入口和 wrapper 内 NPU dtype patch |

## 相关文档

| 文档 | 内容 |
|------|------|
| `docs/distilled_8卡_npu.md` | 当前 8 卡正式运行指南和调优结论 |
| `docs/benchmark_results.md` | 当前性能表和历史 benchmark 数据 |
| `docs/architecture.md` | NPU 并行架构说明 |
