# LTX-2.3 Distilled 8 卡 Ascend NPU 推理指南

本文记录当前仓库在 8 x Ascend 910B3 上运行 LTX-2.3 22B Distilled 文本到视频+音频推理的实际状态、默认配置、性能结果和已知调试结论。

当前推荐路径是 8 卡直接生成视频和模型自生成音频：`Ulysses degree 8 + FSDP + VAE parallel + resident full-BWE audio`。

## 环境要求

| 项目 | 要求 |
|------|------|
| NPU | 8 x Ascend 910B3，单卡 64 GiB HBM |
| Python | 3.10 |
| PyTorch | torch 2.8.0 + torch_npu 2.8.0 |
| 分布式后端 | HCCL |
| 模型精度 | bfloat16 为主，audio resident 默认 fp16 |

建议先进入项目和 Python 环境：

```bash
cd /home/zhoulong/LTX-2-feat-ascend
conda activate torch280_py310_diffusion
```

## 快速运行

运行前先清理本项目残留的 torchrun/python 进程，避免 HCCL 端口或 NPU 显存残留影响结果：

```bash
KILL_TORCHRUN=1 bash scripts/clean_npu.sh && sleep 5
```

推荐的正式 8 卡命令如下。默认会启用 rank0 resident full-BWE audio，非主 rank 使用 silent stub，pipeline profiler 默认关闭。

```bash
cd /home/zhoulong/LTX-2-feat-ascend

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

## 当前默认行为

| 功能 | 当前默认 | 说明 |
|------|----------|------|
| Transformer 并行 | FSDP FULL_SHARD + Ulysses SP | `--ulysses-degree 8` 时按 8 卡切 video sequence，并对 Transformer block 做 FSDP 分片 |
| VAE 视频解码 | 开启需显式传参 | 使用 `--vae-parallel` 开启 H x W 空间切块并行 |
| 模型自生成音频 | 开启需显式 env | `LTX_ENABLE_AUDIO_ON_NPU=1` 时启用 |
| resident audio | 默认开启 | `LTX_ENABLE_RESIDENT_AUDIO` 默认 `1` |
| full-BWE audio | 默认开启 | `LTX_AUDIO_DISABLE_BWE` 默认 `0` |
| audio resident dtype | 默认 fp16 | 可用 `LTX_AUDIO_RESIDENT_DTYPE=fp16|bf16|fp32` 调整 |
| pipeline profiler | 默认关闭 | 只有 `LTX_PIPELINE_PROFILER=1` 才安装 hooks 和打印报告 |
| tqdm 进度条 | 建议关闭 | `LTX_DISABLE_TQDM=1` 可减少多进程 stdout/host overhead |
| audio head parallel | 默认关闭 | `LTX_AUDIO_HEAD_PARALLEL=1` 是实验项，正式测试更慢 |
| all-ranks audio | 默认关闭 | `LTX_AUDIO_ALL_RANKS=1` 是实验项，会让所有 rank 重复解码音频 |

## 性能结果

测试口径：1024 x 1536、121 帧、prompt 为 `A beautiful sunset over the ocean.`、seed 42、8 卡、`--ulysses-degree 8 --vae-parallel`、默认关闭 `LTX_PIPELINE_PROFILER`。

| 配置 | Total Inference | 说明 |
|------|-----------------|------|
| 单卡基线 | 64.62s | 历史单卡端到端口径 |
| 8 卡基线 | 35.91s | 早期 8 卡端到端口径 |
| 8 卡 video-only no-profiler | 16.51s | 关闭音频路径，用于估算视频侧上限 |
| 8 卡 audio rank0-only no-profiler | 22.27s | resident audio 默认化之前的音频路径 |
| 8 卡 resident full-BWE audio 显式 env | 17.60s | 当前最佳实测之一 |
| 8 卡当前默认 resident full-BWE audio | 18.79s | 未显式关闭 tqdm 的生产口径 |
| 8 卡 resident full-BWE audio + no tqdm | 16.61s | 当前带音频最佳实测，输出含 AAC 48kHz 双声道 |

当前带音频最佳实测相对单卡约 `3.89x`，相对 8 卡基线减少约 `19.30s`。video-only 相对单卡约 `3.91x`。

`LTX_PIPELINE_PROFILER=1` 会引入同步和 hook 开销，不应用于正式性能口径。需要拆解模块耗时时再开启。

## 关键适配点

| # | 适配/优化 | 文件 | 说明 |
|---|-----------|------|------|
| 1 | FSDP FULL_SHARD | `ltx_npu/fsdp_manager.py` | Transformer 22B 参数按 block 分片，避免每次推理 build/free 全量模型 |
| 2 | 禁用 FSDP root input cast | `ltx_npu/fsdp_manager.py` | `cast_root_forward_inputs=False`，避免 audio positions 被 cast 到 bf16 |
| 3 | RoPE positions 强制 fp32 | `transformer_args.py`, `ltx_npu/freqs_cache.py` | 修复多卡自生成音频异常根因 |
| 4 | Ulysses 序列并行 | `ltx_npu/ulysses_attn.py` | video sequence 分片，attention 内部通过 AllToAll 恢复计算语义 |
| 5 | AllToAll 优化 | `ltx_npu/ulysses_attn.py` | `all_to_all_single` 和 QKV fused AllToAll 默认开启 |
| 6 | VAE 空间并行 | `ltx_npu/vae_parallel.py` | 视频 VAE 按 H x W patch 并行，降低高分辨率 decode 时间 |
| 7 | resident text encoder | `ltx_npu/pipeline_wrapper.py` | `LTX_ENABLE_RESIDENT_TEXT_ENCODER` 默认 `1` |
| 8 | resident audio | `ltx_npu/pipeline_wrapper.py` | rank0 常驻 audio decoder/vocoder，默认 full-BWE |
| 9 | profiler gate | `ltx_npu/pipeline_wrapper.py` | `LTX_PIPELINE_PROFILER=1` 时才安装 profiler hooks |
| 10 | NPU dtype patch | `run_distilled.py`, `pipeline_wrapper.py` | 修复 Conv/Vocoder 输入 dtype/device 不匹配 |
| 11 | safetensors 目录兼容 | `ltx_npu/__init__.py`, `pipeline_wrapper.py` | 兼容 safetensors 新版本对目录路径的限制 |

## 音频异常根因与修复

之前 8 卡 degree>1 自生成音频出现异常、停顿或动态不足，最初怀疑是 FSDP 分片 audio attention 参数导致。后续通过 exact tensor dump 反证：复制 `block0.audio_attn2` 后输出仍与普通 d4 完全一致，说明 shard 不是根因。

最终定位到 FSDP mixed precision 的 root forward input cast：

```python
MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_root_forward_inputs=False,
)
```

当 `cast_root_forward_inputs=True` 时，`audio.positions` 会从 fp32 被 cast 到 bf16，导致 RoPE cos/sin 出现偏差。偏差首先出现在 `audio_attn1_pre_kwargs_pe_0/1`，随后影响 `audio_attn1` 输出，并在 `audio_attn2` 的 audio hidden/Q 分支被放大。

当前修复包括：

| 修复 | 位置 | 目的 |
|------|------|------|
| `cast_root_forward_inputs=False` | `ltx_npu/fsdp_manager.py` | 保持 root forward 输入原始 dtype |
| `positions = positions.float()` | `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py` | RoPE 入口强制 fp32 positions |
| freqs cache key 更严格 | `ltx_npu/freqs_cache.py` | 避免 video/audio 或不同 position grid 误复用缓存 |

修复后验证结果：

| Tensor | d1 vs d4 |
|--------|----------|
| `audio_prepare_positions` | equal `True`, rel `0.0` |
| `audio_prepare_pe_0/1` | equal `True`, rel `0.0` |
| `audio_attn1_pre_args_0` | equal `True`, rel `0.0` |
| `audio_attn1_pre_kwargs_pe_0` | equal `True`, rel `0.0` |
| `audio_attn2_post_output` | rel 从约 `0.8877` 降到约 `0.00333` |

## 环境变量

### 正式推理建议

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 指定 8 张 NPU |
| `HCCL_CONNECT_TIMEOUT` | `300` | 减少启动慢导致的 HCCL 超时 |
| `ALGO` | `1` | 使用 Laser Attention 后端 |
| `FUSED_RMSNORM` | `1` | 启用 NPU RMSNorm 融合 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片 |
| `LTX_ENABLE_AUDIO_ON_NPU` | `1` | 启用 NPU 音频生成 |
| `LTX_DISABLE_TQDM` | `1` | 正式性能建议关闭进度条，减少 host stdout 开销 |

### 当前默认可覆盖项

| 变量 | 默认 | 可选值 | 说明 |
|------|------|--------|------|
| `LTX_ENABLE_RESIDENT_AUDIO` | `1` | `0/1` | 是否常驻 audio decoder/vocoder |
| `LTX_AUDIO_DISABLE_BWE` | `0` | `0/1` | `0` 为 full-BWE，`1` 为 base-vocoder 实验路径 |
| `LTX_AUDIO_RESIDENT_DTYPE` | `fp16` | `fp16/bf16/fp32` | resident audio 模块 dtype |
| `LTX_PIPELINE_PROFILER` | `0` | `0/1` | 是否启用 pipeline profiler hooks |
| `LTX_DISABLE_TQDM` | `0` | `0/1` | `1` 完全关闭 tqdm；默认只在 rank0 显示 |
| `LTX_AUDIO_HEAD_PARALLEL` | `0` | `0/1` | audio attention head parallel 实验项 |
| `LTX_AUDIO_ALL_RANKS` | `0` | `0/1` | 所有 rank 重复 audio decode 实验项 |
| `LTX_UA2A_USE_A2A_SINGLE` | `1` | `0/1` | Ulysses AllToAll fast path |
| `LTX_UA2A_FUSE_QKV` | `1` | `0/1` | QKV fused AllToAll |

### 调试开关

| 变量 | 用途 |
|------|------|
| `LTX_AUDIO_TRACE` | audio trace 调试 |
| `LTX_AUDIO_ATTN2_PROBE` | audio attn2 probe |
| `LTX_ATTN2_TENSOR_DUMP` | exact attn2 tensor dump |
| `LTX_BLOCK0_MID_DUMP` | block0 中间 tensor dump |
| `LTX_FSDP_REPLICATE_AUDIO` | 复制部分 audio 模块做反证实验 |

正式跑性能前应 unset 上述调试开关。

## 实验项结论

| 实验 | 结果 | 结论 |
|------|------|------|
| `LTX_AUDIO_HEAD_PARALLEL=1` | 8 卡正式约 `22.33s` | 比默认慢，不推荐作为默认 |
| `LTX_AUDIO_ALL_RANKS=1` | 约 `21.15s` | 只是所有 rank 重复 decode，最终仍只用 rank0 audio，不是真并行 |
| `LTX_AUDIO_DISABLE_BWE=1` | resident base-vocoder 约 `18.71s` | 更慢且降低音质，不推荐 |
| `LTX_DISABLE_TQDM=1` | 约 `16.61s` | 当前带音频最佳实测 |
| 复制 FSDP audio_attn2 | d4 与 d4+repl 一致 | 反证 audio_attn2 shard 不是根因 |
| video-only no-profiler | 约 `16.51s` | 当前视频侧性能上限参考 |

后续如果继续优化音频，方向应是 time-chunk + overlap/gather 或异步 audio decode，而不是简单照搬视频 VAE spatial parallel，也不是让所有 rank 重复解码。

## 参数说明

| 参数 | 说明 |
|------|------|
| `--distilled-checkpoint-path` | 主模型 safetensors 路径 |
| `--lora` | LoRA 路径和强度，例如 distilled LoRA `0.8` |
| `--spatial-upsampler-path` | x2 空间上采样器路径 |
| `--gemma-root` | Gemma 3 12B 文本编码器目录 |
| `--prompt` | 文本提示词 |
| `--seed` | 随机种子 |
| `--num-frames` | 输出帧数，当前正式验证使用 `121` |
| `--height`, `--width` | 输出分辨率，当前正式验证使用 `1024 x 1536` |
| `--ulysses-degree` | Ulysses 序列并行度，8 卡正式使用 `8` |
| `--vae-parallel` | 启用视频 VAE 空间并行 |
| `--audio-path` | 可选外部音频路径；当前默认不需要 |
| `--output-path` | 输出 MP4 路径 |

## 调试工具

### Pipeline Profiler

默认关闭 profiler。需要模块级耗时拆解时显式开启：

```bash
export LTX_PIPELINE_PROFILER=1
torchrun --nproc_per_node=8 --master_port=29501 run_distilled.py ...
```

注意 profiler 会同步 NPU 并安装 hooks，因此 Total Inference 不能直接和 no-profiler 正式口径比较。

### Tensor Dump

`ltx_npu/audio_attn2_tensor_dump.py` 和 `ltx_npu/block0_mid_tensor_dump.py` 用于 d1/d4 精度定位。配套比较脚本：

```bash
python tools/compare_attn2_tensors.py <d1_dump_dir> <d4_dump_dir>
python tools/compare_block0_mid_tensors.py <d1_dump_dir> <d4_dump_dir>
```

这些工具通过环境变量触发，会产生大量 `.pt` dump 文件；正式运行前应清理 dump 输出。

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
│   ├── clean_npu.sh
│   ├── benchmark.sh
│   ├── run_benchmark.sh
│   ├── run_all_benchmarks.sh
│   ├── fid_compare.py
│   └── test_dit_only_timing.py
├── testcase/
├── tools/
└── docs/
```

## 故障排查

| 现象 | 常见原因 | 处理方式 |
|------|----------|----------|
| HCCL 初始化失败或端口冲突 | 上次 torchrun 未完全退出 | `KILL_TORCHRUN=1 bash scripts/clean_npu.sh && sleep 5`，必要时换 `--master_port` |
| 正式性能明显偏慢 | profiler 或 dump/probe env 未关闭 | `unset LTX_PIPELINE_PROFILER` 并 unset 所有 dump/probe env |
| 非主 rank 也在解音频 | 设置了 `LTX_AUDIO_ALL_RANKS=1` | 正式运行前 unset 或设为 `0` |
| 音频质量或动态异常 | RoPE positions 被 cast 或旧代码未更新 | 确认 `cast_root_forward_inputs=False`，并确认 positions fp32 修复存在 |
| NPU dtype mismatch | Conv/Vocoder 输入 dtype/device 不一致 | 保留 `run_distilled.py` 和 `pipeline_wrapper.py` 中的 NPU dtype patch |
| 输出目录被忽略 | `.gitignore` 已忽略 `output/`, `logs/`, dump 等 | 这是预期行为 |
