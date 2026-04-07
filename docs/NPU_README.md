# LTX-2 Ascend NPU Adaptation

LTX-2.3 22B Distilled 视频生成模型在华为 Ascend 910B3 NPU 上的多卡并行推理适配。

## 硬件要求

- Ascend 910B3 × 2/4/8 卡（每卡 64 GiB HBM）
- CANN 版本: 与 torch_npu 2.8.0 兼容
- Python 3.10

## 依赖安装

```bash
conda activate torch280_py310_diffusion

# 核心依赖
pip install torch==2.8.0 torch_npu==2.8.0

# 可选加速库
pip install mindiesd  # FA 多后端 + fast_layernorm
```

## 快速开始

### 多卡推理（推荐）

```bash
# 4 卡 FSDP + Ulysses SP + VAE 并行
torchrun --nproc_per_node=4 run_distilled.py \
  --distilled-checkpoint-path /path/to/LTX-2.3 \
  --gemma-root /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upsampler-path /path/to/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --prompt "A serene mountain landscape with a flowing river at sunset" \
  --seed 42 --num-frames 65 \
  --ulysses-degree 4 --vae-parallel \
  --output-path output.mp4

# 8 卡
torchrun --nproc_per_node=8 run_distilled.py \
  ... --ulysses-degree 8 --vae-parallel ...
```

### 指定 NPU 卡号

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 run_distilled.py ...
```

## NPU 适配优化点

### 已实现的优化

| # | 优化 | 环境变量 | 默认 | 说明 |
|---|------|---------|------|------|
| 1 | **FSDP FULL_SHARD** | — | 自动 | Transformer 权重分片，消除 build/free 循环 |
| 2 | **Gemma 常驻显存** | — | 自动 | Text Encoder 保持在 NPU，消除每次加载 |
| 3 | **E2E 序列并行** | `--ulysses-degree N` | 1 | 每 rank 处理 T/N tokens，AllToAll 恢复全序列 attention |
| 4 | **VAE 空间并行** | `--vae-parallel` | 关 | H×W 空间切分 + P2P 边界交换 |
| 5 | **RoPE Freqs 缓存** | — | 自动 | 采样循环内复用预计算 cos/sin，消除 12 次/generate 冗余计算 |
| 6 | **RMSNorm 融合** | `FUSED_RMSNORM=1` | 关 | `torch.nn.RMSNorm` → `torch_npu.npu_rms_norm` 单 kernel |
| 7 | **FA 多后端调度** | `ALGO=0/1/3` | 0 | mindiesd attention_forward 多后端选优 |
| 8 | **safetensors 兼容** | — | 自动 | 修复 safetensors ≥0.7.0 目录路径不兼容问题 |

### FA 后端选择 (ALGO)

| ALGO | 后端 | 适用场景 |
|------|------|---------|
| 0 | `fused_attn_score` (默认) | 通用，兼容性最好 |
| 1 | `ascend_laser_attention` | Self-attention，BSND 布局 |
| 3 | `npu_fused_infer_attention_score` | 量化推理场景 |

Cross-attention 始终使用 ALGO=0 回退。

### 使用示例

```bash
# 启用所有优化
FUSED_RMSNORM=1 ALGO=1 torchrun --nproc_per_node=4 run_distilled.py \
  --distilled-checkpoint-path /path/to/LTX-2.3 \
  --gemma-root /path/to/gemma \
  --spatial-upsampler-path /path/to/upscaler-x2.safetensors \
  --prompt "Your prompt" --seed 42 --num-frames 65 \
  --ulysses-degree 4 --vae-parallel \
  --output-path output.mp4
```

## 性能测试结果

### 测试条件

- 模型: LTX-2.3 22B Distilled (两阶段: Stage 1 半分辨率 8 步 + Stage 2 全分辨率 3 步)
- 精度: bfloat16
- 计时: warmup 后单次 generate 端到端耗时

### 003 阶段: FSDP + Gemma 常驻 (8 卡, 480P/65f)

| 配置 | generate (s) | vs 1 卡 | 说明 |
|------|-------------|---------|------|
| 1 卡 baseline | 38.67 | — | 含 build-free 循环 |
| 2 卡 E2E SP (无 FSDP) | 40.29 | +4.2% ❌ | build-free 抵消 SP 收益 |
| 2 卡 FSDP | 26.45 | -31.6% ✅ | 零 build-free |
| 8 卡 FSDP | 26.08 | -32.6% ✅ | Stage 1 SP 反效果限制扩展 |
| **8 卡 FSDP + Gemma 常驻** | **17.57** | **-54.6% ✅** | 最优配置 |

### 004 阶段: 融合算子 (4 卡, 默认分辨率/65f)

| 配置 | Total Inference (s) | vs Baseline | 说明 |
|------|-------------------|-------------|------|
| Baseline (所有 OFF) | 23.13 | — | RoPE 缓存已自动启用 |
| `FUSED_RMSNORM=1` | 22.65 | -2.1% | RMSNorm 单 kernel 融合 |
| `ALGO=1` (laser_attn) | 22.84 | -1.3% | ascend_laser_attention |
| `FUSED_RMSNORM=1 + ALGO=1` | 22.94 | -0.8% | 全量组合 |

## 代码结构

```
LTX-2/
├── run_distilled.py            # NPU 多卡推理入口
├── ltx_npu/                    # NPU 适配模块
│   ├── __init__.py             # NPU 运行时初始化 + 全局 patch
│   ├── fused_ops.py            # 融合算子: RMSNorm/FA/LayerNorm/精度
│   ├── freqs_cache.py          # RoPE 频率缓存
│   ├── pipeline_wrapper.py     # DistilledPipeline 多卡包装
│   ├── ulysses_attn.py         # Ulysses 序列并行 attention
│   ├── fsdp_manager.py         # FSDP 分片管理
│   ├── vae_parallel.py         # VAE 空间切分并行
│   ├── pipeline_profiler.py    # 性能打点工具
│   ├── device_context.py       # NPU/CUDA/CPU 设备抽象
│   ├── parallel_config.py      # 并行配置
│   ├── timing.py               # 计时报告
│   └── npu_utils.py            # NPU 工具函数
├── testcase/                   # 单元测试
├── scripts/                    # 性能测试脚本
└── docs/                       # 文档
    ├── NPU_README.md           # 本文件
    ├── architecture.md         # 架构说明
    └── benchmark_results.md    # 早期 benchmark 数据
```

## 已知限制

1. **Stage 1 SP 反效果**: 480P 半分辨率序列仅 ~512 tokens，4-way SP 下 AllToAll 通信 > 计算收益
2. **AdaLN 融合不可用**: LTX-2 的 AdaLN 使用 `rms_norm`，mindiesd `layernorm_scale_shift` 仅支持 LayerNorm
3. **ALGO=1 warmup 开销大**: `ascend_laser_attention` 首次编译 ~60s，后续稳定
4. **小模型 build/free**: VideoEncoder、Upsampler、Decoder 仍有 ~3.6s/generate 的框架开销

## 环境变量参考

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | 全部 | 指定可见 NPU 卡号 |
| `FUSED_RMSNORM` | `0` | 启用 npu_rms_norm 融合 |
| `ALGO` | `0` | FA 后端: 0/1/3 |
| `FAST_LAYERNORM` | `0` | 启用 mindiesd fast_layernorm |
| `PRECISION` | `0` | CPU 随机数生成（跨平台精度对齐） |
