# LTX-2 Ascend NPU 多卡并行推理 — 模型结构分析

## 1. 原始 LTX-2 DistilledPipeline 架构

### Pipeline 执行流程（单卡）

```
┌─────────────────────────────────────────────────────────────────┐
│                    DistilledPipeline (Single GPU)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────┐       │
│  │PromptEncoder │    │ Stage 1: DiffusionStage           │       │
│  │ (Gemma 3 12B)│───▶│ 半分辨率 (H/2 × W/2)             │       │
│  └──────────────┘    │ 8 步 distilled denoising          │       │
│                      └───────────────┬──────────────────┘       │
│                                      │                          │
│                      ┌───────────────▼──────────────────┐       │
│                      │ VideoUpsampler (2x Spatial)       │       │
│                      │ latent 空间上采样                  │       │
│                      └───────────────┬──────────────────┘       │
│                                      │                          │
│                      ┌───────────────▼──────────────────┐       │
│                      │ Stage 2: DiffusionStage           │       │
│                      │ 全分辨率 (H × W)                  │       │
│                      │ 4 步 distilled denoising          │       │
│                      └───────────────┬──────────────────┘       │
│                                      │                          │
│                      ┌───────────────▼──────────────────┐       │
│                      │ VideoDecoder (VAE)                │       │
│                      │ latent → pixel video              │       │
│                      └──────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### DiT Transformer 结构（LTXModel）

```
LTXModel (48 layers)
  │
  ├── video_args_preprocessor
  │     ├── patchify_proj: Linear → sequence tokens
  │     ├── adaln: AdaLayerNormSingle → timestep embedding
  │     └── RoPE: 3D positional encoding (frames, height, width)
  │
  ├── transformer_blocks × 48  (BasicAVTransformerBlock)
  │     ├── attn1: Self-Attention     ← Ulysses SP 注入点
  │     │     ├── to_q/to_k/to_v: Linear(4096 → 4096)
  │     │     ├── q_norm/k_norm: RMSNorm
  │     │     ├── apply_rotary_emb (RoPE)
  │     │     ├── attention_function: PytorchAttention/XFormersAttention
  │     │     └── to_out: Linear(4096 → 4096)
  │     │
  │     ├── attn2: Cross-Attention (text conditioning)
  │     ├── ff: FeedForward (4096 → 16384 → 4096)
  │     ├── audio_attn1/audio_attn2/audio_ff (音频路径)
  │     └── audio_to_video_attn / video_to_audio_attn (AV 交叉)
  │
  └── output_head: norm_out → proj_out

参数: num_attention_heads=32, attention_head_dim=128
内部维度: 32 × 128 = 4096
```

### Video VAE 结构

```
VideoDecoder
  │
  ├── per_channel_statistics: 反归一化
  ├── conv_in: Conv3D(128 → 1024)
  │
  ├── up_blocks (反序 encoder_blocks):
  │     ├── UNetMidBlock3D (res_x): 多层 ResnetBlock3D
  │     ├── DepthToSpaceUpsample (compress_all): stride=(2,2,2), ×2 全维度
  │     ├── DepthToSpaceUpsample (compress_time): stride=(2,1,1), ×2 时间
  │     └── DepthToSpaceUpsample (compress_space): stride=(1,2,2), ×2 空间
  │
  ├── conv_norm_out → conv_act → conv_out
  └── unpatchify: channels → spatial (patch_size=4)

压缩率: 时间 8×, 空间 32×
输入: (B, 128, F', H', W')  →  输出: (B, 3, F, H, W)
```

## 2. 适配后多卡 Pipeline 架构

### Pipeline 执行流程（多卡 Ulysses SP + VAE Parallel）

```
┌──────────────────────────────────────────────────────────────────────┐
│              ParallelDistilledPipeline (N-card NPU)                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Rank 0..N-1] ┌──────────────┐                                     │
│  每卡独立运行   │PromptEncoder │  同 prompt → 同 encoding             │
│                 └──────┬───────┘                                     │
│                        │                                             │
│  [Rank 0..N-1] ┌──────▼────────────────────────────────┐            │
│  Ulysses SP    │ Stage 1: DiffusionStage                │            │
│  (AllToAll)    │ attn1.attention_function → Ulysses包装  │            │
│                │ 每卡计算 H/N 个 heads 的 attention      │            │
│                └──────┬────────────────────────────────┘            │
│                       │                                              │
│  [Rank 0 only] ┌─────▼─────────────────────────────────┐           │
│                │ VideoUpsampler (2x) → broadcast latent  │           │
│                └──────┬────────────────────────────────┘            │
│                       │                                              │
│  [Rank 0..N-1] ┌─────▼─────────────────────────────────┐           │
│  Ulysses SP    │ Stage 2: DiffusionStage                │            │
│                │ 同 Stage 1 的 Ulysses 注入              │            │
│                └──────┬────────────────────────────────┘            │
│                       │                                              │
│  [Rank 0..N-1] ┌─────▼─────────────────────────────────┐           │
│  VAE Parallel  │ VideoDecoder (VAE)                      │           │
│  (可选)        │ 空间 H×W 切分到各卡                     │           │
│                │ P2P 边界交换 + dispatch 拼接             │           │
│                └──────┬────────────────────────────────┘            │
│                       │                                              │
│  [Rank 0 only] ┌─────▼────────────┐                                │
│                │ 保存视频 (MP4)     │                                │
│                └──────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. DiT Attention Ulysses 切分示意

### AllToAll 通信模式

```
每个 Rank 持有: (B, T, H, D) — 全序列, 全 heads

═══════ 第一次 AllToAll: scatter head(dim=2), gather seq(dim=1) ═══════

Rank 0 输入: (B, T, H, D)     Rank 1 输入: (B, T, H, D)
    H 个 heads, T 个 tokens        H 个 heads, T 个 tokens

                    ↓ AllToAll ↓

Rank 0 输出: (B, T, H/2, D)   Rank 1 输出: (B, T, H/2, D)
    H/2 个 heads, T 个 tokens      H/2 个 heads, T 个 tokens

═══════ SDPA Attention (本地计算) ═══════

每个 Rank 对 H/N 个 heads 做完整 T×T 的 attention
计算量减少 N 倍 (每卡只算 H/N 个 heads)

═══════ 第二次 AllToAll: scatter seq(dim=1), gather head(dim=2) ═══════

Rank 0: (B, T, H/2, D)  →  (B, T, H, D)  恢复全 heads
Rank 1: (B, T, H/2, D)  →  (B, T, H, D)  恢复全 heads
```

### 注入方式

```python
# 遍历所有 transformer blocks, 只替换 self-attention (attn1)
for block in model.transformer_blocks:
    original_fn = block.attn1.attention_function  # PytorchAttention
    block.attn1.attention_function = UlyssesAttention(original_fn, sp_group)
    # block.attn2 (cross-attention) 保持不变
```

## 4. VAE Spatial Patch 切分示意

### 空间网格分配

```
2 卡 (h_split=2, w_split=1):      4 卡 (h_split=2, w_split=2):
┌──────────────┐                   ┌───────┬───────┐
│   Rank 0     │                   │ Rank 0│ Rank 1│
│   H/2 × W    │                   │ H/2×  │ H/2×  │
├──────────────┤                   │ W/2   │ W/2   │
│   Rank 1     │                   ├───────┼───────┤
│   H/2 × W    │                   │ Rank 2│ Rank 3│
└──────────────┘                   │ H/2×  │ H/2×  │
                                   │ W/2   │ W/2   │
8 卡 (h_split=4, w_split=2):      └───────┴───────┘
┌───────┬───────┐
│ Rank 0│ Rank 1│
├───────┼───────┤
│ Rank 2│ Rank 3│
├───────┼───────┤
│ Rank 4│ Rank 5│
├───────┼───────┤
│ Rank 6│ Rank 7│
└───────┴───────┘
```

### P2P 边界交换

```
conv3d padding=1 时, 需要邻居的 1 行/列:

   ┌───────────────┐
   │   Rank 上方    │
   │ ▄▄▄▄▄▄▄▄▄▄▄▄ │ ← 发送底部 1 行给 Rank 下方
   └───────┬───────┘
           │ recv top_pad
   ┌───────▼───────┐
   │ top_pad (1行) │
   │───────────────│
   │  Rank 当前    │ ← 本地 patch
   │───────────────│
   │ bottom_pad    │
   └───────┬───────┘
           │ recv from below
   ┌───────▼───────┐
   │   Rank 下方    │
   │ ▀▀▀▀▀▀▀▀▀▀▀▀ │ ← 发送顶部 1 行给 Rank 上方
   └───────────────┘

替换后的 conv3d 调用:
  padding=(temporal_pad, 0, 0)  — H/W 方向不再用零填充
  边界数据来自邻居的真实像素
```

## 5. 端到端序列并行 vs Head-only Ulysses 对比

### 方案对比

| 维度 | Head-only Ulysses (v1) | 端到端序列并行 (v2) |
|------|----------------------|-------------------|
| **序列切分** | ❌ 不切分，每 rank 处理全 T 序列 | ✅ 切分到 T/N，每 rank 只处理本地 tokens |
| **LayerNorm/FFN** | 全量 T（无加速） | T/N（N 倍加速） |
| **QKV 线性投影** | 全量 T（无加速） | T/N（N 倍加速） |
| **Self-Attention** | AllToAll 交换 head → SDPA on T | AllToAll gather seq → SDPA on T → AllToAll scatter |
| **Cross-Attention** | 全量 T（无加速） | query=T/N, KV=完整 context（N 倍加速） |
| **通信开销** | 2× AllToAll/层 × 48层 = 96 次 | 同上（但有序列切分带来的计算节省抵消） |
| **净效果** | 通信 > 节省 → 卡数越多越慢 | 计算 N× 缩减 → 真正加速 |

### 关键 Bug 修复

**SPLIT RoPE 4D 张量切分错误**：LTX-2.3 使用 `LTXRopeType.SPLIT`，RoPE 张量为 4D `(B, H, T, D)`。原始 `_split_rope` 在 dim=1（Head 维）切分而非 dim=2（Sequence 维），导致：
- Head 被错误切分 32→16
- `apply_split_rotary_emb` 用错误的 h/t 做 reshape
- 输出从 `(1, 960, 4096)` 变形为 `(1, 1920, 2048)` — 语义完全错乱

修复：`_split_rope` 检测 4D 张量时在 dim=2 切分。

## 6. Benchmark 结果

> 以下为占位表格，实际数据在运行 `bash scripts/benchmark.sh` 后填入。

### 各阶段耗时对比 (秒)

| 阶段 | 1 卡 | 2 卡 | 4 卡 | 8 卡 |
|------|------|------|------|------|
| Prompt Encoding | - | - | - | - |
| Stage 1 Diffusion | - | - | - | - |
| Upsampling | - | - | - | - |
| Stage 2 Diffusion | - | - | - | - |
| VAE Decode | - | - | - | - |
| **Total** | **-** | **-** | **-** | **-** |

### FID 对比

| 对比 | FID 值 | 状态 |
|------|--------|------|
| 1 卡 vs 2 卡 | - | - |
| 1 卡 vs 4 卡 | - | - |
| 1 卡 vs 8 卡 | - | - |
