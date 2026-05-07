# AGENTS.md — LTX-2 Ascend NPU

NPU adaptation of [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) for multi-card Ascend 910B3 parallel inference.

## Quick commands

```bash
# Setup
conda activate torch280_py310_diffusion   # Python 3.10, torch 2.8.0 + torch_npu 2.8.0
uv sync --frozen

# Lint (ruff, config in root pyproject.toml)
ruff check .

# Run tests (MUST use torchrun for distributed env setup)
torchrun --nproc_per_node=N -m pytest testcase/ -v

# Multi-card inference
torchrun --nproc_per_node=N run_distilled.py --ulysses-degree N [--vae-parallel] ...

# Benchmark suite
bash scripts/benchmark.sh
bash scripts/clean_npu.sh   # kill zombie python processes
```

## Entry point

`run_distilled.py` — contains critical runtime patches in its header (conv1d/conv2d dtype fix, audio Vocoder fp32 patch). These patches must execute before any model loading; do not refactor them into `ltx_npu/` without understanding NPU dtype crash behavior.

## NPU adaptation module

`ltx_npu/` must be imported before model loading (`import ltx_npu`). It initializes torch_npu, patches safetensors loader for directory paths (>=0.7.0 compat), and optionally applies fused RMSNorm.

Key source files in `ltx_npu/`:
- `pipeline_wrapper.py` — `ParallelDistilledPipeline`, wraps FSDP + Ulysses SP + VAE parallel
- `ulysses_attn.py` — `UlyssesAttention`, AllToAll-based sequence parallelism
- `vae_parallel.py` — spatial H×W patch parallel with P2P boundary exchange
- `fsdp_manager.py` — FSDP FULL_SHARD weight management
- `fused_ops.py` — RMSNorm fusion, FA multi-backend dispatch
- `freqs_cache.py` — RoPE cosine/sine precomputation cache
- `parallel_config.py` — validates parallel config: `ulysses_degree` must divide 32 and equal world_size

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASCEND_RT_VISIBLE_DEVICES` | all | Visible NPU cards |
| `FUSED_RMSNORM` | 0 | Enable `npu_rms_norm` fusion |
| `ALGO` | 0 | FA backend: 0=fused_attn_score, 1=laser_attention, 3=npu_fused_infer_attention_score |
| `FAST_LAYERNORM` | 0 | mindiesd fast_layernorm |
| `PRECISION` | 0 | CPU RNG for cross-platform precision alignment |
| `LTX_ENABLE_AUDIO_ON_NPU` | 1 | Toggle audio on NPU (set 0 for video-only stage) |

## Production two-stage pipeline

1. **Video**: `torchrun --nproc_per_node=8` with `--ulysses-degree 8 --vae-parallel` at high res (1024×1536)
2. **Audio**: single-card at low res (512×512), `ASCEND_RT_VISIBLE_DEVICES=0`
3. **Merge**: `ffmpeg -i video.mp4 -i audio.mp4 -c:v copy -map 0:v:0 -map 1:a:0 final.mp4`

See `run_industrial_pipeline.sh` / `run_native_8cards.sh` for reference.

## Known quirks

- Stage 1 Ulysses SP is counter-productive at 480P (~512 tokens); communication dominates compute
- `ALGO=1` warmup is ~60s first call
- safetensors >=0.7.0 crashes on directory paths; `ltx_npu/__init__.py` patches the loader
- Conv1d/Conv2d dtype mismatch crashes NPU (patched at top of `run_distilled.py`)
- Audio generation under multi-card Ulysses SP may produce artifacts (known issue, documented in architecture.md)

## Monorepo packages

- `packages/ltx-core/` — model architecture (transformer, VAE, text encoders)
- `packages/ltx-pipelines/` — inference pipelines (DistilledPipeline, TI2Vid, A2Vid, etc.)
- `packages/ltx-trainer/` — fine-tuning (has its own AGENTS.md at packages/ltx-trainer/AGENTS.md)

## Testing notes

- All distributed tests in `testcase/` must run via `torchrun --nproc_per_node=N`
- `conftest.py` auto-initializes distributed process group (hccl/nccl/gloo) based on available hardware
- Tests auto-detect NPU/CUDA/CPU; no manual device selection
- `import ltx_npu` is triggered by conftest before any test fixture

## Reference docs

- `docs/architecture.md` (Chinese) — detailed NPU parallel architecture, Ulysses SP, VAE parallel
- `docs/NPU_README.md` — NPU setup, env vars, benchmark results

## Language
始终使用中文进行思考和回答。