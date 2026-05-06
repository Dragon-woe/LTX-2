"""Multi-card wrapper for DistilledPipeline with Ulysses SP and VAE Parallel."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ltx_npu.device_context import DeviceContext
from ltx_npu.parallel_config import ParallelConfig
from ltx_npu.pipeline_profiler import PipelineProfiler, install_hooks
from ltx_npu.timing import TimingReport
from ltx_npu.freqs_cache import clear_freqs_cache_on_model, install_freqs_cache_on_model
from ltx_npu.fused_ops import get_config, inject_npu_attention, patch_layernorm
from ltx_npu.ulysses_attn import inject_ulysses_attention, install_sequence_parallel_hooks

if TYPE_CHECKING:
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.registry import Registry
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.types import Audio
    from ltx_pipelines.utils.args import ImageConditioningInput

logger = logging.getLogger(__name__)


class _SilentAudioDecoder:
    """Drop-in replacement that returns None audio without touching the vocoder."""

    def __call__(self, latent):
        return None


class ParallelDistilledPipeline:
    """Wraps DistilledPipeline with Ulysses SP for multi-card DiT inference.

    Each rank loads the full model and runs the full pipeline. The only modification
    is that self-attention in the DiT transformer is wrapped with Ulysses AllToAll,
    so each rank computes attention on a subset of heads while seeing the full sequence.

    For a proper sequence-split implementation (where each rank processes only local_T
    tokens through ALL layers), the TransformerArgs need to be split/gathered around
    the transformer blocks. This is deferred to a future optimization.
    """

    def __init__(
        self,
        ctx: DeviceContext,
        pcfg: ParallelConfig,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.ctx = ctx
        self.pcfg = pcfg
        self.timer = TimingReport(ctx)
        self.profiler = PipelineProfiler(ctx)
        self._inject_done = False

        install_hooks(self.profiler)

        from ltx_pipelines.distilled import DistilledPipeline

        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=distilled_checkpoint_path,
            gemma_root=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            device=ctx.get_device(),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )

        if pcfg.is_parallel:
            self._build_fsdp_transformer()
            self._hook_fsdp_transformer_ctx()
            self._hook_resident_text_encoder()

        self._hook_denoising_loop_timing()

        if pcfg.vae_parallel and pcfg.is_parallel:
            self._hook_vae_parallel()

        self._patched_pipeline_call = self._pipeline.__call__
        self._fix_audio_vocoder_dtype()

    @staticmethod
    def _fix_builder_model_path(stage) -> None:
        """Fix model_path when it's a directory instead of a safetensors file.

        safetensors >= 0.7.0 requires a file path, not a directory.
        When model_path is a directory, find the distilled safetensors file.
        """
        import os
        from dataclasses import replace as dc_replace

        builder = stage._transformer_builder
        path = builder.model_path
        if isinstance(path, str) and os.path.isdir(path):
            candidates = [f for f in os.listdir(path) if f.endswith(".safetensors") and "distilled" in f and "lora" not in f]
            if candidates:
                resolved = os.path.join(path, candidates[0])
                stage._transformer_builder = dc_replace(builder, model_path=resolved)
                logger.info("Fixed model_path: %s → %s", path, resolved)

    def _fix_audio_vocoder_dtype(self) -> None:
        """Replace audio_decoder with silent stub to avoid vocoder dtype crash on NPU.

        The vocoder loads with bfloat16 biases but receives float32 input,
        causing RuntimeError on NPU. Since audio is out of adaptation scope,
        we proactively replace it with a silent audio stub.
        """
        if self.ctx.device_type == "npu":
            logger.info("Replacing audio decoder with silent stub (NPU vocoder dtype workaround).")
            self._pipeline.audio_decoder = _SilentAudioDecoder()
        self._patched_pipeline_call = self._pipeline.__call__

    def _build_fsdp_transformer(self) -> None:
        """Build transformer on CPU, inject Ulysses SP, wrap with FSDP.

        The FSDP model is stored as self._fsdp_transformer and reused for
        every generate call — no build/free/cleanup cycle during inference.
        """
        from ltx_npu.fsdp_manager import shard_transformer

        stage = self._pipeline.stage
        self._fix_builder_model_path(stage)
        logger.info("Building transformer on CPU for FSDP sharding...")
        model = stage._build_transformer(device=torch.device("cpu"))

        fused_cfg = get_config()

        velocity_model = model.velocity_model

        if fused_cfg.algo != 0 or fused_cfg.fast_layernorm:
            if fused_cfg.algo != 0:
                n_attn = inject_npu_attention(velocity_model)
                logger.info("Injected NPUAttention (ALGO=%d) into %d modules", fused_cfg.algo, n_attn)
            if fused_cfg.fast_layernorm:
                n_ln = patch_layernorm(velocity_model)
                logger.info("Patched %d LayerNorm modules with fast_layernorm", n_ln)

        if self.pcfg.sp_group is not None:
            inject_ulysses_attention(velocity_model, self.pcfg.sp_group)
            install_sequence_parallel_hooks(velocity_model, self.pcfg.sp_group)
            logger.info(
                "Installed e2e sequence parallel (degree=%d) before FSDP wrap",
                self.pcfg.ulysses_degree,
            )

        self._fsdp_transformer = shard_transformer(
            model,
            device_id=self.ctx.get_device(),
        )
        self._fsdp_transformer.eval()

        n_cached = install_freqs_cache_on_model(self._fsdp_transformer)
        if n_cached:
            logger.info("Installed RoPE freqs cache on %d preprocessors", n_cached)

        logger.info("FSDP transformer ready on %s", self.ctx.get_device())

    def _hook_fsdp_transformer_ctx(self) -> None:
        """Replace DiffusionStage._transformer_ctx to yield the FSDP model directly.

        Bypasses gpu_model entirely — no build/to_meta/cleanup_memory.
        The FSDP model stays sharded on device across all generate calls.
        """
        fsdp_transformer = self._fsdp_transformer
        profiler = self.profiler

        @contextmanager
        def fsdp_ctx(streaming_prefetch_count=None, **kwargs):
            label = profiler.next_label("X0Model")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                yield fsdp_transformer
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        self._pipeline.stage._transformer_ctx = fsdp_ctx

    def _hook_resident_text_encoder(self) -> None:
        """Pre-build Gemma text encoder and keep it resident on device.

        Eliminates the ~6s per-generate disk I/O for Gemma loading.
        The model stays on device across all generate calls.
        """
        prompt_encoder = self._pipeline.prompt_encoder
        device = self.ctx.get_device()
        dtype = prompt_encoder._dtype

        logger.info("Building Gemma text encoder on device (resident mode)...")
        self._resident_text_encoder = (
            prompt_encoder._text_encoder_builder
            .build(device=device, dtype=dtype)
            .eval()
        )
        logger.info("Gemma text encoder resident on %s", device)

        cached_encoder = self._resident_text_encoder
        profiler = self.profiler

        @contextmanager
        def resident_text_encoder_ctx(streaming_prefetch_count=None):
            label = profiler.next_label("GemmaTextEncoder")
            profiler.sync()
            t0 = time.perf_counter()
            try:
                yield cached_encoder
            finally:
                profiler.sync()
                profiler.add(f"use {label}", time.perf_counter() - t0)
                profiler._last_gpu_model_exit = time.perf_counter()

        prompt_encoder._text_encoder_ctx = resident_text_encoder_ctx

    def _hook_denoising_loop_timing(self) -> None:
        """Monkey-patch euler_denoising_loop in both samplers AND blocks modules."""
        import time
        import ltx_pipelines.utils.samplers as samplers_mod
        import ltx_pipelines.utils.blocks as blocks_mod

        ctx = self.ctx
        pcfg = self.pcfg
        original_loop = samplers_mod.euler_denoising_loop

        def timed_euler_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
            if ctx is not None:
                ctx.synchronize()
            if dist.is_initialized():
                dist.barrier()
            t0 = time.perf_counter()
            result = original_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser)
            if ctx is not None:
                ctx.synchronize()
            if dist.is_initialized():
                dist.barrier()
            elapsed = time.perf_counter() - t0
            n_steps = len(sigmas) - 1
            if pcfg.is_main:
                logger.info("[DiT Loop] %d steps in %.3fs (%.3fs/step)", n_steps, elapsed, elapsed / max(1, n_steps))
            return result

        samplers_mod.euler_denoising_loop = timed_euler_loop
        blocks_mod.euler_denoising_loop = timed_euler_loop

    def _hook_vae_parallel(self) -> None:
        """Monkey-patch video_decoder to use VAE spatial patch parallel."""
        from ltx_npu.vae_parallel import VAEParallelContext, vae_parallel_decode

        vae_ctx = VAEParallelContext(
            world_size=self.pcfg.world_size,
            rank=self.pcfg.rank,
            device=self.ctx.get_device(),
        )

        video_decoder_block = self._pipeline.video_decoder
        original_vd_call = video_decoder_block.__class__.__call__

        pcfg = self.pcfg

        def patched_vd_call(self_vd, latent, tiling_config=None, generator=None):
            decoder = self_vd._decoder_builder.build(device=self_vd._device, dtype=self_vd._dtype).to(self_vd._device).eval()
            with vae_parallel_decode(decoder, vae_ctx):
                chunks = list(decoder.decode_video(latent, tiling_config, generator))
            from ltx_pipelines.utils.gpu_model import gpu_model
            decoder.to("meta")
            return iter(chunks)

        import types
        video_decoder_block.__call__ = types.MethodType(patched_vd_call, video_decoder_block)
        logger.info("Hooked VAE parallel decode (grid=%dx%d)", vae_ctx.h_split, vae_ctx.w_split)

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio] | None:
        video, audio = self._patched_pipeline_call(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if hasattr(self, "_fsdp_transformer"):
            clear_freqs_cache_on_model(self._fsdp_transformer)

        if self.pcfg.is_main:
            return video, audio
        return None

    def run_with_timing(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        warmup: bool = True,
    ) -> tuple[Iterator[torch.Tensor], Audio] | None:
        """Run inference with optional warmup and timing report."""

        call_kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images,
            tiling_config=tiling_config, enhance_prompt=enhance_prompt,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if warmup:
            logger.info("Running warmup inference (not timed)...")
            result = self._patched_pipeline_call(**call_kwargs)
            if result is not None:
                # Consume video iterator to ensure full execution
                for _ in result[0]:
                    pass
            if dist.is_initialized():
                dist.barrier()
            logger.info("Warmup complete.")

        self.timer = TimingReport(self.ctx)
        self.profiler.reset()
        with self.timer.stage("Total Inference"):
            result = self._patched_pipeline_call(**call_kwargs)
            if self.pcfg.is_main and result is not None:
                video_chunks = list(result[0])
                audio = result[1]

        if self.pcfg.is_main:
            print(self.timer.report())
            print(self.profiler.report())
            return iter(video_chunks), audio
        return None

