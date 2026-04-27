"""Measure DiT denoising loop time ONLY, isolating it from model loading and other stages.

Usage:
  # Single card baseline
  python scripts/test_dit_only_timing.py --nproc 1

  # 2-card SP
  torchrun --nproc_per_node=2 --master_port=35000 scripts/test_dit_only_timing.py --nproc 2

This script:
1. Builds the transformer once
2. Creates dummy latent states matching real shapes
3. Runs the denoising loop (8 steps for Stage 1) with precise timing
4. Reports ONLY the denoising loop time (no model loading, no VAE, no prompt encoding)
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist

import ltx_npu  # noqa: F401

from ltx_npu.device_context import DeviceContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--checkpoint", type=str,
                        default="../models/LTX-2.3/ltx-2.3-22b-distilled.safetensors")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=65)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    ctx = DeviceContext.create()
    if "RANK" in os.environ:
        ctx.init_distributed()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = ctx.get_device()
    dtype = torch.bfloat16

    if rank == 0:
        logger.info("=== DiT-Only Timing Test ===")
        logger.info("Device: %s | Cards: %d | Resolution: %dx%d | Frames: %d | Steps: %d",
                     ctx.device_type, world_size, args.height, args.width, args.num_frames, args.steps)

    # Build transformer
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.transformer import LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, X0Model

    builder = Builder(
        model_path=args.checkpoint,
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    model = X0Model(builder.build(device=device)).to(device).eval()

    # Inject Ulysses SP if multi-card
    if world_size > 1:
        from ltx_npu.ulysses_attn import inject_ulysses_attention, install_sequence_parallel_hooks
        sp_group = dist.new_group(list(range(world_size)))
        inject_ulysses_attention(model.velocity_model, sp_group)
        install_sequence_parallel_hooks(model.velocity_model, sp_group)
        logger.info("[Rank %d] Installed E2E SP + Ulysses (degree=%d)", rank, world_size)

    # Create dummy inputs matching real shapes
    # Latent shape: (B, C, F', H', W') where F'=(F-1)/8+1, H'=H/32, W'=W/32
    latent_f = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128

    if rank == 0:
        logger.info("Latent shape: (1, %d, %d, %d, %d)", latent_channels, latent_f, latent_h, latent_w)

    # Create modality inputs
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.types import VideoLatentShape, VideoPixelShape

    pixel_shape = VideoPixelShape(batch=1, frames=args.num_frames, height=args.height, width=args.width, fps=25.0)
    latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    patchifier = VideoLatentPatchifier(patch_size=1)

    # Patchify dummy latent to get sequence length
    dummy_latent = torch.randn(1, latent_channels, latent_f, latent_h, latent_w, device=device, dtype=dtype)
    seq_len = patchifier.patchify(dummy_latent).shape[1]
    if rank == 0:
        logger.info("Sequence length (T): %d tokens", seq_len)

    # Create sigma schedule (8 distilled steps)
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES[:args.steps + 1], device=device)

    # Build dummy Modality with correct shapes
    video_latent = torch.randn(1, latent_channels, latent_f, latent_h, latent_w, device=device, dtype=dtype)
    patchified = patchifier.patchify(video_latent)

    timesteps = torch.full((1, patchified.shape[1]), sigmas[0].item(), device=device, dtype=dtype)
    positions = torch.zeros(1, 3, patchified.shape[1], device=device, dtype=torch.long)

    context = torch.randn(1, 256, 4096, device=device, dtype=dtype)
    context_mask = torch.ones(1, 256, device=device, dtype=dtype)

    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    # Simple denoiser: just call transformer
    def simple_denoiser(transformer, video_state, audio_state, sigmas, step_idx):
        sigma = sigmas[step_idx]
        video_mod = Modality(
            latent=video_state.latent,
            timesteps=timesteps * 0 + sigma,
            positions=positions,
            context=context,
            enabled=True,
            context_mask=context_mask,
        )
        perturbations = BatchedPerturbationConfig.empty(1)
        denoised_v, _ = transformer(video=video_mod, audio=None, perturbations=perturbations)
        from ltx_core.utils import to_denoised
        return to_denoised(video_state.latent, denoised_v, sigma), None

    # Create initial state
    from ltx_core.types import LatentState
    denoise_mask = torch.ones(1, seq_len, 1, device=device, dtype=dtype)
    video_state = LatentState(
        latent=video_latent,
        denoise_mask=denoise_mask,
        clean_latent=torch.zeros_like(video_latent),
    )

    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    stepper = EulerDiffusionStep()

    # Warmup: 1 step
    if rank == 0:
        logger.info("Warmup (1 step)...")
    _ = simple_denoiser(model, video_state, None, sigmas, 0)
    ctx.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # Timed run: full loop
    if rank == 0:
        logger.info("Running %d denoising steps...", args.steps)

    ctx.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()

    for step_idx in range(args.steps):
        denoised_v, _ = simple_denoiser(model, video_state, None, sigmas, step_idx)

    ctx.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    per_step = elapsed / args.steps

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DiT-Only Timing Results")
        print(f"{'='*60}")
        print(f"Cards:        {world_size}")
        print(f"Resolution:   {args.height}x{args.width}")
        print(f"Frames:       {args.num_frames}")
        print(f"Seq length:   {seq_len} tokens")
        print(f"Steps:        {args.steps}")
        print(f"Total time:   {elapsed:.3f}s")
        print(f"Per step:     {per_step:.3f}s")
        print(f"{'='*60}")

    # Cleanup
    del model
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
