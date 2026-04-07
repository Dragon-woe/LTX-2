"""Multi-card DistilledPipeline entry point for Ascend NPU.

Usage:
    # Single-card (equivalent to original DistilledPipeline)
    python run_distilled.py --distilled-checkpoint-path ... --prompt "..."

    # Multi-card with Ulysses SP
    torchrun --nproc_per_node=4 run_distilled.py --ulysses-degree 4 --prompt "..."

    # Multi-card with Ulysses SP + VAE Parallel
    torchrun --nproc_per_node=8 run_distilled.py --ulysses-degree 8 --vae-parallel --prompt "..."
"""

import logging
import os
import sys

import torch

import ltx_npu  # noqa: F401 — NPU runtime init

from ltx_npu.device_context import DeviceContext
from ltx_npu.parallel_config import ParallelConfig


def build_arg_parser():
    from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, detect_checkpoint_path
    from ltx_pipelines.utils.constants import detect_params

    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)

    parser.add_argument(
        "--ulysses-degree", type=int, default=1,
        help="Ulysses Sequence Parallelism degree. Must divide 32 (num_attention_heads). "
             "Use with torchrun --nproc_per_node=N where N equals this value.",
    )
    parser.add_argument(
        "--vae-parallel", action="store_true", default=False,
        help="Enable VAE spatial patch parallel decoding across all ranks.",
    )
    parser.add_argument(
        "--no-warmup", action="store_true", default=False,
        help="Skip warmup inference (for debugging; timing will include NPU compilation overhead).",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    ctx = DeviceContext.create()
    if "RANK" in os.environ:
        ctx.init_distributed()

    parser = build_arg_parser()
    args = parser.parse_args()

    pcfg = ParallelConfig.from_args(args)

    if pcfg.is_main:
        logger.info("Device: %s | World size: %d | Ulysses degree: %d | VAE parallel: %s",
                     ctx.device_type, pcfg.world_size, pcfg.ulysses_degree, pcfg.vae_parallel)

    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    from ltx_npu.pipeline_wrapper import ParallelDistilledPipeline

    pipeline = ParallelDistilledPipeline(
        ctx=ctx,
        pcfg=pcfg,
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    result = pipeline.run_with_timing(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        streaming_prefetch_count=args.streaming_prefetch_count,
        warmup=not args.no_warmup,
    )

    if pcfg.is_main and result is not None:
        video, audio = result
        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=video_chunks_number,
        )
        logger.info("Video saved to %s", args.output_path)


if __name__ == "__main__":
    main()
