
# LTX_RANK0_ONLY_OUTPUT_ARGV_PATCH
# Only rank0 writes the requested final output path.
# Non-main ranks rewrite --output-path in sys.argv before argparse parses it.
import os as _ltx_rank_os
import sys as _ltx_rank_sys

_ltx_rank = int(_ltx_rank_os.getenv("RANK", _ltx_rank_os.getenv("LOCAL_RANK", "0")))

if _ltx_rank != 0 and "--output-path" in _ltx_rank_sys.argv:
    _i = _ltx_rank_sys.argv.index("--output-path")
    if _i + 1 < len(_ltx_rank_sys.argv):
        _orig_out = _ltx_rank_sys.argv[_i + 1]
        _out_dir = _ltx_rank_os.path.dirname(_orig_out) or "."
        _out_base = _ltx_rank_os.path.basename(_orig_out)
        _ltx_rank_sys.argv[_i + 1] = _ltx_rank_os.path.join(
            _out_dir,
            f".rank{_ltx_rank}_{_out_base}",
        )

# === ULTIMATE GOD MODE PATCH ===
import torch
import torch.nn.functional as F
import torch.nn as nn
import inspect

# [A] 解决 NPU F.conv1d 精度崩溃问题 (保留这个保命符)
if not hasattr(F, "_npu_patched"):
    _orig_F_conv1d = F.conv1d
    def _patched_F_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype: input = input.to(weight.dtype)
        return _orig_F_conv1d(input, weight, bias, stride, padding, dilation, groups)
    F.conv1d = _patched_F_conv1d
    
    _orig_nn_conv1d = nn.Conv1d.forward
    def _patched_nn_conv1d(self, input):
        if input.dtype != self.weight.dtype: input = input.to(self.weight.dtype)
        return _orig_nn_conv1d(self, input)
    nn.Conv1d.forward = _patched_nn_conv1d
    F._npu_patched = True

# [B] 解决 Ulysses 并行导致音频相位断裂的杂音问题
try:
    from ltx_core.model.transformer.model import LTXModel
    if not hasattr(LTXModel, "_audio_patched"):
        sig = inspect.signature(LTXModel.__init__)
        
        LTXModel._audio_patched = True
except ImportError:
    print("⚠️ 找不到 LTXModel，拦截失败。")
# ===============================
import torch.nn.functional as F
import torch.nn as nn

if not hasattr(F, "_npu_patched"):
    # 拦截函数级卷积 (F.conv1d / F.conv2d)
    _orig_F_conv1d = F.conv1d
    def _patched_F_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_conv1d(input, weight, bias, stride, padding, dilation, groups)
    F.conv1d = _patched_F_conv1d

    _orig_F_conv2d = F.conv2d
    def _patched_F_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_conv2d(input, weight, bias, stride, padding, dilation, groups)
    F.conv2d = _patched_F_conv2d

    # 拦截类级卷积 (nn.Conv1d / nn.Conv2d)
    _orig_nn_conv1d = nn.Conv1d.forward
    def _patched_nn_conv1d(self, input):
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_nn_conv1d(self, input)
    nn.Conv1d.forward = _patched_nn_conv1d

    _orig_nn_conv2d = nn.Conv2d.forward
    def _patched_nn_conv2d(self, input):
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_nn_conv2d(self, input)
    nn.Conv2d.forward = _patched_nn_conv2d

    F._npu_patched = True
# =================================================================
import torch
from ltx_core.model.audio_vae.vocoder import Vocoder
if not hasattr(Vocoder, "_fp32_patched"):
    _orig_vocoder_forward = Vocoder.forward
    def _patched_vocoder_forward(self, mel_spec):
        if _ltx_rank_os.getenv("LTX_VOCODER_FORCE_FP32", "0") == "1":
            if next(self.parameters()).dtype != torch.float32:
                self.to(torch.float32)
        else:
            try:
                p = next(self.parameters())
            except StopIteration:
                p = None
            if p is not None and torch.is_tensor(mel_spec):
                if mel_spec.device != p.device:
                    mel_spec = mel_spec.to(p.device)
                if mel_spec.is_floating_point() and mel_spec.dtype != p.dtype:
                    mel_spec = mel_spec.to(p.dtype)
        return _orig_vocoder_forward(self, mel_spec)
    Vocoder.forward = _patched_vocoder_forward
    Vocoder._fp32_patched = True
# =================================================================
import torch.nn as nn
if not hasattr(nn.Conv1d, "_npu_patched"):
    _orig_conv1d = nn.Conv1d.forward
    def _patched_conv1d(self, input):
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_conv1d(self, input)
    nn.Conv1d.forward = _patched_conv1d
    nn.Conv1d._npu_patched = True

    _orig_conv2d = nn.Conv2d.forward
    def _patched_conv2d(self, input):
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return _orig_conv2d(self, input)
    nn.Conv2d.forward = _patched_conv2d
# =================================================================
# === Precision Debugging Utilities ===
# Controlled by --debug-precision flag. Two hooks:
#   1. Loop-level: dumps video/audio latent states at step 0 of first denoising loop call
#   2. DiT-level:  dumps X0Model input/output tensors on first forward call
# All tensors are saved as .pt files + human-readable stats .txt for cross-run comparison.
# =================================================================
import os as _debug_os
import logging as _debug_logging
import types as _debug_types
from typing import Any as _Any

_debug_logger = _debug_logging.getLogger("precision_debug")


def _dump_tensor_stats(tensors: dict, filepath: str) -> None:
    """Write human-readable tensor stats (shape, min/max/mean/std/sum) to file."""
    import torch
    with open(filepath, "w") as f:
        for k, v in tensors.items():
            if torch.is_tensor(v):
                f.write(
                    f"{k}: shape={list(v.shape)}, dtype={v.dtype}, "
                    f"min={v.min().item():.6e}, max={v.max().item():.6e}, "
                    f"mean={v.mean().item():.6e}, std={v.std().item():.6e}, "
                    f"sum={v.sum().item():.6e}\n"
                )
            else:
                f.write(f"{k}: {v}\n")


def _save_tensors(tensors: dict, pt_path: str, stats_path: str) -> None:
    """Save tensor dict to .pt and write stats to .txt."""
    import torch
    torch.save(tensors, pt_path)
    _dump_tensor_stats(tensors, stats_path)


def _apply_precision_seed(seed: int, dump_dir: str) -> None:
    """Apply strong seed enforcement for reproducible precision debugging.

    Sets CPU/CUDA/NPU RNG seeds, forces deterministic CUDA algorithms,
    and logs seed configuration to ``dump_dir/seed_info.txt``.
    """
    import torch
    _debug_os.makedirs(dump_dir, exist_ok=True)

    # CPU RNG
    torch.manual_seed(seed)

    # CUDA RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # NPU RNG
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass

    # Environment hints
    _debug_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    info_lines = [
        f"seed={seed}",
        f"torch.initial_seed()={torch.initial_seed()}",
    ]
    if torch.cuda.is_available():
        info_lines.append(f"torch.cuda.initial_seed()={torch.cuda.initial_seed()}")
    try:
        if hasattr(torch, "npu"):
            info_lines.append(f"torch.npu.initial_seed()={torch.npu.initial_seed()}")
    except Exception:
        pass

    with open(_debug_os.path.join(dump_dir, "seed_info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    _debug_logger.info("Precision seed enforced: seed=%d, dump_dir=%s", seed, dump_dir)


def _install_loop_debug_hook(dump_dir: str) -> None:
    """Monkey-patch ``euler_denoising_loop`` to dump first-step latent states.

    Must be called **before** ``ParallelDistilledPipeline`` is created,
    because ``_hook_denoising_loop_timing`` wraps the module-level function.

    On the first call, dumps:
      - ``video_state.latent`` / ``audio_state.latent`` (initial noise)
      - ``denoised_video`` / ``denoised_audio`` (DiT output at step 0)
      - ``sigmas`` (noise schedule)
    """
    import torch
    from tqdm import tqdm
    from dataclasses import replace
    import ltx_pipelines.utils.samplers as _samplers_mod
    import ltx_pipelines.utils.blocks as _blocks_mod
    from ltx_pipelines.utils.helpers import post_process_latent

    _orig_loop = _samplers_mod.euler_denoising_loop
    _called_flag = [False]  # mutable to track first call

    def _step_one(state, denoised, stepper, sigmas, step_idx):
        """Inline equivalent of _step_state — avoids import fragility."""
        if state is None or denoised is None:
            return state
        denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))

    def _debug_loop(*args, **kwargs):
        is_first = not _called_flag[0]
        _called_flag[0] = True

        if not is_first:
            return _orig_loop(*args, **kwargs)

        # First call: manual iteration so we can intercept step 0.
        # Accept both old API (sigmas, video_state, audio_state, stepper, denoise_fn)
        # and new API (sigmas, video_state, audio_state, stepper, transformer, denoiser).
        if len(args) >= 6:
            sigmas, video_state, audio_state, stepper, transformer, denoiser = args[:6]
        else:
            sigmas, video_state, audio_state, stepper = args[:4]
            denoise_fn = args[4] if len(args) > 4 else None
            transformer = kwargs.get("transformer", None)
            denoiser = kwargs.get("denoiser", None)
            # Fallback: old-style denoise_fn is denoiser+transformer combined
            if denoise_fn is not None and transformer is None and denoiser is None:
                denoiser = denoise_fn  # denoise_fn(video_state, audio_state, sigmas, step_idx)

        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if denoiser is not None and transformer is not None:
                denoised_video, denoised_audio = denoiser(
                    transformer, video_state, audio_state, sigmas, step_idx
                )
            else:
                # Old-style: denoise_fn(video_state, audio_state, sigmas, step_idx)
                _dfn = denoiser if denoiser is not None else denoise_fn
                denoised_video, denoised_audio = _dfn(
                    video_state, audio_state, sigmas, step_idx
                )

            if step_idx == 0:
                _dump: dict = {"sigmas": sigmas.cpu().clone(), "step_idx": step_idx}
                if video_state is not None:
                    _dump["video_state_latent"] = video_state.latent.cpu().clone()
                    _dump["video_state_denoise_mask"] = video_state.denoise_mask.cpu().clone()
                if audio_state is not None:
                    _dump["audio_state_latent"] = audio_state.latent.cpu().clone()
                    _dump["audio_state_denoise_mask"] = audio_state.denoise_mask.cpu().clone()
                if denoised_video is not None:
                    _dump["denoised_video"] = denoised_video.cpu().clone()
                if denoised_audio is not None:
                    _dump["denoised_audio"] = denoised_audio.cpu().clone()

                rank = _debug_os.environ.get("RANK", "0")
                _save_tensors(
                    _dump,
                    _debug_os.path.join(dump_dir, f"rank{rank}_step0_loop_tensors.pt"),
                    _debug_os.path.join(dump_dir, f"rank{rank}_step0_loop_stats.txt"),
                )
                _debug_logger.info(
                    "[rank%s] Dumped step 0 loop tensors to %s", rank, dump_dir
                )

            video_state = _step_one(video_state, denoised_video, stepper, sigmas, step_idx)
            audio_state = _step_one(audio_state, denoised_audio, stepper, sigmas, step_idx)

        return (video_state, audio_state)

    _samplers_mod.euler_denoising_loop = _debug_loop
    try:
        _blocks_mod.euler_denoising_loop = _debug_loop
    except Exception:
        _debug_logger.debug("blocks_mod.euler_denoising_loop not patchable (old codebase)")
    _debug_logger.info("Installed denoising loop debug hook (step 0 tensor dump)")


def _install_dit_debug_hook(pipeline: _Any, dump_dir: str) -> None:
    """Patch ``X0Model.forward`` to dump first-call DiT internals.

    Must be called **after** ``ParallelDistilledPipeline`` is created
    (needs access to ``_fsdp_transformer.module``).

    On the first forward, dumps:
      - ``video.latent`` / ``audio.latent`` (Modality inputs)
      - ``video.timesteps`` / ``audio.timesteps``
      - ``vx`` / ``ax`` (velocity predictions from LTXModel)
      - ``denoised_video`` / ``denoised_audio`` (X0Model outputs)
    """
    import torch
    from ltx_core.utils import to_denoised

    if not hasattr(pipeline, "_fsdp_transformer"):
        _debug_logger.warning(
            "DiT debug hook skipped: pipeline has no _fsdp_transformer (single-card mode)"
        )
        return

    _fsdp_model = pipeline._fsdp_transformer
    _x0_model = _fsdp_model.module if hasattr(_fsdp_model, "module") else _fsdp_model

    _orig_forward = _x0_model.forward
    _counter = [0]

    def _patched_forward(self, video, audio, perturbations):
        if _counter[0] > 0:
            return _orig_forward(video, audio, perturbations)

        _counter[0] += 1

        vx, ax = _orig_forward(video, audio, perturbations)

        denoised_video = (
            to_denoised(video.latent, vx, video.timesteps)
            if vx is not None
            else None
        )
        denoised_audio = (
            to_denoised(audio.latent, ax, audio.timesteps)
            if ax is not None
            else None
        )

        _dump: dict = {}
        if video is not None:
            _dump["video_latent"] = video.latent.detach().cpu()
            if hasattr(video, "timesteps") and torch.is_tensor(video.timesteps):
                _dump["video_timesteps"] = video.timesteps.detach().cpu()
            else:
                _dump["video_timesteps"] = str(getattr(video, "timesteps", None))
        if audio is not None:
            _dump["audio_latent"] = audio.latent.detach().cpu()
            if hasattr(audio, "timesteps") and torch.is_tensor(audio.timesteps):
                _dump["audio_timesteps"] = audio.timesteps.detach().cpu()
            else:
                _dump["audio_timesteps"] = str(getattr(audio, "timesteps", None))
        if vx is not None:
            _dump["vx"] = vx.detach().cpu()
        if ax is not None:
            _dump["ax"] = ax.detach().cpu()
        if denoised_video is not None:
            _dump["denoised_video"] = denoised_video.detach().cpu()
        if denoised_audio is not None:
            _dump["denoised_audio"] = denoised_audio.detach().cpu()

        rank = _debug_os.environ.get("RANK", "0")
        _save_tensors(
            _dump,
            _debug_os.path.join(dump_dir, f"rank{rank}_step0_dit_internal.pt"),
            _debug_os.path.join(dump_dir, f"rank{rank}_step0_dit_stats.txt"),
        )
        _debug_logger.info(
            "[rank%s] Dumped DiT internal tensors to %s", rank, dump_dir
        )

        _x0_model.forward = _orig_forward  # restore original after first dump
        return vx, ax

    _x0_model.forward = _debug_types.MethodType(_patched_forward, _x0_model)
    _debug_logger.info("Installed DiT model debug hook (first-call tensor dump)")

# =================================================================

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
    parser.add_argument(
        "--debug-precision", action="store_true", default=False,
        help="Enable precision debugging: force fixed seed and dump first-step intermediate tensors "
             "to --debug-precision-dump-dir for cross-run comparison.",
    )
    parser.add_argument(
        "--debug-precision-dump-dir", type=str, default="./precision_debug",
        help="Directory to save precision debug tensors (default: ./precision_debug).",
    )
    parser.add_argument(
        "--audio-path", type=str, default=None,
        help="Path to external WAV audio file. When provided, this audio is muxed "
             "directly (bypasses model audio generation). Use this to avoid Ulysses SP "
             "audio quality issues on multi-card runs.",
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

    if args.debug_precision:
        _apply_precision_seed(args.seed, args.debug_precision_dump_dir)
        _install_loop_debug_hook(args.debug_precision_dump_dir)

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
        torch_compile=getattr(args,"compile",False),
    )

    if args.debug_precision:
        _install_dit_debug_hook(pipeline, args.debug_precision_dump_dir)

    if os.environ.get("LTX_AUDIO_TRACE") == "1":
        try:
            from ltx_npu.audio_trace import install_audio_trace_hooks
            _fsdp_model = getattr(pipeline, "_fsdp_transformer", None)
            if _fsdp_model is not None:
                _inner = _fsdp_model.module if hasattr(_fsdp_model, "module") else _fsdp_model
                install_audio_trace_hooks(_inner.velocity_model)
            else:
                install_audio_trace_hooks(pipeline._pipeline.model_ledger.transformer())
        except Exception as _e:
            logger.warning("audio_trace install failed: %s", _e)

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    if os.environ.get("LTX_NPU_PROFILE") == "1":
        from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        _prof_dir = os.environ.get("LTX_NPU_PROFILE_DIR", "./prof_data")
        os.makedirs(_prof_dir, exist_ok=True)
        _prof_name = os.environ.get("LTX_NPU_PROFILE_NAME", "npu_trace")
        _prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(f"{_prof_dir}/{_prof_name}"),
        )
        _prof.start()
        logger.info("NPU profiler started, output: %s/%s", _prof_dir, _prof_name)

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

    if os.environ.get("LTX_NPU_PROFILE") == "1":
        _prof.stop()
        logger.info("NPU profiler stopped")

    if pcfg.is_main and result is not None:
        video, audio = result
        if args.audio_path:
            from scipy.io import wavfile
            sr, waveform_np = wavfile.read(args.audio_path)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[:, None]
            waveform = torch.from_numpy(waveform_np.T.copy()).unsqueeze(0).float() / 32768.0
            from ltx_core.types import Audio
            audio = Audio(waveform=waveform, sampling_rate=sr)
            logger.info("Using external audio: %s (%dHz, %.1fs)",
                        args.audio_path, sr, waveform.shape[-1] / sr)
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
