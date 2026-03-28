import inspect
import logging
import os
import shutil
import sys
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES, detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class TeeStream:
    """Mirror stdout/stderr to a file while preserving console output."""

    def __init__(self, original_stream, log_stream):
        self.original_stream = original_stream
        self.log_stream = log_stream
        self.encoding = getattr(original_stream, "encoding", "utf-8")

    def write(self, data):
        self.original_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.original_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.original_stream.isatty()

    def fileno(self):
        return self.original_stream.fileno()

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class ScreenLoggerContext:
    """Redirect stdout/stderr to both console and log file, and configure logging."""

    def __init__(self, log_file: str | Path):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._log_fp = self.log_path.open("a", encoding="utf-8", buffering=1)

        sys.stdout = TeeStream(self._stdout, self._log_fp)
        sys.stderr = TeeStream(self._stderr, self._log_fp)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        logging.info("screen log file: %s", self.log_path.resolve())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.shutdown()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self._log_fp is not None:
            self._log_fp.close()


def sync_device(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def record_time(timings: dict[str, float], key: str, t0: float, device: torch.device) -> None:
    sync_device(device)
    timings[key] = time.perf_counter() - t0


def log_timings(title: str, timings: dict[str, float]) -> None:
    logging.info("========== %s ==========", title)
    for k, v in timings.items():
        logging.info("[TIMING] %-28s %.3f s", k, v)
    logging.info("====================================")


def log_transformer_debug(transformer, stage_name: str) -> None:
    """Print lightweight debug info for the loaded transformer."""
    try:
        candidates = []
        total = 0
        for name, module in transformer.named_modules():
            has_qkv = all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v", "to_out", "heads"))
            if has_qkv:
                total += 1
                if len(candidates) < 20:
                    candidates.append(f"{name}:{module.__class__.__name__}[to_q,to_k,to_v,to_out,heads]")

        logging.info("[ENTRY-DEBUG] %s transformer class: %s", stage_name, transformer.__class__.__name__)
        logging.info("[ENTRY-DEBUG] %s attention-like module count: %s", stage_name, total)
        if candidates:
            logging.info("[ENTRY-DEBUG] %s attention-like samples: %s", stage_name, "; ".join(candidates))
    except Exception as e:
        logging.warning("[ENTRY-DEBUG] failed to inspect %s transformer: %r", stage_name, e)


def default_sidecar_path(output_path: str, suffix: str) -> Path:
    return Path(output_path).expanduser().resolve().with_suffix(suffix)


def default_current_dir_sidecar_path(output_path: str, suffix: str) -> Path:
    output_stem = Path(output_path).expanduser().stem or "ltx_output"
    return (Path.cwd() / output_stem).with_suffix(suffix)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


ASCEND_PROFILE_DEFAULTS = {
    "wait": 0,
    "warmup": 0,
    "active": 10,
    "repeat": 1,
    "skip_first": 0,
    "record_shapes": False,
    "memory": False,
    "with_stack": False,
    "with_modules": False,
    "with_flops": False,
    "level": "Level1",
    "aic_metrics": "PipeUtilization",
    "l2_cache": False,
    "export_db": False,
    "analyse": True,
    "data_simplification": None,
    "worker_name": None,
}


def call_with_supported_kwargs(func, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in parameters and v is not None}
    skipped = sorted(set(kwargs) - set(filtered))
    if skipped:
        logging.info("[ASCEND-PROFILE] skip unsupported kwargs for %s: %s", func.__name__, ", ".join(skipped))
    return func(**filtered)


def resolve_ascend_enum(enum_container, name: str, default=None):
    return getattr(enum_container, name, default)


def build_ascend_experimental_config(profiler_module, args):
    config_kwargs = {}

    export_text = getattr(profiler_module, "ExportType", None)
    if export_text is not None and hasattr(export_text, "Text"):
        if args.ascend_profile_export_db and hasattr(export_text, "Db"):
            config_kwargs["export_type"] = [export_text.Text, export_text.Db]
        else:
            config_kwargs["export_type"] = export_text.Text

    profiler_level_enum = getattr(profiler_module, "ProfilerLevel", None)
    if profiler_level_enum is not None:
        level_value = resolve_ascend_enum(profiler_level_enum, args.ascend_profile_level)
        if level_value is not None:
            config_kwargs["profiler_level"] = level_value

    aic_metrics_enum = getattr(profiler_module, "AiCMetrics", None)
    if aic_metrics_enum is not None:
        aic_value = resolve_ascend_enum(aic_metrics_enum, args.ascend_profile_aic_metrics)
        if aic_value is not None:
            config_kwargs["aic_metrics"] = aic_value

    config_kwargs["l2_cache"] = args.ascend_profile_l2_cache
    config_kwargs["msprof_tx"] = False
    config_kwargs["op_attr"] = False
    config_kwargs["record_op_args"] = False
    if args.ascend_profile_data_simplification is not None:
        config_kwargs["data_simplification"] = args.ascend_profile_data_simplification

    experimental_ctor = getattr(profiler_module, "_ExperimentalConfig", None)
    if experimental_ctor is None:
        logging.warning("[ASCEND-PROFILE] torch_npu.profiler._ExperimentalConfig not found; using default profiler config")
        return None

    return call_with_supported_kwargs(experimental_ctor, **config_kwargs)


def iter_dir_tree_lines(root: Path, max_depth: int = 3, _depth: int = 0):
    if _depth == 0:
        yield f"{root.name}/"
    if _depth >= max_depth:
        return

    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except FileNotFoundError:
        return

    for entry in entries:
        prefix = "  " * (_depth + 1)
        suffix = "/" if entry.is_dir() else ""
        yield f"{prefix}{entry.name}{suffix}"
        if entry.is_dir():
            yield from iter_dir_tree_lines(entry, max_depth=max_depth, _depth=_depth + 1)


def log_dir_tree(root: Path, title: str, max_depth: int = 3) -> None:
    if not root.exists():
        logging.warning("%s path does not exist: %s", title, root)
        return
    logging.info("%s", title)
    for line in iter_dir_tree_lines(root, max_depth=max_depth):
        logging.info("[DIR] %s", line)


def finalize_ascend_profiler_output(profile_root: Path, final_output_dir: Path) -> Path | None:
    if profile_root is None:
        return None

    if not profile_root.exists():
        logging.warning("[ASCEND-PROFILE] profile root does not exist: %s", profile_root)
        return None

    candidates = sorted(profile_root.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    if not candidates:
        logging.warning("[ASCEND-PROFILE] ASCEND_PROFILER_OUTPUT not found under: %s", profile_root)
        log_dir_tree(profile_root, "[ASCEND-PROFILE] temporary profile root tree", max_depth=4)
        return None

    source_dir = candidates[-1]
    final_output_dir = final_output_dir.expanduser().resolve()
    if final_output_dir.exists():
        logging.info("[ASCEND-PROFILE] removing existing final output dir: %s", final_output_dir)
        shutil.rmtree(final_output_dir)

    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, final_output_dir)
    logging.info("[ASCEND-PROFILE] copied final output dir: %s -> %s", source_dir, final_output_dir)
    log_dir_tree(final_output_dir, "[ASCEND-PROFILE] final output tree", max_depth=2)

    try:
        shutil.rmtree(profile_root)
        logging.info("[ASCEND-PROFILE] removed temporary profile root: %s", profile_root)
    except Exception:
        logging.exception("[ASCEND-PROFILE] failed to remove temporary profile root: %s", profile_root)

    return final_output_dir


def create_ascend_profiler(enable_ascend_profile: bool, output_path: str, runtime_device: torch.device):
    if not enable_ascend_profile:
        return nullcontext(None), None

    if runtime_device.type != "npu":
        raise RuntimeError("LTX_ENABLE_PROFILE=1 仅支持 NPU 设备，当前 device=%s" % runtime_device)

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("已开启 Ascend NPU profile，但当前环境未安装 torch_npu") from exc

    profiler_module = torch_npu.profiler
    output_stem = Path(output_path).expanduser().stem or "ltx_output"
    profile_dir = Path.cwd() / f".{output_stem}.ascend_profile_tmp"
    profile_dir.mkdir(parents=True, exist_ok=True)

    defaults = ASCEND_PROFILE_DEFAULTS
    handler_kwargs = {}
    if defaults["worker_name"]:
        handler_kwargs["worker_name"] = defaults["worker_name"]
    if defaults["analyse"] is not None:
        handler_kwargs["analyse_flag"] = defaults["analyse"]

    on_trace_ready = call_with_supported_kwargs(
        profiler_module.tensorboard_trace_handler,
        dir_name=str(profile_dir),
        **handler_kwargs,
    )

    activities = []
    if hasattr(profiler_module, "ProfilerActivity"):
        activities.append(profiler_module.ProfilerActivity.CPU)
        activities.append(profiler_module.ProfilerActivity.NPU)

    schedule = profiler_module.schedule(
        wait=defaults["wait"],
        warmup=defaults["warmup"],
        active=defaults["active"],
        repeat=defaults["repeat"],
        skip_first=defaults["skip_first"],
    )

    profiler_args = type("AscendProfilerDefaults", (), {
        "ascend_profile_export_db": defaults["export_db"],
        "ascend_profile_level": defaults["level"],
        "ascend_profile_aic_metrics": defaults["aic_metrics"],
        "ascend_profile_l2_cache": defaults["l2_cache"],
        "ascend_profile_data_simplification": defaults["data_simplification"],
    })()
    experimental_config = build_ascend_experimental_config(profiler_module, profiler_args)

    profiler_cm = call_with_supported_kwargs(
        profiler_module.profile,
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=defaults["record_shapes"],
        profile_memory=defaults["memory"],
        with_stack=defaults["with_stack"],
        with_modules=defaults["with_modules"],
        with_flops=defaults["with_flops"],
        experimental_config=experimental_config,
    )

    logging.info("[ASCEND-PROFILE] enabled=True")
    logging.info("[ASCEND-PROFILE] output_dir=%s", profile_dir)
    logging.info(
        "[ASCEND-PROFILE] switch source: environment variable LTX_ENABLE_PROFILE=1"
    )
    logging.info(
        "[ASCEND-PROFILE] schedule(wait=%s, warmup=%s, active=%s, repeat=%s, skip_first=%s)",
        defaults["wait"],
        defaults["warmup"],
        defaults["active"],
        defaults["repeat"],
        defaults["skip_first"],
    )
    logging.info(
        "[ASCEND-PROFILE] flags(record_shapes=%s, profile_memory=%s, with_stack=%s, with_modules=%s, with_flops=%s)",
        defaults["record_shapes"],
        defaults["memory"],
        defaults["with_stack"],
        defaults["with_modules"],
        defaults["with_flops"],
    )
    logging.info(
        "[ASCEND-PROFILE] options(level=%s, aic_metrics=%s, l2_cache=%s, export_db=%s, analyse=%s, data_simplification=%s)",
        defaults["level"],
        defaults["aic_metrics"],
        defaults["l2_cache"],
        defaults["export_db"],
        defaults["analyse"],
        defaults["data_simplification"],
    )
    return profiler_cm, profile_dir

def ascend_profiler_step(profiler, step_name: str) -> None:
    if profiler is None:
        return
    profiler.step()
    logging.info("[ASCEND-PROFILE] step=%s", step_name)


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.

    Stage 1 generates video at half of the target resolution with CFG guidance
    (assuming full model is used), then Stage 2 upsamples by 2x and refines
    using a distilled LoRA for higher quality output.

    Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )
        self.stage_2_model_ledger = self.stage_1_model_ledger.with_additional_loras(
            loras=distilled_lora,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.last_timings: dict[str, float] = {}

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        ascend_profiler=None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        timings: dict[str, float] = OrderedDict()
        t_pipeline_all = time.perf_counter()

        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        # 1) encode prompts
        t0 = time.perf_counter()
        ctx_p, ctx_n = encode_prompts(
            [prompt, negative_prompt],
            self.stage_1_model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        record_time(timings, "encode_prompts", t0, self.device)
        ascend_profiler_step(ascend_profiler, "encode_prompts")

        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1 output shape
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        # 2) stage1 conditionings
        t0 = time.perf_counter()
        video_encoder = self.stage_1_model_ledger.video_encoder()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        record_time(timings, "stage1_conditionings", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage1_conditionings")

        del video_encoder
        cleanup_memory()

        # 3) stage1 model load
        t0 = time.perf_counter()
        transformer = self.stage_1_model_ledger.transformer()
        log_transformer_debug(transformer, "stage1")
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        record_time(timings, "stage1_model_load", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage1_model_load")

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider_factory=create_multimodal_guider_factory(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                ),
            )

        # 4) stage1 denoise
        t0 = time.perf_counter()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )
        record_time(timings, "stage1_denoise", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage1_denoise")

        del transformer
        cleanup_memory()

        # 5) stage2 upsample
        t0 = time.perf_counter()
        video_encoder = self.stage_1_model_ledger.video_encoder()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )
        record_time(timings, "stage2_upsample", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage2_upsample")

        # Stage 2 output shape
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # 6) stage2 conditionings
        t0 = time.perf_counter()
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        record_time(timings, "stage2_conditionings", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage2_conditionings")

        del video_encoder
        cleanup_memory()

        # 7) stage2 model load
        t0 = time.perf_counter()
        transformer = self.stage_2_model_ledger.transformer()
        log_transformer_debug(transformer, "stage2")
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        record_time(timings, "stage2_model_load", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage2_model_load")

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

        # 8) stage2 denoise
        t0 = time.perf_counter()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )
        record_time(timings, "stage2_denoise", t0, self.device)
        ascend_profiler_step(ascend_profiler, "stage2_denoise")

        del transformer
        cleanup_memory()

        # 9) decode video
        t0 = time.perf_counter()
        decoded_video = vae_decode_video(
            video_state.latent,
            self.stage_2_model_ledger.video_decoder(),
            tiling_config,
            generator,
        )
        record_time(timings, "decode_video", t0, self.device)
        ascend_profiler_step(ascend_profiler, "decode_video")

        # 10) decode audio
        t0 = time.perf_counter()
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.stage_2_model_ledger.audio_decoder(),
            self.stage_2_model_ledger.vocoder(),
        )
        record_time(timings, "decode_audio", t0, self.device)
        ascend_profiler_step(ascend_profiler, "decode_audio")

        record_time(timings, "pipeline_total", t_pipeline_all, self.device)
        self.last_timings = timings
        log_timings("PIPELINE TIMINGS", timings)

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    t_e2e = time.perf_counter()

    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save the screen log. Default: ./<output_stem>.log",
    )
    args = parser.parse_args()

    log_file = args.log_file or str(default_current_dir_sidecar_path(args.output_path, ".log"))
    enable_ascend_profile = env_flag("LTX_ENABLE_PROFILE", default=False)

    with ScreenLoggerContext(log_file):
        logging.info("device=%s dtype=%s", device, torch.bfloat16)
        logging.info("output_path=%s", Path(args.output_path).expanduser())
        logging.info("log_file=%s", Path(log_file).expanduser())
        logging.info("ascend_profile_enabled=%s", enable_ascend_profile)
        logging.info("LTX_ENABLE_PROFILE=%s", os.getenv("LTX_ENABLE_PROFILE", "<unset>"))

        try:
            t0 = time.perf_counter()
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=args.checkpoint_path,
                distilled_lora=args.distilled_lora,
                spatial_upsampler_path=args.spatial_upsampler_path,
                gemma_root=args.gemma_root,
                loras=tuple(args.lora) if args.lora else (),
                quantization=args.quantization,
            )
            pipeline_init_time = time.perf_counter() - t0

            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

            profiler_cm, profile_dir = create_ascend_profiler(
                enable_ascend_profile=enable_ascend_profile,
                output_path=args.output_path,
                runtime_device=pipeline.device,
            )
            final_profile_output_dir = Path.cwd() / "ASCEND_PROFILER_OUTPUT"
            if profile_dir is not None:
                logging.info("[ASCEND-PROFILE] temporary results root: %s", profile_dir)
                logging.info("[ASCEND-PROFILE] final exported dir: %s", final_profile_output_dir)

            with profiler_cm as ascend_profiler:
                t0 = time.perf_counter()
                video, audio = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    seed=args.seed,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    num_inference_steps=args.num_inference_steps,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=args.video_cfg_guidance_scale,
                        stg_scale=args.video_stg_guidance_scale,
                        rescale_scale=args.video_rescale_scale,
                        modality_scale=args.a2v_guidance_scale,
                        skip_step=args.video_skip_step,
                        stg_blocks=args.video_stg_blocks,
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=args.audio_cfg_guidance_scale,
                        stg_scale=args.audio_stg_guidance_scale,
                        rescale_scale=args.audio_rescale_scale,
                        modality_scale=args.v2a_guidance_scale,
                        skip_step=args.audio_skip_step,
                        stg_blocks=args.audio_stg_blocks,
                    ),
                    images=args.images,
                    tiling_config=tiling_config,
                    ascend_profiler=ascend_profiler,
                )
                sync_device(pipeline.device)
                pipeline_run_time = time.perf_counter() - t0

            if profile_dir is not None:
                finalized_profile_dir = finalize_ascend_profiler_output(
                    profile_root=profile_dir,
                    final_output_dir=final_profile_output_dir,
                )
                if finalized_profile_dir is not None:
                    logging.info("[ASCEND-PROFILE] ready: %s", finalized_profile_dir)

            t0 = time.perf_counter()
            encode_video(
                video=video,
                fps=args.frame_rate,
                audio=audio,
                output_path=args.output_path,
                video_chunks_number=video_chunks_number,
            )
            encode_video_time = time.perf_counter() - t0

            sync_device(pipeline.device)
            e2e_total = time.perf_counter() - t_e2e

            logging.info("========== E2E TIMINGS ==========")
            logging.info("[TIMING] %-28s %.3f s", "pipeline_init", pipeline_init_time)
            logging.info("[TIMING] %-28s %.3f s", "pipeline_run", pipeline_run_time)
            logging.info("[TIMING] %-28s %.3f s", "encode_video", encode_video_time)
            logging.info("[TIMING] %-28s %.3f s", "e2e_total", e2e_total)
            logging.info("=================================")
        except Exception:
            logging.exception("pipeline failed with an exception")
            raise



if __name__ == "__main__":
    main()
