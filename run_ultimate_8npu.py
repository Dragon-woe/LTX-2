from __future__ import annotations

from torch_npu.contrib import transfer_to_npu  # noqa: F401
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    HAS_TORCH_FSDP = True
except Exception:  # pragma: no cover - depends on runtime stack
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None
    HAS_TORCH_FSDP = False

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, euler_denoising_loop
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    get_device,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents


LOG = logging.getLogger("distilled_8npu")


def _log_rank0(message: str, *args) -> None:
    if not DIST.initialized or DIST.rank == 0:
        LOG.info(message, *args)


def _shape_str(x) -> str:
    if x is None:
        return "None"
    shape = getattr(x, "shape", None)
    return str(tuple(shape)) if shape is not None else str(type(x).__name__)


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


# -----------------------------------------------------------------------------
# distributed / NPU helpers
# -----------------------------------------------------------------------------


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


PROFILE_COMM = _env_flag("LTX_PROFILE_COMM", "0")
PROFILE_MEM = _env_flag("LTX_PROFILE_MEM", "0")
ENABLE_FSDP = _env_flag("LTX_ENABLE_FSDP", "0")
FSDP_USE_ORIG_PARAMS = _env_flag("LTX_FSDP_USE_ORIG_PARAMS", "1")
FSDP_SYNC_MODULE_STATES = _env_flag("LTX_FSDP_SYNC_MODULE_STATES", "0")
FSDP_FORWARD_PREFETCH = _env_flag("LTX_FSDP_FORWARD_PREFETCH", "0")
FSDP_FORCE_UNIFORM_DTYPE = _env_flag("LTX_FSDP_FORCE_UNIFORM_DTYPE", "1")


def _visible_device_count() -> int:
    if hasattr(torch, "npu"):
        try:
            return int(torch.npu.device_count())
        except Exception:
            return 0
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0
    return 0


def _reset_peak_memory_stats() -> None:
    device = DIST.device if DIST.initialized else get_device()
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.reset_peak_memory_stats()
        elif device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_memory_gb() -> tuple[float, float]:
    device = DIST.device if DIST.initialized else get_device()
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            alloc = float(torch.npu.max_memory_allocated()) / (1024 ** 3)
            reserv = float(torch.npu.max_memory_reserved()) / (1024 ** 3)
            return alloc, reserv
        if device.type == "cuda" and torch.cuda.is_available():
            alloc = float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
            reserv = float(torch.cuda.max_memory_reserved()) / (1024 ** 3)
            return alloc, reserv
    except Exception:
        pass
    return 0.0, 0.0


def _log_memory(tag: str) -> None:
    if not PROFILE_MEM:
        return
    alloc, reserv = _peak_memory_gb()
    LOG.info(
        "rank%d memory | %s | peak_alloc=%.2fGB peak_reserved=%.2fGB",
        DIST.rank,
        tag,
        alloc,
        reserv,
    )


def _timed_collective(name: str, fn, *args, **kwargs):
    if not PROFILE_COMM:
        return fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    device_synchronize()
    LOG.info("rank%d collective | %s | elapsed=%s", DIST.rank, name, _fmt_seconds(time.perf_counter() - t0))
    return out


def _distributed_barrier(tag: str) -> None:
    if not DIST.enabled:
        return
    if PROFILE_COMM:
        t0 = time.perf_counter()
        dist.barrier()
        device_synchronize()
        LOG.info("rank%d barrier | %s | elapsed=%s", DIST.rank, tag, _fmt_seconds(time.perf_counter() - t0))
    else:
        dist.barrier()


def _resolve_fsdp_mixed_precision(dtype: torch.dtype):
    if not HAS_TORCH_FSDP or MixedPrecision is None:
        return None
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def _module_dtype_summary(module: torch.nn.Module) -> dict[str, int]:
    stats: Counter[str] = Counter()
    for p in module.parameters():
        stats[str(p.dtype)] += p.numel()
    return dict(stats)


def _first_mixed_dtype_params(module: torch.nn.Module, target_dtype: torch.dtype, limit: int = 32) -> list[tuple[str, str, tuple[int, ...]]]:
    bad = []
    for name, param in module.named_parameters(recurse=True):
        if torch.is_floating_point(param) and param.dtype != target_dtype:
            bad.append((name, str(param.dtype), tuple(param.shape)))
            if len(bad) >= limit:
                break
    return bad


def cast_module_floating_tensors_(module: torch.nn.Module, dtype: torch.dtype) -> None:
    for param in module.parameters(recurse=True):
        if torch.is_floating_point(param) and param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None and torch.is_floating_point(param.grad):
                param.grad.data = param.grad.data.to(dtype=dtype)

    for submodule in module.modules():
        for buf_name, buf in list(submodule._buffers.items()):
            if isinstance(buf, torch.Tensor) and torch.is_floating_point(buf) and buf.dtype != dtype:
                submodule._buffers[buf_name] = buf.to(dtype=dtype)


def maybe_wrap_fsdp(module: torch.nn.Module, module_name: str, dtype: torch.dtype) -> torch.nn.Module:
    if not ENABLE_FSDP:
        return module
    if not DIST.enabled or DIST.world_size <= 1:
        LOG.warning("FSDP requested for %s but distributed world_size <= 1; skip wrap", module_name)
        return module
    if not HAS_TORCH_FSDP or FSDP is None or ShardingStrategy is None:
        raise RuntimeError(
            "LTX_ENABLE_FSDP=1 but torch.distributed.fsdp is unavailable in this runtime. "
            "Please verify your torch/torch_npu stack supports FSDP on the current backend."
        )

    if isinstance(module, FSDP):
        return module

    mp_policy = _resolve_fsdp_mixed_precision(dtype)
    wrapped = FSDP(
        module,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=FSDP_USE_ORIG_PARAMS,
        sync_module_states=FSDP_SYNC_MODULE_STATES,
        forward_prefetch=FSDP_FORWARD_PREFETCH,
        device_id=None,
    )
    LOG.info(
        "rank%d fsdp | wrapped %s | sharding=%s dtype=%s use_orig_params=%s sync_module_states=%s forward_prefetch=%s",
        DIST.rank,
        module_name,
        "FULL_SHARD",
        dtype,
        FSDP_USE_ORIG_PARAMS,
        FSDP_SYNC_MODULE_STATES,
        FSDP_FORWARD_PREFETCH,
    )
    return wrapped


class DistCtx:
    def __init__(self) -> None:
        self.initialized = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")
        self.group = None

    @property
    def enabled(self) -> bool:
        return self.initialized and self.world_size > 1


DIST = DistCtx()


def init_dist_if_needed() -> DistCtx:
    if DIST.initialized:
        return DIST

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    visible = _visible_device_count()
    if visible <= 0 and (hasattr(torch, "npu") or torch.cuda.is_available()):
        raise RuntimeError(
            "No visible accelerator devices found. "
            f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    if visible > 0 and local_rank >= visible:
        raise RuntimeError(
            f"local_rank={local_rank} but only {visible} devices are visible. "
            f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    if hasattr(torch, "npu"):
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device()

    if world_size > 1 and not dist.is_initialized():
        backend = "hccl" if device.type == "npu" else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")

    DIST.initialized = True
    DIST.rank = rank
    DIST.local_rank = local_rank
    DIST.world_size = world_size if dist.is_initialized() else 1
    DIST.device = device
    DIST.group = dist.group.WORLD if dist.is_initialized() else None

    LOG.info(
        "dist initialized | rank=%d local_rank=%d world_size=%d device=%s visible_devices=%d",
        DIST.rank,
        DIST.local_rank,
        DIST.world_size,
        DIST.device,
        visible,
    )
    return DIST


def get_runtime_device() -> torch.device:
    ctx = init_dist_if_needed()
    return ctx.device if ctx.initialized else get_device()


def device_synchronize() -> None:
    device = DIST.device if DIST.initialized else get_runtime_device()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# sequence sharding / Ulysses helpers
# -----------------------------------------------------------------------------


def _require_divisible(n: int, d: int, what: str) -> None:
    if d <= 0 or n % d != 0:
        raise RuntimeError(f"{what}={n} is not divisible by world_size={d}")


def _get_local_slice(seq_len: int) -> tuple[int, int, int]:
    _require_divisible(seq_len, DIST.world_size, "sequence length")
    chunk = seq_len // DIST.world_size
    start = DIST.rank * chunk
    end = start + chunk
    return start, end, chunk


def _slice_dim(x: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    index = [slice(None)] * x.ndim
    index[dim] = slice(start, end)
    return x[tuple(index)].contiguous()


def _find_seq_dim(x: torch.Tensor, seq_len: int) -> int | None:
    if x.ndim >= 2 and x.shape[1] == seq_len:
        return 1
    for d in range(min(x.ndim, 3)):
        if x.shape[d] == seq_len:
            return d
    return None


def _shard_seq_tensor(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if not DIST.enabled:
        return x
    dim = _find_seq_dim(x, seq_len)
    if dim is None:
        return x
    start, end, _ = _get_local_slice(seq_len)
    return _slice_dim(x, dim, start, end)


def _maybe_shard_field(value: Any, seq_len: int) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _shard_seq_tensor(value, seq_len)
    if isinstance(value, tuple):
        return tuple(_maybe_shard_field(v, seq_len) for v in value)
    if isinstance(value, list):
        return [_maybe_shard_field(v, seq_len) for v in value]
    return value


def _gather_seq_dim1(x: torch.Tensor) -> torch.Tensor:
    if not DIST.enabled:
        return x
    gathered = [torch.empty_like(x) for _ in range(DIST.world_size)]
    dist.all_gather(gathered, x.contiguous(), group=DIST.group)
    return torch.cat(gathered, dim=1).contiguous()


def _ulysses_sp_to_hp(x_bsnd: torch.Tensor, heads: int) -> tuple[torch.Tensor, int]:
    if not DIST.enabled:
        return x_bsnd, heads

    _require_divisible(heads, DIST.world_size, "num_heads")
    h_local = heads // DIST.world_size
    b, s_local, _, d = x_bsnd.shape

    send = x_bsnd.view(b, s_local, DIST.world_size, h_local, d).permute(2, 0, 1, 3, 4).contiguous()
    send_list = [t.contiguous() for t in send.unbind(0)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(DIST.world_size)]
    _timed_collective("ulysses.all_to_all", dist.all_to_all, recv_list, send_list, group=DIST.group)

    out = torch.cat(recv_list, dim=1).contiguous()
    return out, h_local


def _ulysses_hp_to_sp(x_bsnd: torch.Tensor) -> torch.Tensor:
    if not DIST.enabled:
        return x_bsnd

    _, s_global, _, _ = x_bsnd.shape
    _require_divisible(s_global, DIST.world_size, "global sequence length")

    send_list = [t.contiguous() for t in x_bsnd.chunk(DIST.world_size, dim=1)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(DIST.world_size)]
    _timed_collective("ulysses.all_to_all", dist.all_to_all, recv_list, send_list, group=DIST.group)

    return torch.cat(recv_list, dim=2).contiguous()


# -----------------------------------------------------------------------------
# monkey patches from reference launcher
# -----------------------------------------------------------------------------


def patch_basic_block() -> None:
    tf_mod = importlib.import_module("ltx_core.model.transformer.transformer")
    BasicAVTransformerBlock = tf_mod.BasicAVTransformerBlock
    if getattr(BasicAVTransformerBlock, "_ulysses_patched", False):
        return

    orig_init = BasicAVTransformerBlock.__init__

    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if hasattr(self, "attn1"):
            self.attn1._ulysses_enabled = _env_flag("LTX_ULYSSES_PATCH_VIDEO_SELF_ATTN", "1")
            self.attn1._ulysses_name = "attn1"
        if hasattr(self, "audio_attn1"):
            self.audio_attn1._ulysses_enabled = _env_flag("LTX_ULYSSES_PATCH_AUDIO_SELF_ATTN", "0")
            self.audio_attn1._ulysses_name = "audio_attn1"

    BasicAVTransformerBlock.__init__ = wrapped_init
    BasicAVTransformerBlock._ulysses_patched = True


def patch_attention_forward() -> None:
    attn_mod = importlib.import_module("ltx_core.model.transformer.attention")
    Attention = attn_mod.Attention
    if getattr(Attention, "_ulysses_patched", False):
        return

    orig_forward = Attention.forward

    def wrapped_forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        use_ulysses = (
            DIST.enabled
            and getattr(self, "_ulysses_enabled", False)
            and context is None
            and not all_perturbed
        )
        if not use_ulysses:
            return orig_forward(
                self,
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                perturbation_mask=perturbation_mask,
                all_perturbed=all_perturbed,
            )

        context = x
        v = self.to_v(context)
        q = self.to_q(x)
        k = self.to_k(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            if attn_mod._should_use_mindie_rope(q, self.rope_type):
                try:
                    q = attn_mod._mindie_rope(q, pe, self.rope_type, self.heads)
                    k = attn_mod._mindie_rope(k, pe if k_pe is None else k_pe, self.rope_type, self.heads)
                except Exception:
                    q = attn_mod.apply_rotary_emb(q, pe, self.rope_type)
                    k = attn_mod.apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
            else:
                q = attn_mod.apply_rotary_emb(q, pe, self.rope_type)
                k = attn_mod.apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        b, s_local, inner_dim = q.shape
        dim_head = inner_dim // self.heads

        q_bsnd = q.view(b, s_local, self.heads, dim_head)
        k_bsnd = k.view(b, s_local, self.heads, dim_head)
        v_bsnd = v.view(b, s_local, self.heads, dim_head)

        q_hp, heads_local = _ulysses_sp_to_hp(q_bsnd, self.heads)
        k_hp, _ = _ulysses_sp_to_hp(k_bsnd, self.heads)
        v_hp, _ = _ulysses_sp_to_hp(v_bsnd, self.heads)

        q_flat = q_hp.reshape(b, q_hp.shape[1], heads_local * dim_head)
        k_flat = k_hp.reshape(b, q_hp.shape[1], heads_local * dim_head)
        v_flat = v_hp.reshape(b, q_hp.shape[1], heads_local * dim_head)

        if attn_mod._should_use_mindie(q_flat):
            out_flat = attn_mod._mindie_attention(q_flat, k_flat, v_flat, heads_local, mask)
        else:
            out_flat = self.attention_function(q_flat, k_flat, v_flat, heads_local, mask)

        out_hp = out_flat.view(b, q_hp.shape[1], heads_local, dim_head)
        out_sp = _ulysses_hp_to_sp(out_hp)
        out = out_sp.reshape(b, s_local, self.heads * dim_head)

        if perturbation_mask is not None:
            out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            out = out.view(b, s_local, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, s_local, self.heads * self.dim_head)

        return self.to_out(out)

    Attention.forward = wrapped_forward
    Attention._ulysses_patched = True


def _import_ltx_model_module():
    for mod_name in (
        "ltx_core.model.transformer.model",
        "ltx_core.model.transformer.x0_model",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if hasattr(mod, "LTXModel"):
            return mod
    raise ModuleNotFoundError(
        "Cannot find LTXModel. Tried ltx_core.model.transformer.model and ltx_core.model.transformer.x0_model"
    )


def patch_ltxmodel_forward() -> None:
    ltx_mod = _import_ltx_model_module()
    LTXModel = ltx_mod.LTXModel
    if getattr(LTXModel, "_ulysses_patched", False):
        return

    def shard_transformer_args(args):
        if args is None or not DIST.enabled:
            return args
        seq_len = args.x.shape[1]
        _require_divisible(seq_len, DIST.world_size, "TransformerArgs.x")
        return replace(
            args,
            x=_maybe_shard_field(args.x, seq_len),
            timesteps=_maybe_shard_field(args.timesteps, seq_len),
            embedded_timestep=_maybe_shard_field(args.embedded_timestep, seq_len),
            positional_embeddings=_maybe_shard_field(args.positional_embeddings, seq_len),
            cross_positional_embeddings=_maybe_shard_field(args.cross_positional_embeddings, seq_len),
            cross_scale_shift_timestep=_maybe_shard_field(args.cross_scale_shift_timestep, seq_len),
            cross_gate_timestep=_maybe_shard_field(args.cross_gate_timestep, seq_len),
            prompt_timestep=_maybe_shard_field(args.prompt_timestep, seq_len),
            self_attention_mask=args.self_attention_mask,
            context=args.context,
            context_mask=args.context_mask,
        )

    def wrapped_forward(self, video, audio, perturbations):
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        patch_video = _env_flag("LTX_ULYSSES_PATCH_VIDEO_SELF_ATTN", "1")
        patch_audio = _env_flag("LTX_ULYSSES_PATCH_AUDIO_SELF_ATTN", "0")

        video_args = shard_transformer_args(video_args) if patch_video else video_args
        audio_args = shard_transformer_args(audio_args) if patch_audio else audio_args

        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )

        if vx is not None and DIST.enabled and patch_video:
            vx = _gather_seq_dim1(vx)
        if ax is not None and DIST.enabled and patch_audio:
            ax = _gather_seq_dim1(ax)
        return vx, ax

    LTXModel.forward = wrapped_forward
    LTXModel._ulysses_patched = True


def install_all_patches() -> None:
    init_dist_if_needed()
    patch_basic_block()
    patch_attention_forward()
    patch_ltxmodel_forward()


# -----------------------------------------------------------------------------
# cli compatibility helpers
# -----------------------------------------------------------------------------


def _consume_cli_value(argv: list[str], i: int) -> tuple[str, int]:
    if i + 1 >= len(argv):
        raise SystemExit(f"missing value for {argv[i]}")
    return argv[i + 1], i + 2


def normalize_cli_args(argv: list[str]) -> tuple[list[str], int]:
    out: list[str] = []
    warmup_runs: int | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in {"--checkpoint-path", "--checkpoint_path"}:
            value, i = _consume_cli_value(argv, i)
            out.extend(["--distilled-checkpoint-path", value])
            continue
        if arg.startswith("--checkpoint-path=") or arg.startswith("--checkpoint_path="):
            out.append("--distilled-checkpoint-path=" + arg.split("=", 1)[1])
            i += 1
            continue

        if arg in {"--distilled-lora", "--distilled_lora"}:
            out.append("--lora")
            i += 1
            continue
        if arg.startswith("--distilled-lora=") or arg.startswith("--distilled_lora="):
            out.append("--lora=" + arg.split("=", 1)[1])
            i += 1
            continue

        if arg in {"--warmup-runs", "--warmup_runs"}:
            value, i = _consume_cli_value(argv, i)
            warmup_runs = int(value)
            continue
        if arg.startswith("--warmup-runs=") or arg.startswith("--warmup_runs="):
            warmup_runs = int(arg.split("=", 1)[1])
            i += 1
            continue

        out.append(arg)
        i += 1

    if warmup_runs is None:
        warmup_runs = int(os.getenv("LTX_WARMUP_RUNS", "0"))
    if warmup_runs < 0:
        raise SystemExit("warmup-runs must be >= 0")
    return out, warmup_runs


# -----------------------------------------------------------------------------
# distilled pipeline
# -----------------------------------------------------------------------------


class DistilledPipeline:
    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps] | tuple[LoraPathStrengthAndSDOps, ...],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
    ):
        init_dist_if_needed()
        install_all_patches()

        self.device = device or get_runtime_device()
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=distilled_checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

        self._video_encoder = None
        self._transformer = None
        self._video_decoder = None
        self._audio_decoder = None
        self._vocoder = None

    def _build_denoise_modules(self):
        if self._video_encoder is None:
            self._video_encoder = self.model_ledger.video_encoder()
        if self._transformer is None:
            self._transformer = self.model_ledger.transformer()
            if ENABLE_FSDP:
                before_stats = _module_dtype_summary(self._transformer)
                LOG.info("rank%d fsdp | transformer dtype summary before normalize: %s", DIST.rank, before_stats)
                if FSDP_FORCE_UNIFORM_DTYPE:
                    cast_module_floating_tensors_(self._transformer, self.dtype)
                    after_stats = _module_dtype_summary(self._transformer)
                    LOG.info("rank%d fsdp | transformer dtype summary after normalize: %s", DIST.rank, after_stats)
                    offenders = _first_mixed_dtype_params(self._transformer, self.dtype)
                    if offenders:
                        raise RuntimeError(
                            f"transformer still has mixed floating dtypes after normalize; target={self.dtype}, "
                            f"summary={after_stats}, offenders={offenders}"
                        )
            self._transformer = maybe_wrap_fsdp(
                self._transformer,
                module_name="transformer",
                dtype=self.dtype,
            )
        return self._video_encoder, self._transformer

    def _release_denoise_modules(self) -> None:
        self._transformer = None
        self._video_encoder = None
        cleanup_memory()

    def _build_decode_modules(self):
        if self._video_decoder is None:
            self._video_decoder = self.model_ledger.video_decoder()
        if self._audio_decoder is None:
            self._audio_decoder = self.model_ledger.audio_decoder()
        if self._vocoder is None:
            self._vocoder = self.model_ledger.vocoder()
        return self._video_decoder, self._audio_decoder, self._vocoder

    def _release_decode_modules(self) -> None:
        self._video_decoder = None
        self._audio_decoder = None
        self._vocoder = None
        cleanup_memory()

    def _decode_rank0(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        tiling_config: TilingConfig | None,
        generator: torch.Generator,
        call_t0: float,
    ) -> tuple[Iterator[torch.Tensor], Audio | None]:
        video_decoder, audio_decoder, vocoder = self._build_decode_modules()

        _log_rank0("decode start")
        _log_memory("before decode")
        decode_t0 = time.perf_counter()

        decoded_video = vae_decode_video(
            video_latent,
            video_decoder,
            tiling_config,
            generator,
        )
        decoded_audio = vae_decode_audio(
            audio_latent,
            audio_decoder,
            vocoder,
        )
        device_synchronize()
        _log_memory("after decode")
        _log_rank0(
            "decode done: video=%s audio=%s elapsed=%s total=%s",
            _shape_str(decoded_video),
            _shape_str(decoded_audio),
            _fmt_seconds(time.perf_counter() - decode_t0),
            _fmt_seconds(time.perf_counter() - call_t0),
        )
        self._release_decode_modules()
        return decoded_video, decoded_audio

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
    ) -> tuple[Iterator[torch.Tensor] | None, Audio | None]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        call_t0 = time.perf_counter()
        _reset_peak_memory_stats()
        _log_memory("pipeline-start")
        _log_rank0(
            "pipeline start: prompt_len=%d seed=%d size=%dx%d frames=%d fps=%s images=%d enhance_prompt=%s",
            len(prompt),
            seed,
            height,
            width,
            num_frames,
            frame_rate,
            len(images),
            enhance_prompt,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        encode_t0 = time.perf_counter()
        (ctx_p,) = encode_prompts(
            [prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        _log_rank0(
            "prompt encoded: video_context=%s audio_context=%s elapsed=%s",
            _shape_str(video_context),
            _shape_str(audio_context),
            _fmt_seconds(time.perf_counter() - encode_t0),
        )

        video_encoder, transformer = self._build_denoise_modules()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        _log_rank0(
            "stage1 start: output=%dx%d frames=%d sigma_steps=%d",
            stage_1_output_shape.width,
            stage_1_output_shape.height,
            stage_1_output_shape.frames,
            int(stage_1_sigmas.numel()),
        )
        stage1_t0 = time.perf_counter()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )
        device_synchronize()
        _log_memory("after-stage1")
        _log_rank0(
            "stage1 done: video_latent=%s audio_latent=%s elapsed=%s",
            _shape_str(video_state.latent),
            _shape_str(audio_state.latent),
            _fmt_seconds(time.perf_counter() - stage1_t0),
        )

        _log_rank0("upsample start: input_video_latent=%s", _shape_str(video_state.latent[:1]))
        upsample_t0 = time.perf_counter()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler(),
        )
        device_synchronize()
        _log_rank0(
            "upsample done: upscaled_video_latent=%s elapsed=%s",
            _shape_str(upscaled_video_latent),
            _fmt_seconds(time.perf_counter() - upsample_t0),
        )
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        _log_rank0(
            "stage2 start: output=%dx%d frames=%d sigma_steps=%d noise_scale=%s",
            stage_2_output_shape.width,
            stage_2_output_shape.height,
            stage_2_output_shape.frames,
            int(stage_2_sigmas.numel()),
            float(stage_2_sigmas[0].item()),
        )
        stage2_t0 = time.perf_counter()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        device_synchronize()
        _log_memory("after-stage2")
        _log_rank0(
            "stage2 done: video_latent=%s audio_latent=%s elapsed=%s",
            _shape_str(video_state.latent),
            _shape_str(audio_state.latent),
            _fmt_seconds(time.perf_counter() - stage2_t0),
        )
        self._release_denoise_modules()

        should_decode = (not DIST.enabled) or DIST.rank == 0
        if DIST.enabled:
            LOG.info("rank%d waiting at post-denoise barrier before decode", DIST.rank)
            _distributed_barrier("post-denoise-before-decode")

        if not should_decode:
            LOG.info("rank%d skip decode, rank0 will decode and save final output", DIST.rank)
            return None, None

        return self._decode_rank0(
            video_latent=video_state.latent,
            audio_latent=audio_state.latent,
            tiling_config=tiling_config,
            generator=generator,
            call_t0=call_t0,
        )


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    init_dist_if_needed()
    install_all_patches()

    raw_argv = sys.argv[1:]
    normalized_argv, warmup_runs = normalize_cli_args(raw_argv)
    sys.argv = [sys.argv[0], *normalized_argv]

    LOG.info(
        "rank=%d local_rank=%d world_size=%d device=%s video_self_attn=%s audio_self_attn=%s warmup_runs=%d profile=%s",
        DIST.rank,
        DIST.local_rank,
        DIST.world_size,
        DIST.device,
        _env_flag("LTX_ULYSSES_PATCH_VIDEO_SELF_ATTN", "1"),
        _env_flag("LTX_ULYSSES_PATCH_AUDIO_SELF_ATTN", "0"),
        warmup_runs,
        _env_flag("LTX_ENABLE_PROFILE", "0"),
    )
    _log_rank0("argv normalized: raw=%s normalized=%s", raw_argv, normalized_argv)

    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    args = parser.parse_args()

    _log_rank0(
        "args parsed: checkpoint=%s upsampler=%s gemma_root=%s output=%s height=%s width=%s frames=%s fps=%s seed=%s lora_count=%d",
        args.distilled_checkpoint_path,
        args.spatial_upsampler_path,
        args.gemma_root,
        args.output_path,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
        args.seed,
        len(args.lora) if args.lora else 0,
    )

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=get_runtime_device(),
        quantization=args.quantization,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    _log_rank0("tiling ready: config=%s video_chunks_number=%s", tiling_config, video_chunks_number)
    _reset_peak_memory_stats()
    _log_memory("main-before-warmup")

    for warmup_idx in range(warmup_runs):
        warmup_seed = args.seed + warmup_idx + 1
        _log_rank0("warmup %d/%d start: seed=%d (profile disabled)", warmup_idx + 1, warmup_runs, warmup_seed)
        warmup_t0 = time.perf_counter()
        _video, _audio = pipeline(
            prompt=args.prompt,
            seed=warmup_seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )
        del _video, _audio
        cleanup_memory()
        if DIST.enabled:
            LOG.info("rank%d waiting at post-warmup barrier %d/%d", DIST.rank, warmup_idx + 1, warmup_runs)
            dist.barrier()
        _log_rank0("warmup %d/%d done: elapsed=%s", warmup_idx + 1, warmup_runs, _fmt_seconds(time.perf_counter() - warmup_t0))

    _reset_peak_memory_stats()
    _log_memory("main-before-formal")
    infer_t0 = time.perf_counter()
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    if DIST.enabled:
        LOG.info("rank%d waiting at pre-save barrier", DIST.rank)
        dist.barrier()

    _log_rank0("inference finished, entering save step: elapsed=%s", _fmt_seconds(time.perf_counter() - infer_t0))

    if DIST.rank == 0 and video is not None:
        save_t0 = time.perf_counter()
        _log_memory("before-save")
        _log_rank0("save start: output=%s chunks=%s", args.output_path, video_chunks_number)
        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=video_chunks_number,
        )
        _log_memory("after-save")
        _log_rank0("save done: output=%s elapsed=%s", args.output_path, _fmt_seconds(time.perf_counter() - save_t0))
    else:
        LOG.info("rank%d skips encode_video(), rank0 owns final write", DIST.rank)


if __name__ == "__main__":
    main()




