#!/usr/bin/env python3
from __future__ import annotations

"""
Ascend NPU / HCCL Ulysses launcher for LTX-2 two-stage inference.

Usage:
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTHONPATH=/path/to/LTX-2/packages/ltx-pipelines/src:/path/to/LTX-2/packages/ltx-core/src:$PYTHONPATH
  torchrun --nproc_per_node=8 --master_port=29501 /path/to/ltx_ulysses_hccl_launcher_profile_warmup1.py \
    --checkpoint-path ... \
    --distilled-lora ... 0.8 \
    --spatial-upsampler-path ... \
    --gemma-root ... \
    --prompt "..." \
    --output-path output.mp4

What this launcher does:
  1) Initialize HCCL when started under torchrun.
  2) Monkey-patch BasicAVTransformerBlock so attn1 (and optionally audio_attn1)
     are marked for Ulysses self-attention.
  3) Monkey-patch Attention.forward so self-attention uses Ulysses all-to-all.
  4) Monkey-patch LTXModel.forward so TransformerArgs are sequence-sharded before
     transformer blocks and gathered back after _process_output().
  5) Reuse ti2vid_two_stages.py's own profiler mechanism, but allow overriding
     Ascend profiler defaults from environment variables.
  6) Call the original ltx_pipelines.ti2vid_two_stages.main().

Notes:
  - This version assumes the sequence length is divisible by world_size.
  - Video self-attention Ulysses is enabled by default.
  - Audio self-attention Ulysses is disabled by default; enable with:
        export LTX_ULYSSES_PATCH_AUDIO_SELF_ATTN=1
  - Cross-attention paths remain unchanged.
  - Profiling is enabled only when:
        export LTX_ENABLE_PROFILE=1
    and warmup defaults to 1 unless overridden by:
        export LTX_PROFILE_WARMUP=1
"""

import importlib
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

LOG = logging.getLogger("ltx_ulysses")


# -----------------------------------------------------------------------------
# env / distributed helpers
# -----------------------------------------------------------------------------


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


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

    if hasattr(torch, "npu"):
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "hccl" if device.type == "npu" else "nccl"
        dist.init_process_group(backend=backend, init_method="env://")

    DIST.initialized = True
    DIST.rank = rank
    DIST.local_rank = local_rank
    DIST.world_size = world_size if dist.is_initialized() else 1
    DIST.device = device
    DIST.group = dist.group.WORLD if dist.is_initialized() else None
    return DIST


# -----------------------------------------------------------------------------
# sequence sharding helpers
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
    # Prefer dim=1 for [B,S,D], otherwise search first three dims.
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


# -----------------------------------------------------------------------------
# Ulysses A2A helpers
# -----------------------------------------------------------------------------


def _ulysses_sp_to_hp(x_bsnd: torch.Tensor, heads: int) -> tuple[torch.Tensor, int]:
    """
    seq-parallel -> head-parallel
    input : [B, S_local, H, D]
    output: [B, S_global, H_local, D]
    """
    if not DIST.enabled:
        return x_bsnd, heads

    _require_divisible(heads, DIST.world_size, "num_heads")
    h_local = heads // DIST.world_size
    b, s_local, _, d = x_bsnd.shape

    # [B, S_local, H, D] -> [P, B, S_local, H_local, D]
    send = x_bsnd.view(b, s_local, DIST.world_size, h_local, d).permute(2, 0, 1, 3, 4).contiguous()
    send_list = [t.contiguous() for t in send.unbind(0)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(DIST.world_size)]
    dist.all_to_all(recv_list, send_list, group=DIST.group)

    # Concatenate along sequence: [B, S_global, H_local, D]
    out = torch.cat(recv_list, dim=1).contiguous()
    return out, h_local


def _ulysses_hp_to_sp(x_bsnd: torch.Tensor) -> torch.Tensor:
    """
    head-parallel -> seq-parallel
    input : [B, S_global, H_local, D]
    output: [B, S_local, H, D]
    """
    if not DIST.enabled:
        return x_bsnd

    b, s_global, h_local, d = x_bsnd.shape
    _require_divisible(s_global, DIST.world_size, "global sequence length")

    send_list = [t.contiguous() for t in x_bsnd.chunk(DIST.world_size, dim=1)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(DIST.world_size)]
    dist.all_to_all(recv_list, send_list, group=DIST.group)

    return torch.cat(recv_list, dim=2).contiguous()  # [B, S_local, H, D]


# -----------------------------------------------------------------------------
# monkey patches
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
        # Only patch self-attention. Cross-attention remains original.
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
        k_flat = k_hp.reshape(b, k_hp.shape[1], heads_local * dim_head)
        v_flat = v_hp.reshape(b, v_hp.shape[1], heads_local * dim_head)

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
            # Keep full self_attention_mask on all ranks: Attention sees global seq after Ulysses A2A.
            self_attention_mask=args.self_attention_mask,
            # Keep context/context_mask replicated for now.
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


# -----------------------------------------------------------------------------
# runtime patching around original two-stage script
# -----------------------------------------------------------------------------


def _infer_output_path(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg in {"--output-path", "--output_path"} and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--output-path="):
            return arg.split("=", 1)[1]
        if arg.startswith("--output_path="):
            return arg.split("=", 1)[1]
    return None


def _has_flag(argv: list[str], *flags: str) -> bool:
    return any(a == f or a.startswith(f + "=") for a in argv for f in flags)


def _inject_rank_log_file(argv: list[str]) -> list[str]:
    if _has_flag(argv, "--log-file", "--log_file"):
        return argv
    output_path = _infer_output_path(argv) or "ltx_output.mp4"
    stem = Path(output_path).expanduser().resolve().stem or "ltx_output"
    log_path = str((Path.cwd() / f"{stem}.rank{DIST.rank}.log").resolve())
    return argv + ["--log-file", log_path]


def _patch_two_stage_runtime(two_stage_mod) -> None:
    # Only rank0 writes final video file.
    if DIST.rank != 0 and hasattr(two_stage_mod, "encode_video"):
        orig_encode_video = two_stage_mod.encode_video

        def silent_encode_video(*args, **kwargs):
            LOG.info("rank%d skips encode_video()", DIST.rank)
            return None

        two_stage_mod.encode_video = silent_encode_video
        two_stage_mod._orig_encode_video = orig_encode_video

    # Give each rank a distinct default log path.
    if hasattr(two_stage_mod, "default_current_dir_sidecar_path"):
        orig_sidecar = two_stage_mod.default_current_dir_sidecar_path

        def rank_sidecar(output_path: str, suffix: str):
            base = orig_sidecar(output_path, suffix)
            if DIST.world_size <= 1:
                return base
            return base.with_name(f"{base.stem}.rank{DIST.rank}{base.suffix}")

        two_stage_mod.default_current_dir_sidecar_path = rank_sidecar


def install_all_patches() -> None:
    init_dist_if_needed()
    patch_basic_block()
    patch_attention_forward()
    patch_ltxmodel_forward()


def patch_two_stage_profiler_defaults() -> None:
    """
    Reuse ti2vid_two_stages.py's own Ascend profiler mechanism,
    but allow overriding schedule defaults from env.
    """
    two_stage_mod = importlib.import_module("ltx_pipelines.ti2vid_two_stages")

    defaults = getattr(two_stage_mod, "ASCEND_PROFILE_DEFAULTS", None)
    if not isinstance(defaults, dict):
        LOG.warning("ASCEND_PROFILE_DEFAULTS not found in ti2vid_two_stages")
        return

    def _env_int(name: str, fallback: int) -> int:
        v = os.getenv(name)
        return fallback if v is None or v == "" else int(v)

    def _env_bool(name: str, fallback: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return fallback
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    defaults["wait"] = _env_int("LTX_PROFILE_WAIT", defaults.get("wait", 0))
    defaults["warmup"] = _env_int("LTX_PROFILE_WARMUP", 1)
    defaults["active"] = _env_int("LTX_PROFILE_ACTIVE", defaults.get("active", 10))
    defaults["repeat"] = _env_int("LTX_PROFILE_REPEAT", defaults.get("repeat", 1))
    defaults["skip_first"] = _env_int("LTX_PROFILE_SKIP_FIRST", defaults.get("skip_first", 0))

    defaults["record_shapes"] = _env_bool("LTX_PROFILE_RECORD_SHAPES", defaults.get("record_shapes", False))
    defaults["memory"] = _env_bool("LTX_PROFILE_MEMORY", defaults.get("memory", False))
    defaults["with_stack"] = _env_bool("LTX_PROFILE_WITH_STACK", defaults.get("with_stack", False))
    defaults["with_modules"] = _env_bool("LTX_PROFILE_WITH_MODULES", defaults.get("with_modules", False))
    defaults["with_flops"] = _env_bool("LTX_PROFILE_WITH_FLOPS", defaults.get("with_flops", False))
    defaults["l2_cache"] = _env_bool("LTX_PROFILE_L2_CACHE", defaults.get("l2_cache", False))
    defaults["export_db"] = _env_bool("LTX_PROFILE_EXPORT_DB", defaults.get("export_db", False))
    defaults["analyse"] = _env_bool("LTX_PROFILE_ANALYSE", defaults.get("analyse", True))

    if os.getenv("LTX_PROFILE_LEVEL"):
        defaults["level"] = os.getenv("LTX_PROFILE_LEVEL")
    if os.getenv("LTX_PROFILE_AIC_METRICS"):
        defaults["aic_metrics"] = os.getenv("LTX_PROFILE_AIC_METRICS")

    LOG.info(
        "patched ASCEND_PROFILE_DEFAULTS: wait=%s warmup=%s active=%s repeat=%s skip_first=%s",
        defaults["wait"], defaults["warmup"], defaults["active"], defaults["repeat"], defaults["skip_first"],
    )


# -----------------------------------------------------------------------------
# launcher main
# -----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    init_dist_if_needed()
    install_all_patches()
    patch_two_stage_profiler_defaults()

    # Prevent default log collisions when caller didn't pass --log-file.
    sys.argv = [sys.argv[0], *_inject_rank_log_file(sys.argv[1:])]

    two_stage_mod = importlib.import_module("ltx_pipelines.ti2vid_two_stages")
    _patch_two_stage_runtime(two_stage_mod)

    LOG.info(
        "Ulysses launcher ready: rank=%d local_rank=%d world_size=%d device=%s "
        "video_self_attn=%s audio_self_attn=%s profile=%s warmup=%s",
        DIST.rank,
        DIST.local_rank,
        DIST.world_size,
        DIST.device,
        _env_flag("LTX_ULYSSES_PATCH_VIDEO_SELF_ATTN", "1"),
        _env_flag("LTX_ULYSSES_PATCH_AUDIO_SELF_ATTN", "0"),
        _env_flag("LTX_ENABLE_PROFILE", "0"),
        os.getenv("LTX_PROFILE_WARMUP", "1"),
    )

    if not hasattr(two_stage_mod, "main"):
        raise RuntimeError("ltx_pipelines.ti2vid_two_stages.main() not found")
    two_stage_mod.main()


if __name__ == "__main__":
    main()