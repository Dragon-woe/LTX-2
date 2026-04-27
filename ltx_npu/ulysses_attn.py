"""Ulysses Sequence Parallelism: AllToAll-based attention + sequence split.

Optimized version:
- env-controlled all_to_all_single fast path
- fused QKV all_to_all option to reduce 3 collectives to 1
- HCCL-compatible fallback to list-based all_to_all
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_USE_A2A_SINGLE = os.getenv("LTX_UA2A_USE_A2A_SINGLE", "1") == "1"
_FUSE_QKV_A2A = os.getenv("LTX_UA2A_FUSE_QKV", "1") == "1"


def _a2a_list_fallback(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input
    shape = list(input.shape)
    assert shape[scatter_idx] % world_size == 0, (
        f"dim {scatter_idx} size {shape[scatter_idx]} not divisible by world_size {world_size}"
    )
    input_chunks = [c.contiguous() for c in input.chunk(world_size, dim=scatter_idx)]
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks, group=group)
    return torch.cat(output_chunks, dim=gather_idx)


def all_to_all_4d(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """AllToAll on a 4D tensor: scatter on one dim, gather on another."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input

    shape = list(input.shape)
    assert len(shape) == 4, f"expected 4D tensor, got {tuple(shape)}"
    assert shape[scatter_idx] % world_size == 0, (
        f"dim {scatter_idx} size {shape[scatter_idx]} not divisible by world_size {world_size}"
    )

    if not _USE_A2A_SINGLE:
        return _a2a_list_fallback(input, scatter_idx, gather_idx, group)

    try:
        remain = [i for i in range(4) if i not in (scatter_idx, gather_idx)]
        perm = [scatter_idx, gather_idx, *remain]
        x = input.permute(perm).contiguous()  # [S, G, R0, R1]
        s_dim, g_dim, r0, r1 = x.shape
        s_local = s_dim // world_size

        send = x.view(world_size, s_local, g_dim, r0, r1).reshape(world_size, -1).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)

        y = recv.view(world_size, s_local, g_dim, r0, r1)
        y = y.permute(1, 0, 2, 3, 4).contiguous().reshape(s_local, world_size * g_dim, r0, r1)

        inv = [0] * 4
        for i, a in enumerate(perm):
            inv[a] = i
        return y.permute(inv).contiguous()
    except Exception:
        return _a2a_list_fallback(input, scatter_idx, gather_idx, group)


class UlyssesAttention:
    """Wraps attention with pre/post AllToAll for correct full-sequence attention."""

    def __init__(self, original_fn, sp_group: dist.ProcessGroup):
        self.original_fn = original_fn
        self.sp_group = sp_group
        self.world_size = dist.get_world_size(sp_group)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.world_size <= 1:
            return self.original_fn(q, k, v, heads, mask)

        b, local_t, hd = q.shape
        dim_head = hd // heads
        q4 = q.view(b, local_t, heads, dim_head)
        k4 = k.view(b, local_t, heads, dim_head)
        v4 = v.view(b, local_t, heads, dim_head)

        # AllToAll: scatter head(2), gather seq(1)
        # Each rank: (B, local_T, H, D) -> (B, global_T, H/N, D)
        if _FUSE_QKV_A2A:
            qkv4 = torch.cat([q4, k4, v4], dim=0)
            qkv4 = all_to_all_4d(qkv4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            q4, k4, v4 = torch.chunk(qkv4, 3, dim=0)
        else:
            q4 = all_to_all_4d(q4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            k4 = all_to_all_4d(k4, scatter_idx=2, gather_idx=1, group=self.sp_group)
            v4 = all_to_all_4d(v4, scatter_idx=2, gather_idx=1, group=self.sp_group)

        local_heads = heads // self.world_size
        gathered_t = q4.shape[1]

        q_flat = q4.reshape(b, gathered_t, local_heads * dim_head)
        k_flat = k4.reshape(b, gathered_t, local_heads * dim_head)
        v_flat = v4.reshape(b, gathered_t, local_heads * dim_head)

        out = self.original_fn(q_flat, k_flat, v_flat, local_heads, mask)
        out4 = out.view(b, gathered_t, local_heads, dim_head)

        # Reverse AllToAll: scatter seq(1), gather head(2)
        out4 = all_to_all_4d(out4, scatter_idx=1, gather_idx=2, group=self.sp_group)
        return out4.reshape(b, local_t, heads * dim_head)


def inject_ulysses_attention(
    model: torch.nn.Module,
    sp_group: dist.ProcessGroup,
) -> None:
    """Replace self-attention functions in all transformer blocks with Ulysses-wrapped versions."""
    blocks = None
    if hasattr(model, "transformer_blocks"):
        blocks = model.transformer_blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    if blocks is None:
        return

    video_inner_dim = 4096  # 32 heads × 128 dim_head
    injected = 0
    for i, block in enumerate(blocks):
        if hasattr(block, "attn1") and hasattr(block.attn1, "attention_function"):
            attn = block.attn1
            inner = attn.heads * attn.dim_head
            if inner == video_inner_dim:
                original = attn.attention_function
                attn.attention_function = UlyssesAttention(original, sp_group)
                injected += 1

    logger.info("inject_ulysses_attention: injected %d / %d blocks", injected, len(blocks))


def _split_tensor_along_seq(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    seq_len = tensor.shape[1]
    chunk_size = seq_len // world_size
    return tensor[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()


def _gather_tensor_along_seq(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=1)


def _split_rope(pe, rank: int, world_size: int):
    if pe is None:
        return None
    cos, sin = pe
    if cos.ndim == 4:
        seq_len = cos.shape[2]
        chunk = seq_len // world_size
        start = rank * chunk
        end = start + chunk
        return (
            cos[:, :, start:end].contiguous(),
            sin[:, :, start:end].contiguous(),
        )
    return (
        _split_tensor_along_seq(cos, rank, world_size),
        _split_tensor_along_seq(sin, rank, world_size),
    )


def install_sequence_parallel_hooks(
    model: torch.nn.Module,
    sp_group: dist.ProcessGroup,
) -> None:
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)
    if world_size <= 1:
        return

    original_process_blocks = model._process_transformer_blocks

    def parallel_process_blocks(video, audio, perturbations):
        original_seq_len = video.x.shape[1] if video is not None else 0
        original_audio_len = audio.x.shape[1] if audio is not None else 0

        # 1. 视频切片 (原逻辑)
        if video is not None:
            if original_seq_len % world_size != 0:
                pad_len = world_size - (original_seq_len % world_size)
                video = _pad_transformer_args(video, pad_len)
            video = _split_transformer_args(video, rank, world_size)
            
        # 2. 【核心修复】音频切片！防止 AllToAll 产生 8 重回音污染
        if audio is not None:
            if original_audio_len % world_size != 0:
                pad_len = world_size - (original_audio_len % world_size)
                audio = _pad_transformer_args(audio, pad_len)
            audio = _split_transformer_args(audio, rank, world_size)

        video_out, audio_out = original_process_blocks(video, audio, perturbations)

        # 3. 视频收集 (原逻辑)
        if video_out is not None:
            gathered_x = _gather_tensor_along_seq(video_out.x, sp_group)
            gathered_et = video_out.embedded_timestep
            if gathered_et is not None and gathered_et.shape[1] > 1:
                gathered_et = _gather_tensor_along_seq(gathered_et, sp_group)

            if original_seq_len > 0 and gathered_x.shape[1] != original_seq_len:
                gathered_x = gathered_x[:, :original_seq_len]
                if gathered_et is not None and gathered_et.shape[1] > 1:
                    gathered_et = gathered_et[:, :original_seq_len]

            video_out = replace(video_out, x=gathered_x, embedded_timestep=gathered_et)

        # 4. 【核心修复】音频收集！完美还原声音特征
        if audio_out is not None:
            gathered_audio_x = _gather_tensor_along_seq(audio_out.x, sp_group)
            gathered_audio_et = audio_out.embedded_timestep
            if gathered_audio_et is not None and gathered_audio_et.shape[1] > 1:
                gathered_audio_et = _gather_tensor_along_seq(gathered_audio_et, sp_group)

            if original_audio_len > 0 and gathered_audio_x.shape[1] != original_audio_len:
                gathered_audio_x = gathered_audio_x[:, :original_audio_len]
                if gathered_audio_et is not None and gathered_audio_et.shape[1] > 1:
                    gathered_audio_et = gathered_audio_et[:, :original_audio_len]

                audio_out = replace(audio_out, x=gathered_audio_x, embedded_timestep=gathered_audio_et)

        return video_out, audio_out

    model._process_transformer_blocks = parallel_process_blocks
    logger.info(
        "Installed sequence-parallel hooks on LTXModel (rank=%d, world_size=%d)",
        rank,
        world_size,
    )


def _pad_rope(pe, pad_len: int):
    if pe is None or pad_len == 0:
        return pe
    cos, sin = pe
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad_len))
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad_len))
    return (cos, sin)


def _pad_seq_dim(t: torch.Tensor | None, pad_len: int) -> torch.Tensor | None:
    if t is None or pad_len == 0:
        return t
    if t.shape[1] <= 1:
        return t
    return torch.nn.functional.pad(t, (0, 0, 0, pad_len))


def _split_seq_dim(t: torch.Tensor | None, rank: int, world_size: int) -> torch.Tensor | None:
    if t is None:
        return t
    if t.shape[1] <= 1:
        return t
    return _split_tensor_along_seq(t, rank, world_size)


def _pad_transformer_args(args, pad_len: int):
    return replace(
        args,
        x=torch.nn.functional.pad(args.x, (0, 0, 0, pad_len)),
        positional_embeddings=_pad_rope(args.positional_embeddings, pad_len),
        cross_positional_embeddings=_pad_rope(args.cross_positional_embeddings, pad_len),
        timesteps=_pad_seq_dim(args.timesteps, pad_len),
        embedded_timestep=_pad_seq_dim(args.embedded_timestep, pad_len),
    )


def _split_transformer_args(args, rank: int, world_size: int):
    return replace(
        args,
        x=_split_tensor_along_seq(args.x, rank, world_size),
        positional_embeddings=_split_rope(args.positional_embeddings, rank, world_size),
        cross_positional_embeddings=_split_rope(args.cross_positional_embeddings, rank, world_size),
        timesteps=_split_seq_dim(args.timesteps, rank, world_size),
        embedded_timestep=_split_seq_dim(args.embedded_timestep, rank, world_size),
        self_attention_mask=None,
    )
