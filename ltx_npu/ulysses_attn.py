"""Ulysses Sequence Parallelism: AllToAll-based attention + sequence split.

Two levels of parallelism:
1. Sequence split: each rank processes T/N tokens through LayerNorm, QKV proj, FFN
2. AllToAll in attention: scatter heads, gather seq for correct full-sequence attention
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def all_to_all_4d(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """AllToAll on a 4D tensor: scatter on one dim, gather on another.

    Uses dist.all_to_all (list version) for HCCL compatibility.

    Args:
        input: (B, T, H, D) tensor
        scatter_idx: dimension to scatter (split across ranks)
        gather_idx: dimension to gather (concatenate from ranks)
        group: process group for communication
    Returns:
        Tensor with scatter_idx shrunk by world_size and gather_idx grown by world_size.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input

    shape = list(input.shape)
    assert shape[scatter_idx] % world_size == 0, (
        f"dim {scatter_idx} size {shape[scatter_idx]} not divisible by world_size {world_size}"
    )

    input_chunks = list(input.chunk(world_size, dim=scatter_idx))
    input_chunks = [c.contiguous() for c in input_chunks]

    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks, group=group)

    return torch.cat(output_chunks, dim=gather_idx)


class UlyssesAttention:
    """Wraps attention with pre/post AllToAll for correct full-sequence attention.

    When the sequence is already split across ranks (each rank has T/N tokens),
    AllToAll gathers the full sequence but scatters heads before SDPA,
    then reverses after SDPA.
    """

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
        # Each rank: (B, local_T, H, D) → (B, global_T, H/N, D)
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
    """Split a (B, T, ...) tensor along dim=1 for this rank."""
    seq_len = tensor.shape[1]
    chunk_size = seq_len // world_size
    return tensor[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()


def _gather_tensor_along_seq(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather (B, local_T, ...) tensors from all ranks along dim=1."""
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=1)


def _split_rope(pe, rank: int, world_size: int):
    """Split RoPE positional embeddings (cos, sin) tuple along sequence dim.

    INTERLEAVED RoPE: (B, T, D) — sequence is dim 1.
    SPLIT RoPE:       (B, H, T, D) — sequence is dim 2.
    """
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
    """Install hooks on LTXModel to split/gather sequences around transformer blocks.

    This is the KEY optimization: each rank processes only T/N tokens through ALL layers
    (LayerNorm, QKV proj, FFN), giving N× compute reduction. AllToAll inside attention
    ensures correct full-sequence attention despite the split.
    """
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)

    if world_size <= 1:
        return

    original_process_blocks = model._process_transformer_blocks

    def parallel_process_blocks(video, audio, perturbations):
        """Split sequence before blocks, gather after."""
        original_seq_len = 0

        if video is not None:
            original_seq_len = video.x.shape[1]
            seq_len = original_seq_len
            if seq_len % world_size != 0:
                pad_len = world_size - (seq_len % world_size)
                video = _pad_transformer_args(video, pad_len)

            video = _split_transformer_args(video, rank, world_size)

        video_out, audio_out = original_process_blocks(video, audio, perturbations)

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

        return video_out, audio_out

    model._process_transformer_blocks = parallel_process_blocks
    logger.info("Installed sequence-parallel hooks on LTXModel (rank=%d, world_size=%d)", rank, world_size)


def _pad_rope(pe, pad_len: int):
    """Pad RoPE (cos, sin) tuple to make sequence length divisible."""
    if pe is None or pad_len == 0:
        return pe
    cos, sin = pe
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad_len))
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad_len))
    return (cos, sin)


def _pad_seq_dim(t: torch.Tensor | None, pad_len: int) -> torch.Tensor | None:
    """Pad a (B, T, ...) tensor along dim=1 if it has more than 1 token."""
    if t is None or pad_len == 0:
        return t
    if t.shape[1] <= 1:
        return t
    return torch.nn.functional.pad(t, (0, 0, 0, pad_len))


def _split_seq_dim(t: torch.Tensor | None, rank: int, world_size: int) -> torch.Tensor | None:
    """Split a (B, T, ...) tensor along dim=1. Skip if T==1 (broadcast)."""
    if t is None:
        return t
    if t.shape[1] <= 1:
        return t
    return _split_tensor_along_seq(t, rank, world_size)


def _pad_transformer_args(args, pad_len: int):
    """Pad all sequence-dependent fields of TransformerArgs."""
    return replace(
        args,
        x=torch.nn.functional.pad(args.x, (0, 0, 0, pad_len)),
        positional_embeddings=_pad_rope(args.positional_embeddings, pad_len),
        cross_positional_embeddings=_pad_rope(args.cross_positional_embeddings, pad_len),
        timesteps=_pad_seq_dim(args.timesteps, pad_len),
        embedded_timestep=_pad_seq_dim(args.embedded_timestep, pad_len),
    )


def _split_transformer_args(args, rank: int, world_size: int):
    """Split all sequence-dependent fields for this rank."""
    return replace(
        args,
        x=_split_tensor_along_seq(args.x, rank, world_size),
        positional_embeddings=_split_rope(args.positional_embeddings, rank, world_size),
        cross_positional_embeddings=_split_rope(args.cross_positional_embeddings, rank, world_size),
        timesteps=_split_seq_dim(args.timesteps, rank, world_size),
        embedded_timestep=_split_seq_dim(args.embedded_timestep, rank, world_size),
        self_attention_mask=None,
    )
