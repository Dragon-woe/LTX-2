"""Unit tests for end-to-end sequence parallelism.

Run with: torchrun --nproc_per_node=2 -m pytest testcase/test_e2e_seq_parallel.py -v
"""

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist


def _skip_if_not_distributed():
    if "RANK" not in os.environ:
        pytest.skip("Requires torchrun (distributed env)")


@dataclass(frozen=True)
class MockTransformerArgs:
    """Minimal mock matching the fields that split/gather touches."""
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    cross_positional_embeddings: object = None
    cross_scale_shift_timestep: object = None
    cross_gate_timestep: object = None
    enabled: bool = True
    prompt_timestep: object = None
    self_attention_mask: object = None


class TestSplitGatherRoundtrip:
    """T003: Verify split→gather roundtrip preserves data."""

    def test_split_gather_roundtrip(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import (
            _split_transformer_args,
            _gather_tensor_along_seq,
        )

        B, T, D = 1, 8, 64
        x = torch.arange(B * T * D, dtype=torch.float32, device=device).reshape(B, T, D)
        ts = torch.randn(B, T, D, device=device)
        et = torch.randn(B, T, D, device=device)
        pe_cos = torch.randn(B, T, D // 2, device=device)
        pe_sin = torch.randn(B, T, D // 2, device=device)
        ctx_tensor = torch.randn(B, 5, D, device=device)

        dist.broadcast(x, src=0)
        dist.broadcast(ts, src=0)
        dist.broadcast(et, src=0)
        dist.broadcast(pe_cos, src=0)
        dist.broadcast(pe_sin, src=0)
        dist.broadcast(ctx_tensor, src=0)

        args = MockTransformerArgs(
            x=x, context=ctx_tensor, context_mask=torch.ones(B, 5, device=device),
            timesteps=ts, embedded_timestep=et,
            positional_embeddings=(pe_cos, pe_sin),
        )

        split_args = _split_transformer_args(args, rank, world_size)

        assert split_args.x.shape == (B, T // world_size, D)
        assert split_args.timesteps.shape == (B, T // world_size, D)
        assert split_args.context.shape == (B, 5, D), "context must NOT be split"

        gathered_x = _gather_tensor_along_seq(split_args.x, dist.group.WORLD)
        assert torch.allclose(gathered_x, x), f"max diff = {(gathered_x - x).abs().max()}"


class TestSplitWithPadding:
    """T004: Verify pad→split→gather→trim for non-divisible seq lengths."""

    def test_split_with_padding(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import (
            _pad_transformer_args,
            _split_transformer_args,
            _gather_tensor_along_seq,
        )

        B, T, D = 1, 7, 32
        x = torch.randn(B, T, D, device=device)
        dist.broadcast(x, src=0)

        args = MockTransformerArgs(
            x=x, context=torch.randn(B, 3, D, device=device),
            context_mask=torch.ones(B, 3, device=device),
            timesteps=torch.randn(B, T, D, device=device),
            embedded_timestep=torch.randn(B, T, D, device=device),
            positional_embeddings=(torch.randn(B, T, 16, device=device),
                                   torch.randn(B, T, 16, device=device)),
        )
        dist.broadcast(args.timesteps, src=0)
        dist.broadcast(args.embedded_timestep, src=0)

        pad_len = world_size - (T % world_size)
        padded = _pad_transformer_args(args, pad_len)
        assert padded.x.shape[1] % world_size == 0

        split = _split_transformer_args(padded, rank, world_size)
        gathered = _gather_tensor_along_seq(split.x, dist.group.WORLD)
        trimmed = gathered[:, :T]

        assert torch.allclose(trimmed, x), f"max diff = {(trimmed - x).abs().max()}"


class TestE2EBlockEquivalence:
    """T005: Verify split→blocks→gather numerical equivalence with single-card."""

    def test_e2e_block_equivalence(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import (
            UlyssesAttention,
            _split_transformer_args,
            _gather_tensor_along_seq,
        )

        B, T, H, D = 1, 8, 4, 16
        inner_dim = H * D

        torch.manual_seed(42)
        q_proj = torch.nn.Linear(inner_dim, inner_dim, bias=True, device=device, dtype=torch.bfloat16)
        k_proj = torch.nn.Linear(inner_dim, inner_dim, bias=True, device=device, dtype=torch.bfloat16)
        v_proj = torch.nn.Linear(inner_dim, inner_dim, bias=True, device=device, dtype=torch.bfloat16)
        o_proj = torch.nn.Linear(inner_dim, inner_dim, bias=True, device=device, dtype=torch.bfloat16)

        for p in [q_proj, k_proj, v_proj, o_proj]:
            dist.broadcast(p.weight.data, src=0)
            dist.broadcast(p.bias.data, src=0)

        x_full = torch.randn(B, T, inner_dim, dtype=torch.bfloat16, device=device)
        dist.broadcast(x_full, src=0)

        def ref_attn(q, k, v, heads, mask=None):
            b, t, hd = q.shape
            d = hd // heads
            q2 = q.view(b, t, heads, d).transpose(1, 2)
            k2 = k.view(b, t, heads, d).transpose(1, 2)
            v2 = v.view(b, t, heads, d).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
            return out.transpose(1, 2).reshape(b, t, hd)

        # Single-card reference
        q_full = q_proj(x_full)
        k_full = k_proj(x_full)
        v_full = v_proj(x_full)
        ref_out = o_proj(ref_attn(q_full, k_full, v_full, H))

        # Parallel: split sequence, AllToAll attention, gather
        chunk = T // world_size
        x_local = x_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

        q_local = q_proj(x_local)
        k_local = k_proj(x_local)
        v_local = v_proj(x_local)

        ulysses_fn = UlyssesAttention(ref_attn, dist.group.WORLD)
        attn_out_local = ulysses_fn(q_local, k_local, v_local, H)
        out_local = o_proj(attn_out_local)

        out_gathered = _gather_tensor_along_seq(out_local, dist.group.WORLD)

        assert torch.allclose(out_gathered.float(), ref_out.float(), atol=1e-2, rtol=1e-2), \
            f"Rank {rank}: max diff = {(out_gathered.float() - ref_out.float()).abs().max()}"
