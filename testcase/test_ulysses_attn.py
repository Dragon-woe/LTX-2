"""Unit tests for Ulysses Sequence Parallelism attention.

Run with: torchrun --nproc_per_node=2 -m pytest testcase/test_ulysses_attn.py -v
"""

import os
import pytest
import torch
import torch.distributed as dist


def _skip_if_not_distributed():
    if "RANK" not in os.environ:
        pytest.skip("Requires torchrun (distributed env)")


class TestAllToAll4D:
    """T011: Verify scatter/gather correctness on head and seq dims."""

    def test_scatter_head_gather_seq(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import all_to_all_4d

        B, T, H, D = 1, 4, 4, 8
        x = torch.arange(B * T * H * D, dtype=torch.float32, device=device).reshape(B, T, H, D)
        # scatter on head (dim=2), gather on seq (dim=1)
        y = all_to_all_4d(x, scatter_idx=2, gather_idx=1, group=dist.group.WORLD)

        assert y.shape == (B, T * world_size, H // world_size, D)

        # reverse: scatter on seq (dim=1), gather on head (dim=2)
        z = all_to_all_4d(y, scatter_idx=1, gather_idx=2, group=dist.group.WORLD)
        assert z.shape == x.shape
        assert torch.allclose(z, x), f"Roundtrip failed: max diff = {(z - x).abs().max()}"

    def test_preserves_dtype(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import all_to_all_4d

        x = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16, device=device)
        y = all_to_all_4d(x, scatter_idx=2, gather_idx=1, group=dist.group.WORLD)
        assert y.dtype == torch.bfloat16


class TestUlyssesAttention:
    """T012: Verify numerical equivalence with single-card full attention."""

    def test_numerical_equivalence(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import UlyssesAttention

        B, global_T, H, D = 1, 8, 4, 16
        heads = H

        torch.manual_seed(42)
        full_q = torch.randn(B, global_T, H * D, dtype=torch.bfloat16, device=device)
        full_k = torch.randn(B, global_T, H * D, dtype=torch.bfloat16, device=device)
        full_v = torch.randn(B, global_T, H * D, dtype=torch.bfloat16, device=device)

        # Single-card reference: standard SDPA
        def reference_attn(q, k, v, heads, mask=None):
            b, t, hd = q.shape
            d = hd // heads
            q2 = q.view(b, t, heads, d).transpose(1, 2)
            k2 = k.view(b, t, heads, d).transpose(1, 2)
            v2 = v.view(b, t, heads, d).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
            return out.transpose(1, 2).reshape(b, t, hd)

        ref_out = reference_attn(full_q, full_k, full_v, heads)

        # Ulysses: each rank gets a chunk of the sequence
        chunk = global_T // world_size
        local_q = full_q[:, rank * chunk:(rank + 1) * chunk].contiguous()
        local_k = full_k[:, rank * chunk:(rank + 1) * chunk].contiguous()
        local_v = full_v[:, rank * chunk:(rank + 1) * chunk].contiguous()

        ulysses_fn = UlyssesAttention(reference_attn, dist.group.WORLD)
        local_out = ulysses_fn(local_q, local_k, local_v, heads)

        expected_local = ref_out[:, rank * chunk:(rank + 1) * chunk]
        assert torch.allclose(local_out.float(), expected_local.float(), atol=1e-2, rtol=1e-2), \
            f"Rank {rank}: max diff = {(local_out.float() - expected_local.float()).abs().max()}"


class TestInjectUlysses:
    """T013: Verify attention_function replacement on a model."""

    def test_inject_replaces_attn1(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.ulysses_attn import UlyssesAttention, inject_ulysses_attention

        # Build a minimal mock that mimics BasicAVTransformerBlock structure
        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_function = lambda q, k, v, h, m=None: v

        class FakeBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = FakeAttention()
                self.attn2 = FakeAttention()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])

        model = FakeModel()
        original_fn = model.transformer_blocks[0].attn1.attention_function

        inject_ulysses_attention(model, dist.group.WORLD)

        for block in model.transformer_blocks:
            assert isinstance(block.attn1.attention_function, UlyssesAttention)
            # attn2 (cross-attention) should NOT be replaced
            assert not isinstance(block.attn2.attention_function, UlyssesAttention)
