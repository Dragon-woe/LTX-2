"""Unit tests for VAE spatial patch parallel.

Run with: torchrun --nproc_per_node=2 -m pytest testcase/test_vae_parallel.py -v
(or --nproc_per_node=4 for 4-card tests)
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _skip_if_not_distributed():
    if "RANK" not in os.environ:
        pytest.skip("Requires torchrun (distributed env)")


class TestPatchDispatch:
    """T021: Verify patch/dispatch roundtrip preserves tensor values."""

    def test_roundtrip_correctness(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.vae_parallel import VAEParallelContext

        B, C, T, H, W = 1, 3, 2, 8, 8
        ctx = VAEParallelContext(world_size=world_size, rank=rank, device=device)

        if rank == 0:
            full_tensor = torch.arange(B * C * T * H * W, dtype=torch.float32, device=device).reshape(B, C, T, H, W)
        else:
            full_tensor = torch.zeros(B, C, T, H, W, dtype=torch.float32, device=device)

        dist.broadcast(full_tensor, src=0)
        local_patch = ctx.patch(full_tensor)

        assert local_patch.shape[3] == H // ctx.h_split
        assert local_patch.shape[4] == W // ctx.w_split

        restored = ctx.dispatch(local_patch)
        assert torch.allclose(restored, full_tensor), \
            f"Rank {rank}: max diff = {(restored - full_tensor).abs().max()}"


class TestBoundaryExchange:
    """T022: Verify P2P boundary exchange correctness."""

    def test_exchange_rows(self, rank, world_size, device):
        _skip_if_not_distributed()
        if world_size < 2:
            pytest.skip("Need >=2 ranks for row exchange")

        from ltx_npu.vae_parallel import VAEParallelContext

        ctx = VAEParallelContext(world_size=world_size, rank=rank, device=device)
        B, C, T, H, W = 1, 1, 1, 4, 4
        data = torch.full((B, C, T, H, W), fill_value=float(rank), dtype=torch.float32, device=device)
        pad_h = 1

        result = ctx.exchange_rows(data, pad_h)
        assert result.shape[3] == H + 2 * pad_h


class TestWrappedConv3d:
    """T023: Verify distributed conv3d produces same result as single-card."""

    def test_conv3d_equivalence(self, rank, world_size, device):
        _skip_if_not_distributed()
        from ltx_npu.vae_parallel import VAEParallelContext

        B, C_in, T, H, W = 1, 2, 2, 8, 8
        C_out = 2
        torch.manual_seed(42)

        weight = torch.randn(C_out, C_in, 3, 3, 3, dtype=torch.float32, device=device)
        bias = torch.randn(C_out, dtype=torch.float32, device=device)
        dist.broadcast(weight, src=0)
        dist.broadcast(bias, src=0)

        full_input = torch.randn(B, C_in, T, H, W, dtype=torch.float32, device=device)
        dist.broadcast(full_input, src=0)

        ref_output = F.conv3d(full_input, weight, bias, padding=(1, 1, 1))

        ctx = VAEParallelContext(world_size=world_size, rank=rank, device=device)
        local_input = ctx.patch(full_input)

        padded = ctx.exchange_rows(local_input, pad_h=1)
        padded = ctx.exchange_columns(padded, pad_w=1)

        local_output = F.conv3d(padded, weight, bias, padding=(1, 0, 0))

        restored_output = ctx.dispatch(local_output)

        assert torch.allclose(restored_output, ref_output, atol=1e-4), \
            f"Rank {rank}: max diff = {(restored_output - ref_output).abs().max()}"
