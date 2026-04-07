"""Unit tests for DeviceContext."""

import torch
import torch.distributed as dist
import pytest


def test_detect_device_type():
    from ltx_npu.device_context import DeviceContext
    ctx = DeviceContext.create()
    assert ctx.device_type in ("npu", "cuda", "cpu")


def test_get_device_returns_torch_device():
    from ltx_npu.device_context import DeviceContext
    ctx = DeviceContext.create()
    dev = ctx.get_device()
    assert isinstance(dev, torch.device)


def test_synchronize_does_not_raise():
    from ltx_npu.device_context import DeviceContext
    ctx = DeviceContext.create()
    ctx.synchronize()


def test_singleton_pattern():
    from ltx_npu.device_context import DeviceContext
    ctx1 = DeviceContext.current()
    ctx2 = DeviceContext.current()
    assert ctx1 is ctx2


def test_comm_backend():
    from ltx_npu.device_context import DeviceContext
    ctx = DeviceContext.create()
    backend = ctx.get_comm_backend()
    if ctx.device_type == "npu":
        assert backend == "hccl"
    elif ctx.device_type == "cuda":
        assert backend == "nccl"
    else:
        assert backend == "gloo"


def test_rank_and_world_size():
    from ltx_npu.device_context import DeviceContext
    ctx = DeviceContext.create()
    if dist.is_initialized():
        assert ctx.rank == dist.get_rank()
        assert ctx.world_size == dist.get_world_size()
    else:
        assert ctx.rank == 0
        assert ctx.world_size == 1
