"""Shared pytest fixtures for distributed NPU tests.

Tests are launched via: torchrun --nproc_per_node=N -m pytest testcase/ -v
The distributed environment (RANK, WORLD_SIZE, etc.) is set by torchrun.
"""

import os
import pytest
import torch
import torch.distributed as dist


def _is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


@pytest.fixture(scope="session", autouse=True)
def distributed_env():
    """Initialize and tear down the distributed process group for the test session."""
    if not _is_torchrun():
        yield
        return

    import ltx_npu  # noqa: F401 — triggers NPU runtime init

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.npu.is_available() if hasattr(torch, "npu") else False:
        backend = "hccl"
        torch.npu.set_device(local_rank)
    elif torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    yield

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@pytest.fixture
def world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@pytest.fixture
def device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.npu.is_available() if hasattr(torch, "npu") else False:
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
