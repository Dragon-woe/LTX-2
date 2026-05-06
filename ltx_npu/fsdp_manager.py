"""FSDP wrapper for LTX-2 Transformer inference on Ascend NPU.

Shards the 22B Transformer across N ranks using PyTorch FSDP FULL_SHARD.
Each rank holds 1/N of the parameters permanently. During forward, each
BasicAVTransformerBlock AllGathers its full parameters, computes, then
frees the gathered copy — eliminating the gpu_model build/free/cleanup cycle.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def shard_transformer(
    model: torch.nn.Module,
    device_id: torch.device,
    process_group: ProcessGroup | None = None,
) -> FSDP:
    """Wrap *model* with FSDP FULL_SHARD, sharding per BasicAVTransformerBlock.

    Args:
        model: The full X0Model (with velocity_model containing transformer_blocks).
               Must already have Ulysses SP hooks injected before calling this.
        device_id: Target NPU device (e.g. torch.device("npu:0")).
        process_group: Distributed process group for FSDP communication.
                       Defaults to the world group.

    Returns:
        FSDP-wrapped model with parameters sharded across ranks.
    """
    from torch.distributed.fsdp import MixedPrecision
    from ltx_core.model.transformer.model import BasicAVTransformerBlock

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BasicAVTransformerBlock},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model.to(dtype=torch.bfloat16),
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        process_group=process_group,
        use_orig_params=True,
        mixed_precision=mp_policy,
    )

    num_params = sum(p.numel() for p in fsdp_model.parameters())
    world_size = dist.get_world_size(process_group) if process_group else dist.get_world_size()
    shard_size_gib = num_params * 2 / (1024**3 * world_size)  # bf16
    logger.info(
        "FSDP wrapped transformer: %d params, %d ranks, ~%.1f GiB/rank (bf16)",
        num_params, world_size, shard_size_gib,
    )

    return fsdp_model
