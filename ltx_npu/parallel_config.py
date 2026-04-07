"""Parallel configuration built from CLI arguments."""

from __future__ import annotations

import argparse
import sys

import torch.distributed as dist

NUM_ATTENTION_HEADS = 32


class ParallelConfig:
    def __init__(
        self,
        ulysses_degree: int = 1,
        vae_parallel: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.ulysses_degree = ulysses_degree
        self.vae_parallel = vae_parallel
        self.world_size = world_size
        self.rank = rank
        self.sp_group: dist.ProcessGroup | None = None

    def validate(self) -> None:
        if NUM_ATTENTION_HEADS % self.ulysses_degree != 0:
            print(
                f"[ERROR] --ulysses-degree {self.ulysses_degree} does not divide "
                f"num_attention_heads ({NUM_ATTENTION_HEADS}). "
                f"Allowed values: {[d for d in range(1, NUM_ATTENTION_HEADS + 1) if NUM_ATTENTION_HEADS % d == 0]}",
                file=sys.stderr,
            )
            sys.exit(1)

        if self.ulysses_degree > 1 and self.world_size != self.ulysses_degree:
            print(
                f"[ERROR] world_size ({self.world_size}) must equal "
                f"ulysses_degree ({self.ulysses_degree}). "
                f"Use: torchrun --nproc_per_node={self.ulysses_degree}",
                file=sys.stderr,
            )
            sys.exit(1)

    def init_process_groups(self) -> None:
        if self.ulysses_degree <= 1 or not dist.is_initialized():
            return
        ranks = list(range(self.world_size))
        self.sp_group = dist.new_group(ranks)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ParallelConfig:
        ws = 1
        rk = 0
        if dist.is_initialized():
            ws = dist.get_world_size()
            rk = dist.get_rank()
        cfg = cls(
            ulysses_degree=getattr(args, "ulysses_degree", 1),
            vae_parallel=getattr(args, "vae_parallel", False),
            world_size=ws,
            rank=rk,
        )
        cfg.validate()
        cfg.init_process_groups()
        return cfg

    @property
    def is_parallel(self) -> bool:
        return self.ulysses_degree > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0
