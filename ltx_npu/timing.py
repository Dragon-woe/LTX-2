"""Per-stage timing report with device synchronization."""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltx_npu.device_context import DeviceContext


class TimingReport:
    def __init__(self, ctx: DeviceContext | None = None):
        self.stages: OrderedDict[str, float] = OrderedDict()
        self._ctx = ctx

    @contextmanager
    def stage(self, name: str):
        if self._ctx is not None:
            self._ctx.synchronize()
        t0 = time.perf_counter()
        yield
        if self._ctx is not None:
            self._ctx.synchronize()
        elapsed = time.perf_counter() - t0
        self.stages[name] = elapsed

    def total(self) -> float:
        return sum(self.stages.values())

    def report(self) -> str:
        if not self.stages:
            return "[LTX-2 NPU Parallel] No stages recorded."

        name_width = max(len(n) for n in self.stages) + 2
        name_width = max(name_width, len("Stage") + 2)
        time_width = 10

        top = f"┌{'─' * name_width}┬{'─' * time_width}┐"
        header = f"│{'Stage'.center(name_width)}│{'Time (s)'.center(time_width)}│"
        sep = f"├{'─' * name_width}┼{'─' * time_width}┤"
        bottom = f"└{'─' * name_width}┴{'─' * time_width}┘"

        rows = []
        for name, elapsed in self.stages.items():
            rows.append(f"│ {name:<{name_width - 2}} │{elapsed:>{time_width - 2}.2f}  │")

        rows.append(f"│ {'Total':<{name_width - 2}} │{self.total():>{time_width - 2}.2f}  │")

        lines = [
            "[LTX-2 NPU Parallel] Performance Report (after warmup)",
            top, header, sep, *rows, bottom,
        ]
        return "\n".join(lines)
