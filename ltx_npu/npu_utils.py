"""NPU device utilities: process cleanup, health checks."""

from __future__ import annotations

import os
import re
import signal
import subprocess


def cleanup_npu_processes() -> int:
    """Kill residual processes on all NPU devices. Returns count of killed PIDs."""
    current_pid = os.getpid()
    parent_pid = os.getppid()
    killed = 0

    try:
        result = subprocess.run(
            ["npu-smi", "info", "-l"],
            capture_output=True, text=True, check=False,
        )
        device_ids = re.findall(r"NPU ID\s*:\s*(\d+)", result.stdout)
    except FileNotFoundError:
        return 0

    for dev_id in device_ids:
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "proc-mem", "-i", dev_id, "-c", "0"],
                capture_output=True, text=True, check=False,
            )
            pids = re.findall(r"PID\s*:\s*(\d+)", result.stdout)
        except FileNotFoundError:
            continue

        for pid_str in pids:
            pid = int(pid_str)
            if pid in (current_pid, parent_pid, 1):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass

    return killed


def check_npu_health() -> list[dict]:
    """Return health info for each NPU device."""
    devices = []
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-l"],
            capture_output=True, text=True, check=False,
        )
        device_ids = re.findall(r"NPU ID\s*:\s*(\d+)", result.stdout)
    except FileNotFoundError:
        return devices

    for dev_id in device_ids:
        info = {"id": int(dev_id), "health": "Unknown"}
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "health", "-i", dev_id],
                capture_output=True, text=True, check=False,
            )
            if "OK" in result.stdout:
                info["health"] = "OK"
            elif "Warning" in result.stdout:
                info["health"] = "Warning"
            else:
                info["health"] = "Error"
        except FileNotFoundError:
            pass
        devices.append(info)

    return devices
