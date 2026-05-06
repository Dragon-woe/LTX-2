"""LTX-2 Ascend NPU adaptation package.

Initializes torch_npu runtime before any other torch operations.
Must be imported at the entry point before any model loading.

004 additions: fused operator patches (RMSNorm, FA multi-backend, LayerNorm,
RoPE freqs caching) controlled by environment variables.
"""

import torch

_npu_initialized = False


def init_npu_runtime():
    """Initialize NPU runtime settings. Idempotent — safe to call multiple times."""
    global _npu_initialized
    if _npu_initialized:
        return

    try:
        import torch_npu
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        _npu_initialized = True
    except ImportError:
        pass


def _resolve_safetensors_path(path: str) -> str:
    """Resolve a directory path to a safetensors file path.

    safetensors >= 0.7.0 requires a file path, not a directory.
    """
    import os
    if os.path.isdir(path):
        candidates = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if candidates:
            return os.path.join(path, candidates[0])
    return path


def _patch_sft_loader_for_directory():
    """Fix safetensors loading when model_path is a directory.

    safetensors >= 0.7.0 requires a file path, not a directory.
    Patches both metadata() and load() methods.
    """
    from ltx_core.loader.sft_loader import (
        SafetensorsModelStateDictLoader,
        SafetensorsStateDictLoader,
    )

    _original_metadata = SafetensorsModelStateDictLoader.metadata
    _original_load = SafetensorsStateDictLoader.load

    def _patched_metadata(self, path: str) -> dict:
        return _original_metadata(self, _resolve_safetensors_path(path))

    def _patched_load(self, path, sd_ops=None, device=None):
        if isinstance(path, list):
            path = [_resolve_safetensors_path(p) for p in path]
        elif isinstance(path, str):
            path = _resolve_safetensors_path(path)
        return _original_load(self, path, sd_ops, device)

    SafetensorsModelStateDictLoader.metadata = _patched_metadata
    SafetensorsStateDictLoader.load = _patched_load


def _apply_fused_ops():
    """Apply global fused operator patches based on env vars.

    Called once at import time. Only patches that are safe to apply globally
    (RMSNorm fusion, sft_loader fix) are done here. Model-specific patches
    (FA dispatch, LayerNorm fusion, freqs cache) are applied in
    ParallelDistilledPipeline.
    """
    from ltx_npu.fused_ops import get_config, patch_rmsnorm

    _patch_sft_loader_for_directory()

    cfg = get_config()
    if cfg.fused_rmsnorm:
        patch_rmsnorm()


if torch.npu.is_available() if hasattr(torch, "npu") else False:
    init_npu_runtime()
    _apply_fused_ops()
