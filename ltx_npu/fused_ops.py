"""Fused operator patches for Ascend NPU acceleration.

Provides:
- FusedOperatorConfig: env-var-driven config singleton
- mindiesd availability detection
- RMSNorm fusion (torch_npu.npu_rms_norm)
- FA multi-backend dispatch (NPUAttention)
- LayerNorm fusion (mindiesd.fast_layernorm)
- Timestep precision cleanup

All patches are opt-in via environment variables, defaulting to off.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mindiesd availability detection (T002)
# ---------------------------------------------------------------------------
has_mindiesd = False
_mindiesd_attention_forward = None
_mindiesd_fast_layernorm = None

try:
    from mindiesd import attention_forward as _msd_attn_fwd

    _mindiesd_attention_forward = _msd_attn_fwd
    has_mindiesd = True
    logger.info("mindiesd detected — attention_forward available")
except ImportError:
    logger.info("mindiesd not installed — FA multi-backend and fast_layernorm disabled")

if has_mindiesd:
    try:
        from mindiesd import fast_layernorm as _msd_fast_ln

        _mindiesd_fast_layernorm = _msd_fast_ln
    except ImportError:
        _mindiesd_fast_layernorm = None
        logger.info("mindiesd.fast_layernorm not available")


# ---------------------------------------------------------------------------
# FusedOperatorConfig (T001)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedOperatorConfig:
    """Immutable config read once from environment variables at import time."""

    algo: int = 0
    fused_rmsnorm: bool = False
    fast_layernorm: bool = False
    precision_cpu: bool = False

    @classmethod
    def from_env(cls) -> FusedOperatorConfig:
        algo_raw = int(os.getenv("ALGO", "0"))
        algo = algo_raw if algo_raw in (0, 1, 3) else 0
        if algo_raw not in (0, 1, 3) and algo_raw != 0:
            logger.warning("ALGO=%d not in {0,1,3}, falling back to 0", algo_raw)
        return cls(
            algo=algo,
            fused_rmsnorm=os.getenv("FUSED_RMSNORM", "0") == "1",
            fast_layernorm=os.getenv("FAST_LAYERNORM", "0") == "1",
            precision_cpu=os.getenv("PRECISION", "0") == "1",
        )


_config: FusedOperatorConfig | None = None


def get_config() -> FusedOperatorConfig:
    """Return the global FusedOperatorConfig singleton (lazy-init from env)."""
    global _config
    if _config is None:
        _config = FusedOperatorConfig.from_env()
        logger.info("FusedOperatorConfig: %s", _config)
    return _config


# ---------------------------------------------------------------------------
# RMSNorm fusion (T012-T013)
# ---------------------------------------------------------------------------
_original_rmsnorm_forward = None


def patch_rmsnorm() -> None:
    """Replace torch.nn.RMSNorm.forward with torch_npu.npu_rms_norm fused kernel."""
    global _original_rmsnorm_forward
    import torch_npu  # noqa: F401

    _original_rmsnorm_forward = torch.nn.RMSNorm.forward

    def _fused_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

    torch.nn.RMSNorm.forward = _fused_rmsnorm_forward
    logger.info("Patched torch.nn.RMSNorm.forward → torch_npu.npu_rms_norm")


def unpatch_rmsnorm() -> None:
    """Restore original torch.nn.RMSNorm.forward."""
    global _original_rmsnorm_forward
    if _original_rmsnorm_forward is not None:
        torch.nn.RMSNorm.forward = _original_rmsnorm_forward
        _original_rmsnorm_forward = None
        logger.info("Restored original torch.nn.RMSNorm.forward")


# ---------------------------------------------------------------------------
# FA multi-backend dispatch (T017-T020)
# ---------------------------------------------------------------------------
class NPUAttention:
    """Multi-backend attention dispatch for Ascend NPU.

    Replaces PytorchAttention as the attention callable. Selects the optimal
    FA backend based on ALGO env var and self/cross attention type.

    Input q/k/v: (B, T, H*D) — same as PytorchAttention protocol.
    Internally reshapes to BSND (B, SeqLen, NumHeads, HeadDim) for mindiesd,
    or BNSD (B, NumHeads, SeqLen, HeadDim) for SDPA/ALGO=3.
    """

    def __init__(self, algo: int = 0):
        self.algo = algo

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads

        if torch.npu.is_available():
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        # BSND: (B, SeqLen, NumHeads, HeadDim)
        q_bsnd = q.view(b, -1, heads, dim_head)
        k_bsnd = k.view(b, -1, heads, dim_head)
        v_bsnd = v.view(b, -1, heads, dim_head)

        is_self_attn = q_bsnd.shape[1] == k_bsnd.shape[1]

        out = self._dispatch(q_bsnd, k_bsnd, v_bsnd, heads, is_self_attn, mask)

        if out.ndim == 4 and out.shape[1] == heads:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            out = out.reshape(b, -1, heads * dim_head)
        return out

    def _dispatch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        is_self_attn: bool,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dispatch to the best FA backend. q/k/v are BSND format."""
        if not has_mindiesd or _mindiesd_attention_forward is None:
            q_bnsd = q.transpose(1, 2)
            k_bnsd = k.transpose(1, 2)
            v_bnsd = v.transpose(1, 2)
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
            return F.scaled_dot_product_attention(
                q_bnsd, k_bnsd, v_bnsd, attn_mask=mask, dropout_p=0.0, is_causal=False,
            )

        if is_self_attn and self.algo == 1:
            return _mindiesd_attention_forward(
                q, k, v,
                opt_mode="manual",
                op_type="ascend_laser_attention",
                layout="BNSD",
            )

        if is_self_attn and self.algo == 3:
            import torch_npu
            q_bnsd = q.transpose(1, 2)
            k_bnsd = k.transpose(1, 2)
            v_bnsd = v.transpose(1, 2)
            return torch_npu.npu_fused_infer_attention_score(
                q_bnsd, k_bnsd, v_bnsd,
                num_heads=heads,
                input_layout="BNSD",
            )

        return _mindiesd_attention_forward(
            q, k, v, op_type="fused_attn_score",
        )


def inject_npu_attention(model: torch.nn.Module) -> int:
    """Replace attention_function on all transformer block attentions with NPUAttention.

    Returns the number of attention modules replaced.
    """
    cfg = get_config()
    npu_attn = NPUAttention(algo=cfg.algo)
    injected = 0

    blocks = getattr(model, "transformer_blocks", None) or getattr(model, "blocks", None)
    if blocks is None:
        logger.warning("No transformer_blocks or blocks found on model")
        return 0

    for block in blocks:
        for attn_name in ("attn1", "attn2", "audio_attn1", "audio_attn2"):
            attn = getattr(block, attn_name, None)
            if attn is not None and hasattr(attn, "attention_function"):
                attn.attention_function = npu_attn
                injected += 1

    logger.info("Injected NPUAttention (ALGO=%d) into %d attention modules", cfg.algo, injected)
    return injected


# ---------------------------------------------------------------------------
# LayerNorm fusion (T025-T026)
# ---------------------------------------------------------------------------
_original_layernorm_forwards: dict[int, object] = {}


def patch_layernorm(model: torch.nn.Module) -> int:
    """Patch DiT LayerNorm modules with mindiesd.fast_layernorm.

    Only patches nn.LayerNorm instances found within transformer blocks,
    not VAE GroupNorm or other normalization layers.
    """
    if not has_mindiesd or _mindiesd_fast_layernorm is None:
        logger.info("mindiesd.fast_layernorm not available — skipping LayerNorm patch")
        return 0

    fast_ln = _mindiesd_fast_layernorm
    patched = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and "transformer" in name.lower():
            mid = id(module)
            if mid not in _original_layernorm_forwards:
                _original_layernorm_forwards[mid] = module.forward

                def _make_fast_forward(ln_module):
                    def _fast_forward(x):
                        return fast_ln(ln_module, x)
                    return _fast_forward

                module.forward = _make_fast_forward(module)
                patched += 1

    logger.info("Patched %d LayerNorm modules with mindiesd.fast_layernorm", patched)
    return patched


# ---------------------------------------------------------------------------
# Precision cleanup (T022)
# ---------------------------------------------------------------------------
def patch_timestep_precision() -> None:
    """Remove unnecessary .float() cast in timestep embedding computation.

    Patches PixArtAlphaCombinedTimestepSizeEmbeddings to avoid
    bfloat16→float32→bfloat16 round-trip.
    """
    try:
        from ltx_core.model.transformer.timestep_embedding import timestep_embedding

        original_fn = timestep_embedding

        def _patched_timestep_embedding(timesteps, embedding_dim, max_period=10000):
            half = embedding_dim // 2
            freqs = torch.exp(
                -torch.log(torch.tensor(max_period, dtype=torch.float32))
                * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
                / half
            )
            args = timesteps[:, None].to(freqs.dtype) * freqs[None, :]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if embedding_dim % 2:
                embedding = torch.cat(
                    [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                )
            return embedding

        import ltx_core.model.transformer.timestep_embedding as te_mod
        te_mod.timestep_embedding = _patched_timestep_embedding
        logger.info("Patched timestep_embedding: .float() → .to(freqs.dtype)")
    except (ImportError, AttributeError) as e:
        logger.warning("Could not patch timestep_embedding: %s", e)
