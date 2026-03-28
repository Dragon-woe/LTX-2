from __future__ import annotations

import logging
import math
import os
from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None

try:
    from mindiesd import attention_forward as mindie_attention_forward
except Exception:
    try:
        from mindiesd.layers.flash_attn.attention_forward import attention_forward as mindie_attention_forward
    except Exception:
        mindie_attention_forward = None

try:
    from mindiesd import rotary_position_embedding as mindie_rotary_position_embedding
except Exception:
    try:
        from mindiesd.layers.embedding import rotary_position_embedding as mindie_rotary_position_embedding
    except Exception:
        mindie_rotary_position_embedding = None

logger = logging.getLogger(__name__)


_MINDIE_INIT_LOGGED = False
_MINDIE_FALLBACK_LOGGED = False
_MINDIE_ROPE_INIT_LOGGED = False
_MINDIE_ROPE_FALLBACK_LOGGED = False


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _switch_mode(var_name: str, switch_name: str, legacy_enable_name: str, default: str = "off") -> str:
    raw = os.getenv(var_name)
    if raw is None or not raw.strip():
        raw = os.getenv(switch_name)

    if raw is None or not raw.strip():
        legacy = os.getenv(legacy_enable_name)
        if legacy is not None and legacy.strip() != "":
            return "on" if _env_flag(legacy_enable_name, "0") else "off"
        return default

    return raw.strip().lower()


def _mindie_switch_mode() -> str:
    return _switch_mode("MINDIESD_ATTN_MODE", "MINDIESD_ATTN_SWITCH", "MINDIESD_ATTN_ENABLE", default="off")


def _mindie_rope_switch_mode() -> str:
    return _switch_mode("MINDIESD_ROPE_MODE", "MINDIESD_ROPE_SWITCH", "MINDIESD_ROPE_ENABLE", default="off")


def _mindie_enabled() -> bool:
    return _mindie_switch_mode() in {"on", "1", "true", "yes", "y", "mindie", "auto"}


def _mindie_rope_enabled() -> bool:
    return _mindie_rope_switch_mode() in {"on", "1", "true", "yes", "y", "mindie", "auto"}


def _mindie_verbose() -> bool:
    return _env_flag("MINDIESD_ATTN_VERBOSE", "0") or _env_flag("MINDIESD_ROPE_VERBOSE", "0")


def _mindie_opt_mode() -> str:
    return os.getenv("MINDIESD_ATTN_OPT_MODE", "runtime").strip().lower()


def _mindie_layout() -> str:
    return os.getenv("MINDIESD_ATTN_LAYOUT", "BSND").strip().upper()


def _mindie_short_seq() -> int:
    value = os.getenv("MINDIESD_ATTN_SHORT_SEQ", "4000").strip()
    try:
        return max(int(value), 1)
    except Exception:
        return 4000


def _mindie_short_op() -> str:
    return os.getenv("MINDIESD_ATTN_SHORT_OP", "fused_attn_score").strip()


def _mindie_long_op() -> str:
    return os.getenv("MINDIESD_ATTN_LONG_OP", "ascend_laser_attention").strip()


def _mindie_rope_fused() -> bool:
    return _env_flag("MINDIESD_ROPE_FUSED", "1")


def _should_use_mindie(q: torch.Tensor) -> bool:
    mode = _mindie_switch_mode()
    if mode in {"off", "0", "false", "no", "n", "original", "raw", "disable", "disabled"}:
        return False
    if mindie_attention_forward is None:
        return False
    if q.device.type != "npu":
        return False
    if torch.is_grad_enabled() or q.requires_grad:
        return False
    return True


def _should_use_mindie_rope(x: torch.Tensor, rope_type: LTXRopeType) -> bool:
    mode = _mindie_rope_switch_mode()
    if mode in {"off", "0", "false", "no", "n", "original", "raw", "disable", "disabled"}:
        return False
    if mindie_rotary_position_embedding is None:
        return False
    if x.device.type != "npu":
        return False
    if torch.is_grad_enabled() or x.requires_grad:
        return False
    # 当前优先支持 LTX 默认的 interleaved RoPE；其它模式自动回退原实现
    if rope_type is not LTXRopeType.INTERLEAVED:
        return False
    return True


def _normalize_mask_for_mindie(mask: torch.Tensor | None, q_len: int | None = None) -> torch.Tensor | None:
    if mask is None:
        return None

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        return mask

    if q_len is not None and mask.shape[-2] == 1 and q_len > 1:
        expand_shape = list(mask.shape)
        expand_shape[-2] = q_len
        mask = mask.expand(*expand_shape)

    return mask


def _to_bsnd(x: torch.Tensor, heads: int) -> tuple[torch.Tensor, int]:
    b, s, inner_dim = x.shape
    dim_head = inner_dim // heads
    return x.view(b, s, heads, dim_head), dim_head


def _normalize_interleaved_freq(
    freq: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    heads: int,
    dim_head: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freq = freq.to(device=device, dtype=dtype)

    if freq.ndim == 2:
        if tuple(freq.shape) == (seq_len, dim_head):
            return freq.unsqueeze(0).unsqueeze(2).expand(batch, seq_len, heads, dim_head)
        if tuple(freq.shape) == (seq_len, heads * dim_head):
            return freq.view(seq_len, heads, dim_head).unsqueeze(0).expand(batch, seq_len, heads, dim_head)

    elif freq.ndim == 3:
        if tuple(freq.shape) == (batch, seq_len, dim_head):
            return freq.unsqueeze(2).expand(batch, seq_len, heads, dim_head)
        if tuple(freq.shape) == (batch, seq_len, heads * dim_head):
            return freq.view(batch, seq_len, heads, dim_head)
        if tuple(freq.shape) == (seq_len, heads, dim_head):
            return freq.unsqueeze(0).expand(batch, seq_len, heads, dim_head)

    elif freq.ndim == 4:
        if freq.shape[0] in (1, batch) and freq.shape[1] in (1, seq_len) and freq.shape[2] in (1, heads) and freq.shape[3] == dim_head:
            target = freq
            if target.shape[0] == 1 and batch != 1:
                target = target.expand(batch, target.shape[1], target.shape[2], target.shape[3])
            if target.shape[1] == 1 and seq_len != 1:
                target = target.expand(target.shape[0], seq_len, target.shape[2], target.shape[3])
            if target.shape[2] == 1 and heads != 1:
                target = target.expand(target.shape[0], target.shape[1], heads, target.shape[3])
            return target
        if freq.shape[0] in (1, batch) and freq.shape[1] in (1, heads) and freq.shape[2] in (1, seq_len) and freq.shape[3] == dim_head:
            target = freq.permute(0, 2, 1, 3)
            if target.shape[0] == 1 and batch != 1:
                target = target.expand(batch, target.shape[1], target.shape[2], target.shape[3])
            if target.shape[1] == 1 and seq_len != 1:
                target = target.expand(target.shape[0], seq_len, target.shape[2], target.shape[3])
            if target.shape[2] == 1 and heads != 1:
                target = target.expand(target.shape[0], target.shape[1], heads, target.shape[3])
            return target
        if freq.shape[0] == seq_len and freq.shape[1] == 1 and freq.shape[2] == 1 and freq.shape[3] == dim_head:
            return freq.permute(1, 0, 2, 3).expand(batch, seq_len, heads, dim_head)

    raise RuntimeError(
        f"Unsupported RoPE freq shape for MindIE interleaved mode: {tuple(freq.shape)}, "
        f"expected something broadcastable to (B,S,N,D)=({batch},{seq_len},{heads},{dim_head})."
    )


def _mindie_rope(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType,
    heads: int,
) -> torch.Tensor:
    global _MINDIE_ROPE_INIT_LOGGED

    if rope_type is not LTXRopeType.INTERLEAVED:
        raise RuntimeError(f"MindIE RoPE currently only enabled for INTERLEAVED, got {rope_type}.")

    x_bsnd, dim_head = _to_bsnd(x, heads)
    b, s, _, _ = x_bsnd.shape
    cos, sin = freqs_cis
    cos = _normalize_interleaved_freq(cos, batch=b, seq_len=s, heads=heads, dim_head=dim_head, device=x.device, dtype=x.dtype)
    sin = _normalize_interleaved_freq(sin, batch=b, seq_len=s, heads=heads, dim_head=dim_head, device=x.device, dtype=x.dtype)

    if _mindie_verbose() and not _MINDIE_ROPE_INIT_LOGGED:
        logger.info(
            f"[MindIE-LTX] rope mode={_mindie_rope_switch_mode()}, fused={_mindie_rope_fused()}, "
            f"x_shape={tuple(x_bsnd.shape)}, cos_shape={tuple(cos.shape)}, sin_shape={tuple(sin.shape)}"
        )
        _MINDIE_ROPE_INIT_LOGGED = True

    out = mindie_rotary_position_embedding(
        x_bsnd,
        cos,
        sin,
        rotated_mode="rotated_interleaved",
        head_first=False,
        fused=_mindie_rope_fused(),
    )
    return out.reshape_as(x)


def _mindie_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    global _MINDIE_INIT_LOGGED

    b, tq, inner_dim = q.shape
    tk = k.shape[1]
    dim_head = inner_dim // heads

    q_bsnd = q.view(b, tq, heads, dim_head)
    k_bsnd = k.view(b, tk, heads, dim_head)
    v_bsnd = v.view(b, tk, heads, dim_head)

    attn_mask = _normalize_mask_for_mindie(mask, q_len=tq)

    opt_mode = _mindie_opt_mode()
    kwargs: dict[str, object] = {}
    if opt_mode == "manual":
        seq_len = max(tq, tk)
        op_type = _mindie_short_op() if seq_len < _mindie_short_seq() else _mindie_long_op()
        kwargs["opt_mode"] = "manual"
        kwargs["op_type"] = op_type
        kwargs["layout"] = _mindie_layout()
    elif opt_mode == "static":
        kwargs["opt_mode"] = "static"
    else:
        kwargs["opt_mode"] = "runtime"

    if _mindie_verbose() and not _MINDIE_INIT_LOGGED:
        mode = _mindie_switch_mode()
        msg = f"[MindIE-LTX] attn mode={mode}, attention_forward opt_mode={kwargs.get('opt_mode', 'runtime')}"
        if "op_type" in kwargs:
            msg += f", op_type={kwargs['op_type']}, layout={kwargs.get('layout', 'BSND')}"
        msg += f", q_shape={tuple(q_bsnd.shape)}, k_shape={tuple(k_bsnd.shape)}"
        logger.info(msg)
        _MINDIE_INIT_LOGGED = True

    out = mindie_attention_forward(
        q_bsnd,
        k_bsnd,
        v_bsnd,
        attn_mask=attn_mask,
        scale=1.0 / math.sqrt(dim_head),
        fused=True,
        **kwargs,
    )

    return out.reshape(b, tq, heads * dim_head)


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            pad = 8 - mask.shape[-1] % 8
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        else:
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        global _MINDIE_FALLBACK_LOGGED, _MINDIE_ROPE_FALLBACK_LOGGED

        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                if _should_use_mindie_rope(q, self.rope_type):
                    try:
                        q = _mindie_rope(q, pe, self.rope_type, self.heads)
                        k = _mindie_rope(k, pe if k_pe is None else k_pe, self.rope_type, self.heads)
                    except Exception as exc:
                        if _mindie_verbose() and not _MINDIE_ROPE_FALLBACK_LOGGED:
                            logger.warning(f"[MindIE-LTX] rotary_position_embedding fallback to original RoPE: {exc}")
                            _MINDIE_ROPE_FALLBACK_LOGGED = True
                        q = apply_rotary_emb(q, pe, self.rope_type)
                        k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
                else:
                    q = apply_rotary_emb(q, pe, self.rope_type)
                    k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            if _should_use_mindie(q):
                try:
                    out = _mindie_attention(q, k, v, self.heads, mask)
                except Exception as exc:
                    if _mindie_verbose() and not _MINDIE_FALLBACK_LOGGED:
                        logger.warning(f"[MindIE-LTX] attention_forward fallback to original backend: {exc}")
                        _MINDIE_FALLBACK_LOGGED = True
                    out = self.attention_function(q, k, v, self.heads, mask)
            else:
                out = self.attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
