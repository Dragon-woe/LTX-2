import json
import os
import re
from pathlib import Path

import torch


_INSTALLED = False


def _enabled():
    return os.getenv("LTX_BLOCK0_MID_DUMP", "0") == "1"


def _local_rank():
    return int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))


def _should_write():
    return _enabled() and _local_rank() == int(os.getenv("LTX_BLOCK0_MID_DUMP_LOCAL_RANK", "0"))


def _out_dir():
    d = Path(os.getenv("LTX_BLOCK0_MID_DUMP_DIR", "/tmp/block0_mid_dump"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _interesting_tensor(t: torch.Tensor) -> bool:
    if not torch.is_tensor(t) or t.numel() == 0:
        return False

    shape = tuple(int(x) for x in t.shape)
    if 126 in shape:
        return True

    if t.numel() <= int(os.getenv("LTX_BLOCK0_MID_SMALL_MAX_ELEMS", "200000")):
        if any(x in shape for x in (2048, 4096, 8192, 16384)):
            return True

    return False


def _iter_tensors(obj, prefix):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}_{i}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{prefix}_{k}")


def _tensor_stats(t):
    x = t.detach().float()
    return {
        "shape": tuple(int(v) for v in x.shape),
        "dtype": str(t.dtype),
        "rms": float(torch.sqrt(torch.mean(x * x)).item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()) if x.numel() > 1 else 0.0,
        "max_abs": float(x.abs().max().item()),
    }


def _save_tensor(name, t):
    if not _should_write() or not torch.is_tensor(t) or not _interesting_tensor(t):
        return False

    max_elems = int(os.getenv("LTX_BLOCK0_MID_MAX_ELEMS", "50000000"))
    if t.numel() > max_elems:
        print(f"[block0_mid_dump] skip {name}, numel={t.numel()} > {max_elems}", flush=True)
        return False

    p = _out_dir() / f"{_safe(name)}.pt"
    torch.save(t.detach().cpu(), p)

    rec = {"file": p.name, "name": name, **_tensor_stats(t)}
    with open(_out_dir() / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[block0_mid_dump] saved {p.name} shape={tuple(t.shape)} dtype={t.dtype}", flush=True)
    return True


def install_block0_mid_tensor_dump(model):
    global _INSTALLED

    if _INSTALLED or not _enabled():
        return 0

    block_id = int(os.getenv("LTX_BLOCK0_MID_DUMP_BLOCK", "0"))
    suffix = f"transformer_blocks.{block_id}"
    block = None
    block_name = None

    for name, module in model.named_modules():
        if name.lower().endswith(suffix):
            block = module
            block_name = name
            break

    if block is None:
        print(f"[block0_mid_dump] target block not found: {suffix}", flush=True)
        return 0

    if hasattr(model, "audio_args_preprocessor"):
        prep = model.audio_args_preprocessor
        if not hasattr(prep, "_block0_mid_original_prepare"):
            prep._block0_mid_original_prepare = prep.prepare

            def _prepare_with_dump(modality, cross_modality=None):
                _save_tensor("0001_audio_prepare_positions", getattr(modality, "positions", None))
                _save_tensor("0002_audio_prepare_latent", getattr(modality, "latent", None))
                _save_tensor("0003_audio_prepare_timesteps", getattr(modality, "timesteps", None))
                out = prep._block0_mid_original_prepare(modality, cross_modality)
                pe = getattr(out, "positional_embeddings", None)
                if isinstance(pe, tuple):
                    for i, t in enumerate(pe):
                        _save_tensor(f"0004_audio_prepare_pe_{i}", t)
                else:
                    _save_tensor("0004_audio_prepare_pe", pe)
                return out

            prep.prepare = _prepare_with_dump

    if _should_write():
        for p in _out_dir().glob("*.pt"):
            p.unlink()
        mf = _out_dir() / "manifest.jsonl"
        if mf.exists():
            mf.unlink()

    state = {"call_id": 0, "saved": 0}
    max_saves = int(os.getenv("LTX_BLOCK0_MID_MAX_SAVES", "300"))

    def make_pre_hook(mod_name):
        def h(module, args, kwargs=None):
            if state["saved"] >= max_saves:
                return
            state["call_id"] += 1
            cid = state["call_id"]
            for tn, t in _iter_tensors(args, "args"):
                if _save_tensor(f"{cid:04d}_{mod_name}_pre_{tn}", t):
                    state["saved"] += 1
                    if state["saved"] >= max_saves:
                        return
            if kwargs:
                for tn, t in _iter_tensors(kwargs, "kwargs"):
                    if _save_tensor(f"{cid:04d}_{mod_name}_pre_{tn}", t):
                        state["saved"] += 1
                        if state["saved"] >= max_saves:
                            return
        return h

    def make_post_hook(mod_name):
        def h(module, args, kwargs_or_output, output=None):
            if output is None:
                output = kwargs_or_output
            if state["saved"] >= max_saves:
                return
            state["call_id"] += 1
            cid = state["call_id"]
            for tn, t in _iter_tensors(output, "output"):
                if _save_tensor(f"{cid:04d}_{mod_name}_post_{tn}", t):
                    state["saved"] += 1
                    if state["saved"] >= max_saves:
                        return
        return h

    installed = 0
    for child_name, child in block.named_modules():
        full_name = block_name if child_name == "" else f"{block_name}.{child_name}"
        try:
            child.register_forward_pre_hook(make_pre_hook(full_name), with_kwargs=True)
            child.register_forward_hook(make_post_hook(full_name), with_kwargs=True)
        except TypeError:
            child.register_forward_pre_hook(lambda m, a, fn=make_pre_hook(full_name): fn(m, a, {}))
            child.register_forward_hook(lambda m, a, o, fn=make_post_hook(full_name): fn(m, a, o))
        installed += 1

    _INSTALLED = True
    print(
        f"[block0_mid_dump] installed {installed} hooks on {block_name}, "
        f"local_rank={_local_rank()}, write={_should_write()}",
        flush=True,
    )
    return installed
