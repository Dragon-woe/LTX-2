import os
import re
from pathlib import Path

import torch


_INSTALLED = False


def _enabled():
    return os.getenv("LTX_ATTN2_TENSOR_DUMP", "0") == "1"


def _local_rank():
    return int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))


def _should_write():
    return _enabled() and _local_rank() == int(os.getenv("LTX_ATTN2_TENSOR_DUMP_LOCAL_RANK", "0"))


def _out_dir():
    d = Path(os.getenv("LTX_ATTN2_TENSOR_DUMP_DIR", "/tmp/attn2_tensor_dump"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _save(name, t):
    if not _should_write() or not torch.is_tensor(t):
        return
    max_elems = int(os.getenv("LTX_ATTN2_TENSOR_DUMP_MAX_ELEMS", "50000000"))
    if t.numel() > max_elems:
        print(f"[attn2_tensor_dump] skip {name}, numel={t.numel()} > {max_elems}", flush=True)
        return

    p = _out_dir() / f"{_safe(name)}.pt"
    try:
        torch.save(t.detach().cpu(), p)
        print(f"[attn2_tensor_dump] saved {p} shape={tuple(t.shape)} dtype={t.dtype}", flush=True)
    except Exception as e:
        print(f"[attn2_tensor_dump] save failed {name}: {e}", flush=True)


def _iter_tensors(obj, prefix):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}_{i}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{prefix}_{k}")


def install_audio_attn2_tensor_dump(model):
    global _INSTALLED

    if _INSTALLED or not _enabled():
        return 0

    block = int(os.getenv("LTX_ATTN2_TENSOR_DUMP_BLOCK", "0"))
    suffix = f"transformer_blocks.{block}.audio_attn2"
    target = None
    target_name = None

    for name, module in model.named_modules():
        if name.lower().endswith(suffix):
            target = module
            target_name = name
            break

    if target is None:
        print(f"[attn2_tensor_dump] target not found: {suffix}", flush=True)
        return 0

    if _should_write():
        for p in _out_dir().glob("*.pt"):
            p.unlink()

    state = {"target_pre": 0, "target_post": 0, "attn": 0, "child": {}}

    def target_pre_hook(module, args, kwargs=None):
        if state["target_pre"] >= 1:
            return
        state["target_pre"] += 1
        for n, t in _iter_tensors(args, "target_pre_args"):
            _save(n, t)
        if kwargs:
            for n, t in _iter_tensors(kwargs, "target_pre_kwargs"):
                _save(n, t)

    def target_post_hook(module, args, kwargs, output):
        if state["target_post"] >= 1:
            return
        state["target_post"] += 1
        for n, t in _iter_tensors(output, "target_post_output"):
            _save(n, t)

    try:
        target.register_forward_pre_hook(target_pre_hook, with_kwargs=True)
        target.register_forward_hook(target_post_hook, with_kwargs=True)
    except TypeError:
        target.register_forward_pre_hook(lambda m, a: target_pre_hook(m, a, {}))
        target.register_forward_hook(lambda m, a, o: target_post_hook(m, a, {}, o))

    interesting = ("to_q", "to_k", "to_v", "to_out")
    for cname, child in target.named_modules():
        if cname == "":
            continue
        if not any(cname == x or cname.startswith(x + ".") for x in interesting):
            continue

        state["child"][cname] = {"pre": 0, "post": 0}

        def make_pre(cn):
            def h(module, args):
                if state["child"][cn]["pre"] >= 1:
                    return
                state["child"][cn]["pre"] += 1
                for n, t in _iter_tensors(args, f"{cn}_pre"):
                    _save(n, t)
            return h

        def make_post(cn):
            def h(module, args, output):
                if state["child"][cn]["post"] >= 1:
                    return
                state["child"][cn]["post"] += 1
                for n, t in _iter_tensors(output, f"{cn}_post"):
                    _save(n, t)
            return h

        child.register_forward_pre_hook(make_pre(cname))
        child.register_forward_hook(make_post(cname))

    if hasattr(target, "attention_function"):
        orig = target.attention_function

        def wrapped_attention(q, k, v, heads, mask=None, *args, **kwargs):
            if state["attn"] < 1:
                _save("attention_q", q)
                _save("attention_k", k)
                _save("attention_v", v)
                if mask is not None and torch.is_tensor(mask):
                    _save("attention_mask", mask)
            out = orig(q, k, v, heads, mask, *args, **kwargs)
            if state["attn"] < 1:
                _save("attention_out", out)
                state["attn"] += 1
            return out

        target.attention_function = wrapped_attention

    _INSTALLED = True
    print(
        f"[attn2_tensor_dump] installed on {target_name}, "
        f"local_rank={_local_rank()}, write={_should_write()}",
        flush=True,
    )
    return 1
