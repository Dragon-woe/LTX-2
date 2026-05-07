import os, json, math
from pathlib import Path
import torch

_INSTALLED = False
_GLOBAL_CALL_ID = 0

def _enabled(): return os.environ.get("LTX_AUDIO_TRACE","0")=="1"
def _rank(): return int(os.environ.get("RANK",os.environ.get("LOCAL_RANK","0")))
def _out_path():
    d=Path(os.environ.get("LTX_AUDIO_TRACE_DIR","/tmp/audio_trace"))
    d.mkdir(parents=True,exist_ok=True)
    return d/f"audio_trace_rank{_rank()}.jsonl"

def _find_token_dim(shape):
    for i,s in enumerate(shape):
        if 80<=int(s)<=256: return i
    return None

def _tensor_stats(t):
    if not torch.is_tensor(t) or t.numel()==0 or t.ndim<2: return None
    shape=tuple(int(x) for x in t.shape)
    td=_find_token_dim(shape)
    if td is None: return None
    x=t.detach().float()
    gr=torch.sqrt(torch.mean(x*x)).item()
    ts={}
    for tok in (15,16,77,78):
        if tok<shape[td]:
            xs=x.select(td,tok)
            ts[str(tok)]={"rms":torch.sqrt(torch.mean(xs*xs)).item(),"mean":torch.mean(xs).item(),"std":torch.std(xs).item() if xs.numel()>1 else 0,"max_abs":torch.max(torch.abs(xs)).item()}
    return {"shape":shape,"token_dim":int(td),"global_rms":float(gr),"tokens":ts}

def _iter_tensors(obj,prefix=""):
    if torch.is_tensor(obj): yield prefix or "tensor",obj
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): yield from _iter_tensors(v,f"{prefix}[{i}]")
    elif isinstance(obj,dict):
        for k,v in obj.items(): yield from _iter_tensors(v,f"{prefix}.{k}" if prefix else str(k))
    elif hasattr(obj,"__dict__"):
        for k,v in vars(obj).items():
            if k.startswith("_"): continue
            yield from _iter_tensors(v,f"{prefix}.{k}" if prefix else str(k))

def _write_record(rec):
    with open(_out_path(),"a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")

def _make_pre_hook(name):
    def hook(module,inputs):
        global _GLOBAL_CALL_ID; _GLOBAL_CALL_ID+=1
        maxn=int(os.environ.get("LTX_AUDIO_TRACE_MAX_TENSORS","4")); cnt=0
        for tn,t in _iter_tensors(inputs,"input"):
            st=_tensor_stats(t)
            if st:
                _write_record({"rank":_rank(),"call_id":_GLOBAL_CALL_ID,"module":name,"event":"pre","tensor":tn,**st})
                cnt+=1
                if cnt>=maxn: break
    return hook

def _make_post_hook(name):
    def hook(module,inputs,output):
        global _GLOBAL_CALL_ID; _GLOBAL_CALL_ID+=1
        maxn=int(os.environ.get("LTX_AUDIO_TRACE_MAX_TENSORS","4")); cnt=0
        for tn,t in _iter_tensors(output,"output"):
            st=_tensor_stats(t)
            if st:
                _write_record({"rank":_rank(),"call_id":_GLOBAL_CALL_ID,"module":name,"event":"post","tensor":tn,**st})
                cnt+=1
                if cnt>=maxn: break
    return hook

def install_audio_trace_hooks(model):
    global _INSTALLED
    if _INSTALLED or not _enabled(): return 0
    _out_path().unlink(missing_ok=True)
    installed=0
    for name,module in model.named_modules():
        nl=name.lower()
        if "audio" not in nl: continue
        if not any(k in nl for k in ("audio_attn1","audio_attn2","attn1","attn2","ff","ffn","mlp","feed_forward","norm","block")): continue
        module.register_forward_pre_hook(_make_pre_hook(name))
        module.register_forward_hook(_make_post_hook(name))
        installed+=1
    _INSTALLED=True
    _write_record({"rank":_rank(),"event":"trace_installed","installed_hooks":installed})
    print(f"[audio_trace] installed {installed} hooks rank={_rank()}",flush=True)
    return installed
