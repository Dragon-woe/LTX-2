import os, json
from pathlib import Path
import torch

def _enabled(): return os.environ.get("LTX_AUDIO_ATTN2_PROBE","0")=="1"
def _lrank(): return int(os.environ.get("LOCAL_RANK",os.environ.get("RANK","0")))
def _should_write(): return _enabled() and _lrank()==int(os.environ.get("LTX_AUDIO_ATTN2_PROBE_LOCAL_RANK","0"))
def _path():
    d=Path(os.environ.get("LTX_AUDIO_ATTN2_PROBE_DIR","/tmp/probe"))
    d.mkdir(parents=True,exist_ok=True)
    return d/f"probe_rank{_lrank()}.jsonl"

def _token_dim(shape):
    for i,s in enumerate(shape):
        if 80<=int(s)<=256: return i
    return None

def _tensor_stats(t):
    if not torch.is_tensor(t) or t.numel()==0 or t.ndim<2: return None
    x=t.detach().float(); shape=tuple(int(v) for v in x.shape); td=_token_dim(shape)
    rec={"shape":shape,"global_rms":float(torch.sqrt(torch.mean(x*x)).item()),"mean":float(x.mean().item()),"std":float(x.std().item()) if x.numel()>1 else 0,"max_abs":float(x.abs().max().item())}
    if td is not None:
        rec["token_dim"]=int(td); rec["tokens"]={}
        for tok in (15,16,77,78):
            if tok<shape[td]:
                xs=x.select(td,tok)
                rec["tokens"][str(tok)]={"rms":float(torch.sqrt(torch.mean(xs*xs)).item()),"mean":float(xs.mean().item()),"std":float(xs.std().item()) if xs.numel()>1 else 0,"max_abs":float(xs.abs().max().item())}
    return rec

def _iter_tensors(obj,prefix):
    if torch.is_tensor(obj): yield prefix,obj
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): yield from _iter_tensors(v,f"{prefix}[{i}]")
    elif isinstance(obj,dict):
        for k,v in obj.items(): yield from _iter_tensors(v,f"{prefix}.{k}")
    elif hasattr(obj,"__dict__"):
        for k,v in vars(obj).items():
            if k.startswith("_"): continue
            yield from _iter_tensors(v,f"{prefix}.{k}")

def _write(rec):
    if not _should_write(): return
    with open(_path(),"a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")

def _param_stats(module):
    out={}
    for name,p in module.named_parameters(recurse=True):
        if not torch.is_tensor(p) or p.numel()==0: continue
        x=p.detach().float()
        out[name]={"shape":tuple(int(v) for v in x.shape),"rms":float(torch.sqrt(torch.mean(x*x)).item()),"mean":float(x.mean().item()),"std":float(x.std().item()) if x.numel()>1 else 0,"max_abs":float(x.abs().max().item())}
        if len(out)>=16: break
    return out

def install_audio_attn2_probe(model):
    if not _enabled(): return 0
    if _should_write():
        p=_path(); p.unlink(missing_ok=True)
    target=None; target_name=None
    for name,module in model.named_modules():
        if name.lower().endswith("transformer_blocks.0.audio_attn2"):
            target=module; target_name=name; break
    if target is None:
        print("[attn2_probe] target not found",flush=True); return 0
    def pre_hook(module,args,kwargs=None):
        _write({"event":"pre_params","module":target_name,"params":_param_stats(module)})
        for tn,t in _iter_tensors(args,"args"):
            st=_tensor_stats(t)
            if st: _write({"event":"pre","module":target_name,"tensor":tn,**st})
        if kwargs:
            for tn,t in _iter_tensors(kwargs,"kwargs"):
                st=_tensor_stats(t)
                if st: _write({"event":"pre","module":target_name,"tensor":tn,**st})
    def post_hook(module,args,kwargs,output):
        for tn,t in _iter_tensors(output,"output"):
            st=_tensor_stats(t)
            if st: _write({"event":"post","module":target_name,"tensor":tn,**st})
    try:
        target.register_forward_pre_hook(pre_hook,with_kwargs=True)
        target.register_forward_hook(post_hook,with_kwargs=True)
    except TypeError:
        target.register_forward_pre_hook(lambda m,a: pre_hook(m,a,{}))
        target.register_forward_hook(lambda m,a,o: post_hook(m,a,{},o))
    print(f"[attn2_probe] installed on {target_name} rank={_lrank()}",flush=True)
    return 1
