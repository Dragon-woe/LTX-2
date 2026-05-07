import sys
from pathlib import Path

import torch


d1 = Path(sys.argv[1])
d4 = Path(sys.argv[2])


def rms(x):
    x = x.float()
    return torch.sqrt(torch.mean(x * x)).item()


def token_dim(shape):
    for i, s in enumerate(shape):
        if int(s) == 126:
            return i
    return None


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


rows = []

for p1 in sorted(d1.glob("*.pt")):
    p4 = d4 / p1.name
    if not p4.exists():
        continue

    a = torch.load(p1, map_location="cpu")
    b = torch.load(p4, map_location="cpu")

    if not torch.is_tensor(a) or not torch.is_tensor(b):
        continue

    if tuple(a.shape) != tuple(b.shape):
        rows.append((999, p1.name, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"))
        continue

    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()

    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    rms_a = rms(af)
    rms_b = rms(bf)
    rms_diff = rms(af - bf)
    rel = rms_diff / (rms_a + 1e-8)
    cos = cosine(af, bf)

    detail = (
        f"shape={tuple(a.shape)} dtype={a.dtype}/{b.dtype} "
        f"rms d1={rms_a:.6f} d4={rms_b:.6f} "
        f"diff_rms={rms_diff:.6f} rel={rel:.6f} "
        f"mean_abs={mean_abs:.6f} max_abs={max_abs:.6f} cos={cos:.8f}"
    )

    td = token_dim(a.shape)
    if td is not None:
        toks = []
        for tok in (15, 16, 77, 78):
            if tok < a.shape[td]:
                aa = af.select(td, tok)
                bb = bf.select(td, tok)
                dd = aa - bb
                toks.append(
                    f"tok{tok}:d1={rms(aa):.6f},d4={rms(bb):.6f},"
                    f"diff={rms(dd):.6f},rel={rms(dd)/(rms(aa)+1e-8):.6f}"
                )
        detail += "\n    " + "\n    ".join(toks)

    rows.append((rel, p1.name, detail))

rows.sort(key=lambda x: x[0], reverse=True)

print("===== ALL TENSOR DIFFS, sorted by rel diff =====")
for rel, name, detail in rows:
    print()
    print(name)
    print(detail)

print("\n===== KEY SUMMARY =====")
keys = [
    "target_pre",
    "to_q_post",
    "to_k_post",
    "to_v_post",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_out",
    "target_post",
    "to_out",
]
for k in keys:
    hits = [r for r in rows if k in r[1]]
    if hits:
        rel, name, detail = hits[0]
        print(f"{k}: rel={rel:.6f} file={name}")
