import re
import sys
from pathlib import Path

import torch


d1 = Path(sys.argv[1])
d4 = Path(sys.argv[2])


def rms(x):
    x = x.float()
    return torch.sqrt(torch.mean(x * x)).item()


def rel_diff(a, b):
    af = a.float()
    bf = b.float()
    return rms(af - bf) / (rms(af) + 1e-8)


def token_dim(shape):
    for i, s in enumerate(shape):
        if int(s) == 126:
            return i
    return None


def parse_call_id(name):
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else 999999


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
        rows.append({
            "file": p1.name,
            "call": parse_call_id(p1.name),
            "rel": 999.0,
            "detail": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}",
        })
        continue

    rd = rel_diff(a, b)
    detail = (
        f"shape={tuple(a.shape)} dtype={a.dtype}/{b.dtype} "
        f"rms_d1={rms(a):.6f} rms_d4={rms(b):.6f} "
        f"diff_rms={rms(a.float() - b.float()):.6f} rel={rd:.6f}"
    )

    td = token_dim(a.shape)
    if td is not None:
        toks = []
        for tok in (15, 16, 77, 78):
            if tok < a.shape[td]:
                aa = a.float().select(td, tok)
                bb = b.float().select(td, tok)
                toks.append(
                    f"tok{tok}: d1={rms(aa):.6f}, d4={rms(bb):.6f}, "
                    f"diff={rms(aa - bb):.6f}, rel={rms(aa - bb) / (rms(aa) + 1e-8):.6f}"
                )
        if toks:
            detail += "\n    " + "\n    ".join(toks)

    rows.append({"file": p1.name, "call": parse_call_id(p1.name), "rel": rd, "detail": detail})


rows_by_call = sorted(rows, key=lambda x: (x["call"], x["file"]))
rows_by_rel = sorted(rows, key=lambda x: x["rel"], reverse=True)

print("===== FIRST DIFF BY CALL ORDER rel > 1e-4 =====")
cnt = 0
for r in rows_by_call:
    if r["rel"] > 1e-4:
        print()
        print(f"call={r['call']} rel={r['rel']:.6f} file={r['file']}")
        print(r["detail"])
        cnt += 1
        if cnt >= 80:
            break

print("\n===== FIRST STRONG DIFF BY CALL ORDER rel > 1e-2 =====")
cnt = 0
for r in rows_by_call:
    if r["rel"] > 1e-2:
        print()
        print(f"call={r['call']} rel={r['rel']:.6f} file={r['file']}")
        print(r["detail"])
        cnt += 1
        if cnt >= 50:
            break

print("\n===== TOP 80 DIFF BY REL =====")
for r in rows_by_rel[:80]:
    print()
    print(f"call={r['call']} rel={r['rel']:.6f} file={r['file']}")
    print(r["detail"])
