import torch, os, glob,sys
p = os.path.dirname(torch.__file__)
print("torch installation dir:", p)
so_files = glob.glob(os.path.join(p, "**", "*.so"), recursive=True) + glob.glob(os.path.join(p, "*.so"))
# show a few example .so paths (first 10)
for s in so_files[:20]:
    print("SO:", s)
# Search for gfx tokens inside them (this may be long — we limit output)
import subprocess, shlex
found = set()
for so in so_files:
    try:
        out = subprocess.check_output(["strings", so], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        continue
    for token in ("gfx", "amdgcn", "amdgpu", "gfx1100"):
        if token in out:
            found.add((so, token))
# print matches (up to 40 lines)
for i,(so,tok) in enumerate(sorted(found)[:40]):
    print(i+1, tok, so)
print("done")
