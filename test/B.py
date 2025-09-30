import torch,os,glob,subprocess
p=os.path.dirname(torch.__file__)
candidates = glob.glob(os.path.join(p,"**","*_C*.so"), recursive=True) + glob.glob(os.path.join(p,"**","*.so"), recursive=True)
candidates = [c for c in candidates if "/test" not in c and "python" not in c]
# try the most likely ones first
for so in sorted(candidates)[:30]:
    try:
        out = subprocess.check_output(["ldd", so], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        continue
    if ("amdhip64" in out) or ("libamdhip64" in out) or ("hiprtc" in out) or ("hip_hcc" in out) or ("libhip" in out):
        print("== linked libs for:", so)
        print(out)
