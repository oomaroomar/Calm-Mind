import torch, platform, sys
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)   # will be None on ROCm
print("torch.version.hip  =", getattr(torch.version, "hip", None))
print("is_available       =", torch.cuda.is_available())
print("device_count       =", torch.cuda.device_count())
print("python             =", sys.version)
print("os                 =", platform.platform())
try:
    if torch.cuda.is_available():
        print("device0 name        =", torch.cuda.get_device_name(0))
except Exception as e:
    print("get_device_name error:", e)
