import sys, platform, os
print("python:", sys.version.splitlines()[0])
try:
    import torch
    print("torch.__version__:", torch.__version__)
    # torch.version.hip exists on ROCm builds; torch.version.cuda on CUDA builds
    print("torch.version.hip:", getattr(torch.version, "hip", None))
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    try:
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch.cuda.* check error:", repr(e))
    # show any backend availability attributes
    print("torch.backends.cudnn.enabled:", getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled)
    print("torch.backends.mps.is_available():", getattr(torch.backends, "mps", None) and getattr(torch.backends.mps, "is_available()", "NA"))
except Exception as e:
    print("import torch failed:", repr(e))
