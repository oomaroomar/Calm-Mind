import torch

print("torch:", torch.__version__, getattr(torch.version, "hip", None))
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())

# Try a small tensor on the GPU (this is the operation that previously raised the HIP error)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(10, device=device)
print("ok:", x.mean().item())

