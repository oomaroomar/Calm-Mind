import torch
print("torch:", torch.__version__, getattr(torch.version,'hip',None))
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
x = torch.randn(10, device='cuda' if torch.cuda.is_available() else 'cpu')
print("ok:", x.mean().item())
