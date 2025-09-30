import torch
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.cuda.get_device_name(0)
try:
 x = torch.rand((100, 100), device=device)
 print("GPU tensor creation successful.")
except RuntimeError as e:
 print(f"Error: {e}")
