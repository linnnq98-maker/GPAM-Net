from __future__ import annotations

import inspect
import sys

import torch

try:
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")
except Exception as error:
    print(f"CUDA initialization warning: {error!r}")

try:
    import mamba_ssm
except Exception as error:
    print(f"mamba_ssm import failed: {error!r}")
    raise SystemExit(2)


print(f"python: {sys.executable}")
print(f"torch: {torch.__version__}")
print(f"torch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"mamba_ssm: {mamba_ssm.__file__}")
print(f"Mamba: {inspect.signature(mamba_ssm.Mamba)}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Python environment")
