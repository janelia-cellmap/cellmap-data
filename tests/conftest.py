import os
import torch


def pytest_configure():
    # Force torch to avoid MPS (failing in GitHub CI)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.backends.mps.is_available = lambda: False
