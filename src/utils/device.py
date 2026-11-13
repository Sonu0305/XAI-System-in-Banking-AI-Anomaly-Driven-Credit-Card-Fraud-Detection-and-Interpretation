"""
Hardware acceleration and device management utilities.
Optimized for dual NVIDIA T4 GPUs on Kaggle while maintaining seamless CPU fallback.
"""

from contextlib import contextmanager, nullcontext
import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set global seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """
    Selects the optimal execution device.
    If 'auto', chooses CUDA if available, else CPU.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Returns metadata about available compute resources."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "is_dual_gpu": device_count >= 2,
    }


def prepare_model_for_device(model: nn.Module, device: torch.device, parallelize: bool = True) -> nn.Module:
    """
    Moves model to device, and wraps in DataParallel if multiple GPUs are available (e.g. Kaggle T4 x2).
    """
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and parallelize:
        model = nn.DataParallel(model)
    return model


def get_amp_context(device: torch.device, enabled: bool = True):
    """
    Returns an AMP autocast context manager if CUDA is available and enabled.
    """
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    return nullcontext()


def get_dataloader_kwargs(device: torch.device, num_workers: int = 2) -> dict:
    """
    Returns optimal DataLoader keyword arguments for the compute environment.
    """
    is_cuda = device.type == "cuda"
    return {
        "num_workers": num_workers if is_cuda else 0,
        "pin_memory": is_cuda,
        "persistent_workers": (num_workers > 0 and is_cuda),
    }

