"""Utility helpers for resolving the compute device (CPU/GPU)."""

from __future__ import annotations

from typing import Optional, Union

import torch


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve the target torch.device based on user input and CUDA availability.

    Args:
        device: Optional device specification. Can be a string (e.g. "cuda", "cpu"),
            a torch.device instance, or None (auto-detect).

    Returns:
        torch.device: The resolved device, preferring CUDA when available.
    """
    if isinstance(device, torch.device):
        return device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device)

