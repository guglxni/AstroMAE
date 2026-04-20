"""Device selection."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    Precedence: explicit ``preferred`` argument > CUDA > MPS > CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
