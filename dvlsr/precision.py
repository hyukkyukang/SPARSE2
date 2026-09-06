"""Mixed precision that follows the GPU.

The study ran under bf16 autocast on A100s. Turing cards (TITAN RTX) have no native
bf16, so there we use fp16 autocast with dynamic loss scaling. `DVLSR_AMP` overrides
(bf16 | fp16 | fp32).
"""
from __future__ import annotations
import contextlib, os
import torch


def autocast_dtype():
    forced = os.environ.get("DVLSR_AMP", "").lower()
    if forced == "fp32":
        return torch.float32
    if forced == "fp16":
        return torch.float16
    if forced == "bf16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast():
    dt = autocast_dtype()
    if dt == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dt)


def grad_scaler():
    """Loss scaling is only needed (and only enabled) for fp16."""
    return torch.amp.GradScaler("cuda", enabled=autocast_dtype() == torch.float16)
