from dataclasses import field
import torch
from tensordict import TensorClass


class LedInfo(TensorClass):
    component_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    vf_drop: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )


class LedState(TensorClass):
    # [B, N_led]
    luminosity_pct: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float32)
    )