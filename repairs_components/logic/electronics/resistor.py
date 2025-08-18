from dataclasses import field
import torch
from tensordict import TensorClass


class ResistorInfo(TensorClass):
    component_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    resistance: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
