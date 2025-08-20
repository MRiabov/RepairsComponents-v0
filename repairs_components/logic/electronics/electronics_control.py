from dataclasses import field

import torch
from tensordict import TensorClass


class SwitchInfo(TensorClass):
    component_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    on_res_ohm: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    off_res_ohm: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    # Control relations (indexed in global component id space)
    control_from_component: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    control_type: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    control_threshold: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    control_hysteresis: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )


class SwitchState(TensorClass):
    # [B, Ns]
    state_bits: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.bool)
    )
