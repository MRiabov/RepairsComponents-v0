from dataclasses import dataclass

import torch
from repairs_components.logic.tools.tool import Tool


@dataclass
class Screwdriver(Tool):
    picked_up_fastener_name: str | None = None
    picked_up_fastener: bool = False


def receive_screw_in_action(actions: torch.Tensor):
    assert actions.shape[1] == 9, (
        "Screwdriver action check expects that action has shape [batch, 9]"
    )
    assert actions.ndim == 2, (
        "Screwdriver action check expects that action has shape [batch, action_dim]"
    )
    screw_in = actions[:, 8]
    assert screw_in.ndim == 1, (
        "Screwdriver action check expects that action has shape [batch]"
    )
    return screw_in > 0.5
