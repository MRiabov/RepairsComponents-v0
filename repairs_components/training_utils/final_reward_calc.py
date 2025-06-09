"""Reward calculation in RepairsEnv:
To demonstrate where the maintenance goal to the model - where the model must end the repair,
the \textit{desired state} is introduced. The desired state is a modified subset of the initial assembly
which demonstrates how things should be - e.g. in an electronics circuit, a certain set of
lamps must be ignited when the power is supplied, which does not happen at the moment
"""

import torch
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)


# TODO: get the actual values from the mujoco state.
def calculate_reward_and_done(
    scene_data: ConcurrentSceneData,
    reward_multiplier: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """returns:
    - Reward [ml_batch_dim,]
    - terminated [ml_batch_dim,]"""
    _diff, diff_count = scene_data.current_state.diff(scene_data.desired_state)
    completion_rate = calculate_partial_reward(
        scene_data.initial_diff_counts, diff_count
    )
    return completion_rate * reward_multiplier, completion_rate >= 1.0


def calculate_partial_reward(
    initial_diff_count: torch.Tensor,
    final_diff_count: torch.Tensor,
    partial_multiplier: float = 0.5,
) -> torch.Tensor:
    """
    Computes final reward. Uses this logic, but batched:
    ```
    if completion_percentage == 1.0:
        return torch.ones_like(completion_percentage)
    elif completion_percentage > 0:
        return completion_percentage * partial_multiplier
    else:
        return torch.zeros_like(completion_percentage)
    ```
    """
    completion_percentage = calculate_completion_percentage(
        initial_diff_count, final_diff_count
    )
    return torch.where(
        completion_percentage == 1.0,
        torch.ones_like(completion_percentage),
        torch.where(
            completion_percentage > 0,
            completion_percentage * partial_multiplier,
            torch.zeros_like(completion_percentage),
        ),
    )

    



def calculate_completion_percentage(
    initial_diff_count: torch.Tensor, final_diff_count: torch.Tensor
) -> torch.Tensor:
    return (initial_diff_count - final_diff_count) / initial_diff_count
