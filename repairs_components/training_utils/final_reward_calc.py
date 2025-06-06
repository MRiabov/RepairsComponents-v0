"""Reward calculation in RepairsEnv:
To demonstrate where the maintenance goal to the model - where the model must end the repair,
the \textit{desired state} is introduced. The desired state is a modified subset of the initial assembly
which demonstrates how things should be - e.g. in an electronics circuit, a certain set of
lamps must be ignited when the power is supplied, which does not happen at the moment
"""

import torch
from repairs_components.training_utils.sim_state_global import RepairsSimState


# TODO: get the actual values from the mujoco state.
def calculate_reward_and_done(
    current_state: RepairsSimState,
    desired_state: RepairsSimState,
    initial_diff_count: int,
    reward_multiplier: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """returns:
    - Reward [ml_batch_dim,]
    - terminated [ml_batch_dim, ]"""
    diff_count = current_state.diff(desired_state)
    completion_rate = calculate_partial_reward(initial_diff_count, diff_count)
    return completion_rate * reward_multiplier, completion_rate >= 1.0


def calculate_partial_reward(
    count_initial_delta, count_final_delta, partial_multiplier=0.5
):
    completion_percentage = caclulate_completion_percentage(
        count_initial_delta, count_final_delta
    )
    if completion_percentage == 1.0:
        return 1.0
    elif completion_percentage > 0:
        return completion_percentage * partial_multiplier
    else:
        return 0.0


def caclulate_completion_percentage(count_initial_delta, count_final_delta):
    return (count_initial_delta - count_final_delta) / count_initial_delta
