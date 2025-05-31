"""Reward calculation in RepairsEnv:
To demonstrate where the maintenance goal to the model - where the model must end the repair,
the \textit{desired state} is introduced. The desired state is a modified subset of the initial assembly
which demonstrates how things should be - e.g. in an electronics circuit, a certain set of
lamps must be ignited when the power is supplied, which does not happen at the moment

NOTE: progressive reward calc is different because it is denser, which helps learning the model better.

Give the reward whenever:
1. The fastener is inserted into the model in the correct spot and stood there for at least 2 seconds,
2. The reward for placing a part in a correct place, and it holds for at least 3 seconds.

In the end, if the environment is timed out, give penalty for half of the completed tasks.
"""

from repairs_components.training_utils.sim_state_global import RepairsSimState
import numpy as np


def calculate_reward(
    current_state: RepairsSimState,
    desired_state: RepairsSimState,
    initial_diff: dict[str, np.ndarray],
    diff_with_last: dict[str, np.ndarray],
    reward_multiplier: float = 10.0,
    queue_of_successes: list[dict[str, np.ndarray]] = [],
):
    diff_count = current_state.diff(desired_state)
    return calculate_partial_reward(initial_diff_count, diff_count) * reward_multiplier


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
