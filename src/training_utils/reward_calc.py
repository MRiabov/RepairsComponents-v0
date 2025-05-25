"""Reward calculation in RepairsEnv:
To demonstrate where the maintenance goal to the model - where the model must end the repair,
the \textit{desired state} is introduced. The desired state is a modified subset of the initial assembly
which demonstrates how things should be - e.g. in an electronics circuit, a certain set of
lamps must be ignited when the power is supplied, which does not happen at the moment
"""

from logic.electronics.electronics_state import ElectronicsState
from logic.physical_state import PhysicalState
from logic.fluid_state import FluidState


# TODO: get the actual values from the mujoco state.
def calculate_reward(
    current_state, desired_state, initial_diff_count, reward_multiplier: float = 10.0
):
    diff_count = calculate_diff(current_state, desired_state)
    return calculate_partial_reward(initial_diff_count, diff_count) * reward_multiplier


def calculate_diff(
    current_state: tuple[ElectronicsState, PhysicalState, FluidState],
    desired_state: tuple[ElectronicsState, PhysicalState, FluidState],
):
    electronics_state, physical_state, fluid_state = current_state
    electronics_state_desired, physical_state_desired, fluid_state_desired = (
        desired_state
    )
    electronics_state_diff, electronics_state_total_changes = (
        calculate_electronics_completion(electronics_state, electronics_state_desired)
    )
    physical_state_diff, physical_state_total_changes = calculate_physical_completion(
        physical_state, physical_state_desired
    )
    fluid_state_diff, fluid_state_total_changes = calculate_fluid_completion(
        fluid_state, fluid_state_desired
    )
    return (
        electronics_state_total_changes
        + physical_state_total_changes
        + fluid_state_total_changes
    )


def calculate_electronics_completion(
    electronics_state: ElectronicsState, electronics_state_desired: ElectronicsState
):
    electronics_state_diff, electronics_state_total_changes = electronics_state.diff(
        electronics_state_desired
    )
    return electronics_state_diff, electronics_state_total_changes


def calculate_physical_completion(
    physical_state: PhysicalState, physical_state_desired: PhysicalState
):
    physical_state_diff, physical_state_total_changes = physical_state.diff(
        physical_state_desired
    )
    return physical_state_diff, physical_state_total_changes


def calculate_fluid_completion(fluid_state, fluid_state_desired):
    fluid_state_diff, fluid_state_total_changes = fluid_state.diff(fluid_state_desired)
    return fluid_state_diff, fluid_state_total_changes


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
