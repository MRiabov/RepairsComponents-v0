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

from dataclasses import dataclass, field
import enum
from typing import Callable
from repairs_components.training_utils.concurrent_scene_dataclass import ConcurrentSceneData
from repairs_components.training_utils.sim_state_global import RepairsSimState
import numpy as np


class RewardType(enum.Enum):
    "An enum of reward types to their delays and rewards to check the rewards: give a reward in value timesteps"

    FASTENER_INSERTION = (200, 2.0)
    PART_PLACEMENT = (200, 3.0)


@dataclass
class RewardHistory:  # note: this is stored per every environment yet, not batched.
    sim_id: int
    triggered_reward_per_future_timestep: dict[int, list[RewardType]] = field(
        default_factory=dict
    )
    reward_check_data: dict[int, tuple] = field(default_factory=dict)
    already_triggered: dict[str, list] = field(
        default_factory=lambda: {
            RewardType.FASTENER_INSERTION.name: [],
            RewardType.PART_PLACEMENT.name: [],
        }
    )
    "For every reward type, store which reward were already triggered."

    def register_correct_fastener_insertion(
        self, current_timestep: int, fastener_id: int, to_body_id: int
    ):
        self.triggered_reward_per_future_timestep[
            RewardType.FASTENER_INSERTION.value[0] + current_timestep
        ] = [RewardType.FASTENER_INSERTION]
        self.reward_check_data[current_timestep] = (fastener_id, to_body_id)

    def register_correct_part_placement(self, current_timestep: int, part_id: int):
        self.triggered_reward_per_future_timestep[
            RewardType.PART_PLACEMENT.value[0] + current_timestep
        ] = (RewardType.PART_PLACEMENT,)
        self.reward_check_data[current_timestep] = part_id

    def calculate_reward_this_timestep(
        self,
        current_timestep: int,
        current_sim_state: RepairsSimState,
        desired_sim_state: RepairsSimState,
    ):
        if current_timestep not in self.triggered_reward_per_future_timestep:
            return 0.0  # handle no expected reward
        triggered_reward_this_timestep = self.triggered_reward_per_future_timestep.pop(
            current_timestep
        )  # note: pop because we want to remove it.
        total_reward_this_timestep = 0.0
        for reward_type in triggered_reward_this_timestep:
            if reward_type == RewardType.FASTENER_INSERTION:
                fastener_id, body_id = self.reward_check_data[current_timestep]
                if current_sim_state.physical_state[
                    self.sim_id
                ].check_if_fastener_inserted(body_id, fastener_id):
                    total_reward_this_timestep += RewardType.FASTENER_INSERTION.value[1]

                    # do not give reward for the second time
                    self.already_triggered[RewardType.FASTENER_INSERTION.name].append(
                        (body_id, fastener_id)
                    )
            if reward_type == RewardType.PART_PLACEMENT:
                part_id = self.reward_check_data[current_timestep]
                if current_sim_state.physical_state[self.sim_id].check_if_part_placed(
                    part_id,
                    current_sim_state.physical_state[self.sim_id],
                    desired_sim_state.physical_state[self.sim_id],
                ):
                    total_reward_this_timestep += RewardType.PART_PLACEMENT.value[1]

                    # do not give reward for the second time
                    self.already_triggered[RewardType.PART_PLACEMENT.name].append(
                        part_id
                    )

        # return sum of reward for this timestep (for this environment.)
        return total_reward_this_timestep

def calculate_done(scene_data: ConcurrentSceneData):
    _diff, diff_count = scene_data.current_state.diff(scene_data.desired_state)
    return diff_count == 0
