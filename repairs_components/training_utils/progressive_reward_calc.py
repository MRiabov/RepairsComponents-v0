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
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)
from repairs_components.training_utils.sim_state_global import RepairsSimState
import numpy as np
import torch


class RewardType(enum.Enum):
    "An enum of reward types to their delays and rewards to check the rewards: give a reward in value timesteps"

    FASTENER_INSERTION = (200, 2.0)
    PART_PLACEMENT = (200, 3.0)
    ELECTRONICS_CONNECTION = (200, 2.0)
    EXPECTED_FLUID_SENSOR_HIT = (200, 2.0)


@dataclass
class RewardHistory:  # note: this is stored per every environment yet, not batched.
    triggered_reward_per_future_timestep: list[dict[int, list[RewardType]]] = field(
        default_factory=list
    )
    reward_check_data: list[dict[int, tuple]] = field(default_factory=list)
    already_triggered: list[dict[str, list]] = field(
        default_factory=lambda: [
            {
                RewardType.FASTENER_INSERTION.name: [],
                RewardType.PART_PLACEMENT.name: [],
            }
            for _ in range(1)  # pointless, but whatever.
        ]
    )
    "For every reward type, store which reward were already triggered."

    def __init__(self, batch_dim: int):
        self.triggered_reward_per_future_timestep = [{} for _ in range(batch_dim)]
        self.reward_check_data = [{} for _ in range(batch_dim)]
        self.already_triggered = [
            {
                RewardType.FASTENER_INSERTION.name: [],
                RewardType.PART_PLACEMENT.name: [],
            }
            for _ in range(batch_dim)
        ]

    def register_correct_fastener_insertion(
        self, current_timestep: int, fastener_id: int, to_body_id: int, env_id: int
    ):
        self.triggered_reward_per_future_timestep[env_id][
            RewardType.FASTENER_INSERTION.value[0] + current_timestep
        ] = [RewardType.FASTENER_INSERTION]
        self.reward_check_data[env_id][current_timestep] = (fastener_id, to_body_id)

    def register_correct_part_placement(
        self, current_timestep: int, part_id: int, env_id: int
    ):
        self.triggered_reward_per_future_timestep[env_id][
            RewardType.PART_PLACEMENT.value[0] + current_timestep
        ] = [RewardType.PART_PLACEMENT]
        self.reward_check_data[env_id][current_timestep] = part_id

    def calculate_reward_this_timestep(self, scene_data: ConcurrentSceneData):
        reward_tensor = torch.zeros(scene_data.batch_dim)
        for env_id in range(scene_data.batch_dim):
            current_timestep = scene_data.step_count[env_id]
            if (
                current_timestep
                not in self.triggered_reward_per_future_timestep[env_id]
            ):
                continue
            # if there is expected reward for this timestep, get it.
            triggered_reward_this_timestep = self.triggered_reward_per_future_timestep[
                env_id
            ].pop(current_timestep)  # note: pop because we want to remove it.

            # process the expected reward - is it still here?
            for reward_type in triggered_reward_this_timestep:
                if reward_type == RewardType.FASTENER_INSERTION:
                    fastener_id, body_id = self.reward_check_data[current_timestep]
                    if scene_data.current_state.physical_state[
                        env_id
                    ].check_if_fastener_inserted(body_id, fastener_id):
                        reward_tensor[env_id] += RewardType.FASTENER_INSERTION.value[1]

                        # do not give reward for the second time
                        self.already_triggered[
                            RewardType.FASTENER_INSERTION.name
                        ].append((body_id, fastener_id))
                if reward_type == RewardType.PART_PLACEMENT:
                    part_id = self.reward_check_data[current_timestep]
                    if scene_data.current_state.physical_state[
                        env_id
                    ].check_if_part_placed(
                        part_id,
                        scene_data.current_state.physical_state[env_id],
                        scene_data.desired_state.physical_state[env_id],
                    ):
                        reward_tensor[env_id] += RewardType.PART_PLACEMENT.value[1]

                        # do not give reward for the second time
                        self.already_triggered[RewardType.PART_PLACEMENT.name].append(
                            part_id
                        )

                if reward_type == RewardType.ELECTRONICS_CONNECTION:
                    connector_id_1, connector_id_2 = self.reward_check_data[
                        current_timestep
                    ]
                    if scene_data.current_state.electronics_state[
                        env_id
                    ].check_if_electronics_connected(connector_id_1, connector_id_2):
                        reward_tensor[env_id] += (
                            RewardType.ELECTRONICS_CONNECTION.value[1]
                        )

        # return sum of reward for this timestep (for this environment.)
        return reward_tensor


def calculate_done(scene_data: ConcurrentSceneData):
    _diff, diff_count = scene_data.current_state.diff(scene_data.desired_state)
    return diff_count == 0
