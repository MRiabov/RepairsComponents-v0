import genesis as gs
import json
import uuid

from genesis.engine.entities import RigidEntity

from repairs_components.geometry.base import Component
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.electronics import simulator
import numpy as np
from dataclasses import asdict
from pathlib import Path
from build123d import Compound
from genesis.vis.camera import Camera


from repairs_components.geometry.connectors.connectors import check_connections

from repairs_components.geometry.fasteners import (
    activate_connection,
    check_fastener_possible_insertion,
)
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.processing.translation import (
    translate_genesis_to_python,
    translate_to_genesis_scene,
)
import torch


def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    sim_state: RepairsSimState,
    desired_state: RepairsSimState,
):
    "Step repairs sim."
    # get values (state) from genesis
    (
        sim_state,
        picked_up_fastener_tip_position,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    ) = translate_genesis_to_python(scene, gs_entities, sim_state)

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = sim_state.diff(desired_state)

    # Save the state to a JSON file
    state_file = sim_state.save()
    success = total_diff_left == 0
    for env_idx in range(scene.n_envs):
        # step next action:
        if not success:
            possible_fastener_attachment = check_fastener_possible_insertion(
                picked_up_fastener_tip_position[env_idx], fastener_hole_positions
            )
            picked_up_fastener_name = sim_state.tool_state[
                env_idx
            ].current_tool.picked_up_fastener_name
            if (
                possible_fastener_attachment
                and isinstance(sim_state.tool_state[env_idx].current_tool, Screwdriver)
                and actions["screw_in"]  # note: change to use different tooling.
            ):  # this is done accross environments..
                assert picked_up_fastener_name is not None, (
                    "picked_up_fastener_name is None even though attachment calling is true."
                )
                fastener_entity = gs_entities[picked_up_fastener_name]
                activate_connection(fastener_entity, possible_fastener_attachment)
                sim_state.physical_state[env_idx].connect(
                    picked_up_fastener_name, possible_fastener_attachment, None
                )  # FIXME: the fastener is an edge now, so I'll need to check double the fastener connection possibility during insertion.

            # step electronics
            # find physical connections
            electronics_attachment = check_connections(  # future TODO: GPU.
                male_connector_positions, female_connector_positions
            )

            # clear connections before adding new ones
            sim_state.electronics_state[env_idx].clear_connections()
            for i in electronics_attachment:
                sim_state.electronics_state[env_idx].connect(i[0], i[1])
        # propagate the electricity simulation.
        # simulator.simulate_circuit(
        #     electronics_state.components
        # )  # and why would I need it at all here? I don't. I only need the robot to put everything into place yet.
        # however simming would be useful for future projects, of course.
        # well, and what if we want to find a faulty electrical component? sim is the only way to find it.

    return success, total_diff_left, sim_state, diff


# On reset, the only thing that is necessary in repairs_sim_step is to change the sim state to the new one.
# everything else is done in Genesis.
# def reset(  # get from Repairs-v0
#     scene: gs.Scene,
#     new_build123d_compound: Compound,
#     sim_state: RepairsSimState,
# ):
# get an assembly from build123d.

# # here should be perturb initial state, and so on.
# genesis_scene = translate_to_genesis_scene(scene, new_build123d_compound, sim_state)
# return genesis_scene
