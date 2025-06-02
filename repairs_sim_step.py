import genesis as gs
import json
import uuid

from repairs_components.geometry.base import Component
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
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.fluid_state import FluidState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.processing.translation import (
    translate_genesis_to_python,
    translate_to_genesis_scene,
)
import torch

def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    hex_to_name: dict[str, str],
    sim_state: RepairsSimState,
    desired_state: RepairsSimState,
):
    "Returns diff for potential backu."
    # get values from genesis
    (
        sim_state,
        picked_up_fastener_tip_position,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    ) = translate_genesis_to_python(scene, hex_to_name, sim_state)

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = sim_state.diff(desired_state)

    # Save the state to a JSON file
    state_file = sim_state.save()
    success = total_diff_left == 0

    # step next action:
    if not success:
        possible_fastener_attachment = check_fastener_possible_insertion(
            picked_up_fastener_tip_position, fastener_hole_positions
        )
        picked_up_fastener_name = sim_state.tool_state.tool.picked_up_fastener_name
        if (
            possible_fastener_attachment
            and sim_state.tool_state.tool.name == "screwdriver"  # do something here.
            and actions["screw_in"]  # note: change to use different tooling.
        ):  # this is done accross environments..
            assert picked_up_fastener_name is not None, (
                "picked_up_fastener_name is None even though attachment calling is true."
            )
            fastener_entity = next(
                filter(lambda name: name == picked_up_fastener_name, scene.entities)
            )
            activate_connection(fastener_entity, possible_fastener_attachment)
            sim_state.physical_state.connect(
                picked_up_fastener_name, possible_fastener_attachment
            )
        # step electronics
        # find physical connections
        electronics_attachment = check_connections(
            male_connector_positions, female_connector_positions
        )

        # clear connections before adding new ones
        sim_state.electronics_state.clear_connections()
        for i in electronics_attachment:
            sim_state.electronics_state.connect(i[0], i[1])
        # propagate the electricity simulation.
        # simulator.simulate_circuit(
        #     electronics_state.components
        # )  # and why would I need it at all here? I don't. I only need the robot to put everything into place yet.
        # however simming would be useful for future projects, of course.
        # well, and what if we want to find a faulty electrical component? sim is the only way to find it.

    return success, total_diff_left, sim_state, diff


def reset(  # get from Repairs-v0
    scene: gs.Scene,
    new_build123d_scene: Compound,
    sim_state: RepairsSimState,
    aux: dict[str, np.ndarray],
    cameras: list[Camera],
):  # reset 1 environment
    # get an assembly from build123d.
    genesis_scene = translate_to_genesis_scene(
        scene, new_build123d_scene, sim_state, aux
    )
    return scene
