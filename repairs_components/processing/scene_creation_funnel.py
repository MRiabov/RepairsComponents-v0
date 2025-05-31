"""A "funnel" to create Genesis scenes from desired geometries and tasks.

Order:
1. create_random_scenes is a general, high_level function responsible for processing of the entire funnel.
2. starting_state_geom
"""

import genesis as gs
from repairs_components.processing.voxel_export import export_voxel_grid
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import Compound
from repairs_components.processing.translation import (
    translate_compound_to_sim_state,
    translate_to_genesis_scene,
)
from repairs_components.geometry.base_env.tooling_stand_plate import (
    plate_env_bd_geometry,
)
import torch


def create_random_scenes(
    empty_scene: gs.Scene,
    env_setup: EnvSetup,
    tasks: list[Task],
    num_scenes_per_task: int=128,
):
    """`create_random_scenes` is a general, high_level function responsible for processing of the entire 
    funnel, and is the only method users should call from `scene_creation_funnel`."""
    

    # create starting_state
    starting_scene_geom_ = starting_state_geom(
        env_setup, task
    )  # create task... in a for loop...
    desired_state_geom_ = desired_state_geom(env_setup, task)

    geom_with_base_env = add_base_env_to_geom(
        base_env, starting_scene_geom_, move_by=starting_scene.get_default_pos()
    )

    # voxelize both
    starting_voxel_grid = export_voxel_grid(starting_scene_geom_, voxel_size=64 / 256)
    desired_voxel_grid = export_voxel_grid(desired_state_geom_, voxel_size=64 / 256)

    # create RepairsSimState for both
    starting_sim_state = translate_compound_to_sim_state(geom_with_base_env)
    desired_sim_state = translate_compound_to_sim_state(desired_state_geom_)

    # for starting scene, move it to an appropriate position #no, not here...
    # create a FIRST genesis scene for starting state from desired state; it is to be discarded, however.
    first_desired_scene, hex_to_name = translate_to_genesis_scene(
        empty_scene, desired_state_geom_, desired_sim_state, connector_positions
    )

    # build a single scene... but batched
    first_desired_scene.build(n_envs=num_scenes_per_task*len(tasks))
    
    #move parts to their necessary positions
    

    # note: repairsSimState comparison won't work without moving the desired physical state by `move_by` from base env.


def starting_state_geom(env_setup: EnvSetup, task: Task) -> Compound:
    """
    Perturb the starting state based on the starting state geom.

    Args:
        sim: The simulation object to be set up.
    """
    return task.perturb_initial_state(env_setup.desired_state_geom())


def desired_state_geom(env_setup: EnvSetup, task: Task) -> Compound:
    """
    Perturb the desired state based on the starting state geom.

    Args:
        sim: The simulation object to be set up.
    """
    return task.perturb_desired_state(env_setup.desired_state_geom())


def add_base_env_to_geom(
    base_env: Compound, desired_geom: Compound, move_by: tuple[float, float, float]
) -> Compound:
    """
    Add the base environment to the desired geometry.
    """
    # move to the center of the environment (or wherever the environment expects)
    base_env = base_env.move(Pos(*move_by))
    return Compound(children=[base_env, desired_geom])


def add_geom_to_base_genesis_scene(scene: gs.Scene):
    return ...
