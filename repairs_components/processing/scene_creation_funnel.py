"""A "funnel" to create Genesis scenes from desired geometries and tasks.

Order:
1. create_random_scenes is a general, high_level function responsible for processing of the entire funnel.
2. starting_state_geom
"""

from genesis.engine.entities import RigidEntity
from repairs_components.geometry.base_env import tooling_stand_plate
from repairs_components.processing.voxel_export import export_voxel_grid
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import Compound, Pos
from repairs_components.processing.translation import (
    translate_compound_to_sim_state,
    translate_to_genesis_scene,
)
from repairs_components.geometry.base_env.tooling_stand_plate import (
    genesis_setup,
    plate_env_bd_geometry,
)
import torch
import genesis as gs


def create_random_scenes(
    empty_scene: gs.Scene,
    env_setup: EnvSetup,
    tasks: list[Task],
    num_scenes_per_task: int = 128,
):
    """`create_random_scenes` is a general, high_level function responsible for processing of the entire 
    funnel, and is the only method users should call from `scene_creation_funnel`."""
    # note: ideally, `build` would, of course, happen async.
    voxel_grids_initial = torch.zeros(
        (len(tasks) * num_scenes_per_task, 256, 256, 256), dtype=torch.uint8
    )
    voxel_grids_desired = torch.zeros(
        (len(tasks) * num_scenes_per_task, 256, 256, 256), dtype=torch.uint8
    )
    starting_sim_states = []
    desired_sim_states = []

    # create starting_state
    for task_idx, task in enumerate(tasks):
        for env_i in range(num_scenes_per_task):
            index = task_idx * num_scenes_per_task + env_i
            starting_scene_geom_ = starting_state_geom(
                env_setup, task
            )  # create task... in a for loop...
            desired_state_geom_ = desired_state_geom(env_setup, task)

            # voxelize both
            starting_voxel_grid = export_voxel_grid(
                starting_scene_geom_, voxel_size=64 / 256
            )
            desired_voxel_grid = export_voxel_grid(
                desired_state_geom_, voxel_size=64 / 256
            )

            voxel_grids_initial[index] = torch.from_numpy(starting_voxel_grid)
            voxel_grids_desired[index] = torch.from_numpy(desired_voxel_grid)

            # create RepairsSimState for both
            starting_sim_state = translate_compound_to_sim_state(starting_scene_geom_)
            desired_sim_state = translate_compound_to_sim_state(desired_state_geom_)

            # store states
            starting_sim_states.append(starting_sim_state)
            desired_sim_states.append(desired_sim_state)

    # for starting scene, move it to an appropriate position #no, not here...
    # create a FIRST genesis scene for starting state from desired state; it is to be discarded, however.
    first_desired_scene, hex_to_name, gs_entities = translate_to_genesis_scene(
        empty_scene, desired_state_geom_, desired_sim_state
    )

    # initiate cameras and others in genesis scene:
    first_desired_scene, cameras = add_base_scene_geometry(first_desired_scene)

    # build a single scene... but batched
    first_desired_scene.build(n_envs=num_scenes_per_task * len(tasks))

    # move parts to their necessary positions # yes, this could be batched, but idgaf
    for gs_entity_name, gs_entity in gs_entities.items():
        positions = torch.zeros(len(gs_entities), 3)

        for env_i, starting_sim_state in enumerate(starting_sim_states):
            positions[env_i] = torch.tensor(
                starting_sim_state.physical_state.positions[gs_entity_name]
            )
        # No need to move because the position is already centered.
        # positions = positions + torch.tensor(tooling_stand_plate.SCENE_CENTER) / 100
        gs_entity.set_pos(
            positions, envs_idx=torch.arange(len(tasks) * num_scenes_per_task)
        )
    assert all(
        ((e.get_AABB()[:, 0] <= 1).all() and (e.get_AABB()[:, 1] >= -1).all())
        for e in first_desired_scene.entities
    ), "Entities are out of expected bounds, likely a misconfiguration."

    print(gs_entities["box@solid"].get_AABB())

    # note: RepairsSimState comparison won't work without moving the desired physical state by `move_by` from base env.
    return (
        first_desired_scene,
        cameras,
        gs_entities,
        starting_sim_states,
        desired_sim_states,
        voxel_grids_initial,
        voxel_grids_desired,
    )

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
    desired_state_geom_ = env_setup.desired_state_geom()
    desired_state_geom_ = task.perturb_desired_state(desired_state_geom_)

    return desired_state_geom_


def add_base_env_to_geom(
    base_env: Compound, desired_geom: Compound, move_by: tuple[float, float, float]
) -> Compound:
    """
    Add the base environment to the desired geometry.
    """
    # move to the center of the environment (or wherever the environment expects)
    base_env = base_env.move(Pos(*move_by))
    return Compound(children=[base_env, desired_geom])


def add_base_scene_geometry(scene: gs.Scene):
    # NOTE: the tooling stand is repositioned to 0,0,-0.1 to position all parts on the very center of scene.
    tooling_stand: RigidEntity = scene.add_entity(
        gs.morphs.Mesh(
            file="geom_exports/tooling_stands/tool_stand_plate.gltf",
            scale=1,  # Use 1.0 scale since we're working in cm
            pos=(0, -(0.64 / 2 + 0.2), -0.2),
            euler=(90, 0, 0),  # Rotate 90 degrees around X axis
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 0.7, 0.3, 1)),  # Add color material
    )
    # Set up camera with proper position and lookat
    camera_1 = scene.add_camera(
        # pos=(1, 2.5, 3.5),
        pos=(1, 2.5, 3.5),  # Position camera further away and above
        lookat=(
            0,
            0,
            0.2,
        ),  # Look at the center of the working pos
        # lookat=(
        #     0.64 / 2,
        #     0.64 / 2 + tooling_stand_plate.STAND_PLATE_DEPTH / 100,
        #     0.3,
        # ),  # Look at the center of the working pos
        res=(1024, 1024),
    )

    camera_2 = scene.add_camera(
        pos=(-2.5, 1.5, 1.5),  # second camera from the other side
        lookat=(
            0,
            0,
            0.2,
        ),  # Look at the center of the working pos
        res=(1024, 1024),
    )
    plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, -0.2)))
    return scene, [camera_1, camera_2]


def normalize_to_center(compound: Compound) -> Compound:
    bbox = compound.bounding_box()
    center = bbox.center()
    return compound.move(Pos(-center.x, -center.y, -center.z / 2))