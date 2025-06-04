"""A "funnel" to create Genesis scenes from desired geometries and tasks.

Order:
1. create_random_scenes is a general, high_level function responsible for processing of the entire funnel.
2. starting_state_geom
"""

import pathlib
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
from repairs_components.training_utils.sim_state_global import RepairsSimState
import torch
import genesis as gs
from repairs_components.training_utils.sim_state_global import merge_global_states


def create_random_scenes(
    empty_scene: gs.Scene,
    env_setup: EnvSetup,  # note: there must be one gs scene per EnvSetup. So this could be done in for loop.
    tasks: list[Task],
    create_num_scenes: int,
    num_scenes_per_task: int = 128,  # this is how many states to create.
):
    """`create_random_scenes` is a general, high_level function responsible for processing of the entire
    funnel, and is the only method users should call from `scene_creation_funnel`."""
    # note: ideally, `build` would, of course, happen async.
    assert len(tasks) > 0, "Tasks can not be empty."
    assert num_scenes_per_task > 0, "num_scenes_per_task can not be 0."
    assert (len(tasks) * num_scenes_per_task) % batch_dim == 0, (
        "In order to support easy batching, num of scenes per task * task must be divisible by batch_dim."
    )
    assert batch_dim <= (len(tasks) * num_scenes_per_task), (
        "batch_dim must be less than or equal to the number of training batches."
    )
    training_batches_created = (len(tasks) * num_scenes_per_task) // batch_dim

    voxel_grids_initial = torch.zeros((batch_dim, 256, 256, 256), dtype=torch.uint8)
    voxel_grids_desired = torch.zeros((batch_dim, 256, 256, 256), dtype=torch.uint8)
    starting_sim_states = []
    desired_sim_states = []

    # training batches as for a dataloader/ML training batches.
    training_batches = []

    # create starting_state
    for task_idx, task in enumerate(tasks):
        for env_i in range(num_scenes_per_task):
            index = task_idx * num_scenes_per_task + env_i
            starting_scene_geom_ = starting_state_geom(
                env_setup, task, env_size=(64, 64, 64)
            )  # create task... in a for loop...
            # note: at the moment the starting scene goes out of bounds a little, but whatever, it'll only generalize better.
            desired_state_geom_ = desired_state_geom(
                env_setup, task, env_size=(64, 64, 64)
            )

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
            starting_sim_state = translate_compound_to_sim_state([starting_scene_geom_])
            desired_sim_state = translate_compound_to_sim_state([desired_state_geom_])

            # store states
            starting_sim_states.append(starting_sim_state)
            desired_sim_states.append(desired_sim_state)

            # Store the initial difference count for reward calculation
            diff, initial_diff_count = starting_sim_state.diff(desired_sim_state)

    # pack them into an easily loadable list.
    for i in range(training_batches_created):
        start_idx = i * batch_dim
        end_idx = (i + 1) * batch_dim
        batch_starting_sim_state = merge_global_states(
            starting_sim_states[start_idx:end_idx]
        )
        batch_desired_sim_state = merge_global_states(
            desired_sim_states[start_idx:end_idx]
        )
        training_batches.append((batch_starting_sim_state, batch_desired_sim_state))

    # move all entities to initial positions.
    move_entities_to_pos(
        initial_gs_entities, starting_sim_states[0]
    )  # well, this shouldn't be here, but.

    assert all(
        ((e.get_AABB()[:, 0] <= 1).all() and (e.get_AABB()[:, 1] >= -1).all())
        for e in first_desired_scene.entities
    ), "Entities are out of expected bounds, likely a misconfiguration."
    assert all(
        (e.get_AABB()[:, :, 2] >= 0).all() for e in initial_gs_entities.values()
    ), "Entities are below the base plate."

    print(
        "gs_entities['box@solid'].get_AABB():",
        initial_gs_entities["box@solid"].get_AABB(),
    )
    # Add franka to gs_entities
    initial_gs_entities["franka@control"] = franka
    # to have gs_entities positions updated in visuals too, update the visual states.
    first_desired_scene.visualizer.update_visual_states()

    # note: RepairsSimState comparison won't work without moving the desired physical state by `move_by` from base env.
    return (
        first_desired_scene,
        cameras,
        initial_gs_entities,
        starting_sim_states,
        desired_sim_states,
        voxel_grids_initial,
        voxel_grids_desired,
    )


def starting_state_geom(
    env_setup: EnvSetup, task: Task, env_size=(64, 64, 64)
) -> Compound:
    """
    Perturb the starting state based on the starting state geom.

    Args:
        sim: The simulation object to be set up.
    """
    return task.perturb_initial_state(env_setup.desired_state_geom(), env_size=env_size)


def desired_state_geom(
    env_setup: EnvSetup, task: Task, env_size=(64, 64, 64)
) -> Compound:
    """
    Perturb the desired state based on the starting state geom.

    Args:
        sim: The simulation object to be set up.
    """
    return task.perturb_desired_state(env_setup.desired_state_geom(), env_size=env_size)


def initialize_and_build_scene(
    scene: gs.Scene, desired_state_geom: Compound, desired_sim_state: RepairsSimState, batch_dim:int
):
    # for starting scene, move it to an appropriate position #no, not here...
    # create a FIRST genesis scene for starting state from desired state; it is to be discarded, however.
    first_desired_scene, initial_gs_entities = translate_to_genesis_scene(
        scene, desired_state_geom, desired_sim_state
    )

    # initiate cameras and others in genesis scene:
    first_desired_scene, cameras, franka = add_base_scene_geometry(first_desired_scene)

    # build a single scene... but batched
    first_desired_scene.build(n_envs=batch_dim)
    return first_desired_scene, cameras, franka


def add_base_scene_geometry(scene: gs.Scene):
    # NOTE: the tooling stand is repositioned to 0,0,-0.1 to position all parts on the very center of scene.
    tooling_stand: RigidEntity = scene.add_entity(
        gs.morphs.Mesh(  # note: filepath necessary because debug switches it to other repo when running from Repairs-v0.
            file="/workspace/RepairsComponents-v0/geom_exports/tooling_stands/tool_stand_plate.gltf",
            scale=1,  # Use 1.0 scale since we're working in cm
            pos=(0, -(0.64 / 2 + 0.2), -0.2),
            euler=(90, 0, 0),  # Rotate 90 degrees around X axis
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 0.7, 0.3, 0.3)),  # Add color material
        # 0.3 alpha for debug.
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.3, -(0.64 / 2 + 0.2 / 2), 0),
        ),
    )  # franka arm standing on the correct place in the assembly.

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
        res=(256, 256),  # (1024, 1024),
    )

    camera_2 = scene.add_camera(
        pos=(-2.5, 1.5, 1.5),  # second camera from the other side
        lookat=(
            0,
            0,
            0.2,
        ),  # Look at the center of the working pos
        res=(256, 256),  # (1024, 1024),
    )
    plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, -0.2)))
    return scene, [camera_1, camera_2], franka


def move_entities_to_pos(
    gs_entities: dict[str, RigidEntity],
    starting_sim_state: RepairsSimState,
    env_idx: torch.Tensor | None = None,
):
    """Move parts to their necessary positions. Can be used both in reset and init."""
    if env_idx is None:
        env_idx = torch.arange(len(starting_sim_state.physical_state))
    for gs_entity_name, gs_entity in gs_entities.items():
        positions = torch.zeros((len(gs_entities), 3))

        for env_i in env_idx:
            positions[int(env_i.item())] = (
                torch.tensor(
                    starting_sim_state.physical_state[env_i].positions[gs_entity_name]
                )
                / 100  # cm to meters
            )
        # No need to move because the position is already centered.
        # positions = positions + torch.tensor(tooling_stand_plate.SCENE_CENTER) / 100
        gs_entity.set_pos(positions, envs_idx=env_idx)


# TODO why not used?
def normalize_to_center(compound: Compound) -> Compound:
    bbox = compound.bounding_box()
    center = bbox.center()
    return compound.move(Pos(-center.x, -center.y, -center.z / 2))


def generate_scene_meshes():
    "A function to generate all  meshes for all the scenes."
    if not pathlib.Path("/geom_exports/tooling_stands/tool_stand_plate.gltf").exists():
        tooling_stand_plate.plate_env_bd_geometry()
