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

from repairs_components.training_utils.sim_state_global import RepairsSimState
import torch
import genesis as gs
from repairs_components.training_utils.sim_state_global import merge_global_states
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)
import numpy as np


def create_env_configs(
    env_setups: list[
        EnvSetup
    ],  # note: there must be one gs scene per EnvSetup. So this could be done in for loop.
    tasks: list[Task],
    num_configs_to_generate_per_scene: torch.Tensor,  # int [len]
) -> list[ConcurrentSceneData]:
    """`create_env_configs` is a general, high_level function responsible for creating of randomized configurations
    (problems) for the ML to solve, to later be translated to Genesis. It does not have to do anything to do with Genesis.

    `create_env_configs` should only be called from `multienv_dataloader`."""
    assert len(tasks) > 0, "Tasks can not be empty."
    assert any(num_configs_to_generate_per_scene) > 0, (
        "At least one scene must be generated."
    )
    assert len(num_configs_to_generate_per_scene) == len(tasks), (
        "Number of tasks and number of configs to generate must match."
    )

    # FIXME: below generates only for starting states. This is a subcase of generating for however much we need.
    # instead it should generate as per num_configs_to_generate.

    scene_config_batches: list[ConcurrentSceneData] = []
    # create starting_state
    for scene_idx, scene_gen_count in enumerate(num_configs_to_generate_per_scene):
        if scene_gen_count == 0:
            scene_config_batches.append(None)
            continue
        voxel_grids_initial = torch.zeros(
            (scene_gen_count, 256, 256, 256), dtype=torch.uint8
        )
        voxel_grids_desired = torch.zeros(
            (scene_gen_count, 256, 256, 256), dtype=torch.uint8
        )

        starting_sim_states = []
        desired_sim_states = []

        # training batches as for a dataloader/ML training batches.
        training_batches = []
        init_diffs = []
        init_diff_counts = []
        for _ in range(scene_gen_count):
            starting_scene_geom_ = starting_state_geom(
                env_setups[scene_idx], tasks[scene_idx], env_size=(64, 64, 64)
            )  # create task... in a for loop...
            # note: at the moment the starting scene goes out of bounds a little, but whatever, it'll only generalize better.
            desired_state_geom_ = desired_state_geom(
                env_setups[scene_idx], tasks[scene_idx], env_size=(64, 64, 64)
            )

            # voxelize both
            starting_voxel_grid = export_voxel_grid(
                starting_scene_geom_, voxel_size=64 / 256
            )
            desired_voxel_grid = export_voxel_grid(
                desired_state_geom_, voxel_size=64 / 256
            )

            voxel_grids_initial[scene_idx] = torch.from_numpy(starting_voxel_grid)
            voxel_grids_desired[scene_idx] = torch.from_numpy(desired_voxel_grid)

            # create RepairsSimState for both
            starting_sim_state = translate_compound_to_sim_state([starting_scene_geom_])
            desired_sim_state = translate_compound_to_sim_state([desired_state_geom_])

            # store states
            starting_sim_states.append(starting_sim_state)
            desired_sim_states.append(desired_sim_state)

            # Store the initial difference count for reward calculation
            diff, initial_diff_count = starting_sim_state.diff(desired_sim_state)
            init_diffs.append(diff)
            init_diff_counts.append(initial_diff_count)

        starting_sim_state = merge_global_states(starting_sim_states)
        desired_sim_state = merge_global_states(desired_sim_states)
        this_scene_configs = ConcurrentSceneData(
            scene=None,
            gs_entities=None,
            cameras=None,
            current_state=starting_sim_state,
            desired_state=desired_sim_state,
            vox_init=voxel_grids_initial,
            vox_des=voxel_grids_desired,
            initial_diffs={
                k: torch.cat([diff[k] for diff in init_diffs], dim=0)
                for k in init_diffs[0].keys()
            },  # I think this will fail... diffs are not just flat tensors, they are sparse.
            initial_diff_counts=torch.tensor(init_diff_counts),
            scene_id=scene_idx,
        )
        scene_config_batches.append(this_scene_configs)

    # note: RepairsSimState comparison won't work without moving the desired physical state by `move_by` from base env.
    return scene_config_batches


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
    scene: gs.Scene,
    desired_state_geom: Compound,
    desired_sim_state: RepairsSimState,
    batch_dim: int,
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

    # ===== Control Parameters =====
    # Set PD control gains (tuned for Franka Emika Panda)
    # These values are robot-specific and affect the stiffness and damping
    # of the robot's joints during control
    franka.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )

    # Set force limits for each joint (in Nm for rotational joints, N for prismatic)
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    return first_desired_scene, cameras, initial_gs_entities, franka


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
    franka: RigidEntity = scene.add_entity(
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
