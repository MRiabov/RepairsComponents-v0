"""A "funnel" to create Genesis scenes from desired geometries and tasks.

Order:
1. create_random_scenes is a general, high_level function responsible for processing of the entire funnel.
2. starting_state_geom
"""

import copy
import json
import pathlib
from pathlib import Path
from genesis.engine.entities import RigidEntity
from repairs_components.geometry.base_env import tooling_stand_plate
from repairs_components.processing.voxel_export import export_voxel_grid
from repairs_components.processing.tasks import Task
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import Compound, Pos, export_gltf, export_stl, Unit, CenterOf
from repairs_components.processing.translation import (
    translate_compound_to_sim_state,
    translate_state_to_genesis_scene,
)

import torch
import genesis as gs
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
from repairs_components.training_utils.sim_state_global import (
    RepairsSimState,
    merge_global_states,
)

from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)
import numpy as np


def create_env_configs(  # TODO voxelization and other cache carry mid-loops
    env_setups: list[
        EnvSetup
    ],  # note: there must be one gs scene per EnvSetup. So this could be done in for loop.
    tasks: list[Task],
    num_configs_to_generate_per_scene: torch.Tensor,  # int [len]
    save: bool = False,
    save_path: pathlib.Path | None = None,
    vox_res: int = 256,  # should be equal
) -> tuple[list[ConcurrentSceneData], dict[str, str]]:
    """`create_env_configs` is a general, high_level function responsible for creating of randomized configurations
    (problems) for the ML to solve, to later be translated to Genesis. It does not have to do anything to do with Genesis.

    `create_env_configs` should only be called from `multienv_dataloader`.

    Returns: ConcurrentSceneData for each environment and mesh_file_names if save is True.
    """
    assert len(tasks) > 0, "Tasks can not be empty."
    assert any(num_configs_to_generate_per_scene) > 0, (
        "At least one scene must be generated."
    )
    if save:
        assert save_path is not None, "Save path must be provided if save is True"
    # assert len(num_configs_to_generate_per_scene) == len(tasks), (
    #     "Number of tasks and number of configs to generate must match."
    # ) # not true. it must be split amongst tasks

    scene_config_batches: list[ConcurrentSceneData] = []
    # create starting_state
    for scene_idx, scene_gen_count in enumerate(num_configs_to_generate_per_scene):
        if scene_gen_count == 0:
            scene_config_batches.append(None)
            continue
        # Initialize lists to store sparse tensors and simulation states
        voxel_grids_initial = []
        voxel_grids_desired = []
        starting_sim_states = []
        desired_sim_states = []

        # training batches as for a dataloader/ML training batches.
        init_diffs = []
        init_diff_counts = []

        voxelization_cache = {}
        task_ids = torch.randint(low=0, high=len(tasks), size=(scene_gen_count,))
        for _ in range(scene_gen_count):
            task_id = task_ids[_]  # randomly select a task
            starting_scene_geom_ = starting_state_geom(
                env_setups[scene_idx], tasks[task_id], env_size=(64, 64, 64)
            )  # create task... in a for loop...
            # note: at the moment the starting scene goes out of bounds a little, but whatever, it'll only generalize better.
            desired_state_geom_ = desired_state_geom(
                env_setups[scene_idx], tasks[task_id], env_size=(64, 64, 64)
            )

            # voxelize both (sparsely!)
            starting_voxel_grid, voxelization_cache = export_voxel_grid(
                starting_scene_geom_,
                voxel_size=64 / 256,
                cached=True,
                cache=voxelization_cache,
                save=save,
                save_path=save_path,
                scene_file_name=f"vox_init_{scene_idx}.pt",
            )
            desired_voxel_grid, voxelization_cache = export_voxel_grid(
                desired_state_geom_,
                voxel_size=64 / 256,
                cached=True,
                cache=voxelization_cache,
                save=save,
                save_path=save_path,
                scene_file_name=f"vox_des_{scene_idx}.pt",
            )

            # Store sparse tensors directly
            voxel_grids_initial.append(starting_voxel_grid)
            voxel_grids_desired.append(desired_voxel_grid)

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

        # using the last geom, persist part meshes
        if save:
            mesh_file_names = persist_meshes_and_mjcf(
                desired_state_geom_,
                save_dir=save_path,
                scene_id=scene_idx,
                export_format="gltf",
            )

            torch.save(  # only save after all are done
                torch.stack(voxel_grids_initial, dim=0),
                save_path / "voxels" / f"vox_init_{scene_idx}.pt",
            )
            torch.save(
                torch.stack(voxel_grids_desired, dim=0),
                save_path / "voxels" / f"vox_des_{scene_idx}.pt",
            )

        voxel_grids_initial = torch.stack(voxel_grids_initial, dim=0)
        voxel_grids_desired = torch.stack(voxel_grids_desired, dim=0)
        starting_sim_state = merge_global_states(starting_sim_states)
        desired_sim_state = merge_global_states(desired_sim_states)

        initial_diffs = {k: [d[k][0] for d in init_diffs] for k in init_diffs[0].keys()}

        this_scene_configs = ConcurrentSceneData(
            scene=None,
            gs_entities=None,
            cameras=None,
            init_state=starting_sim_state,
            current_state=copy.deepcopy(starting_sim_state),
            desired_state=desired_sim_state,
            vox_init=voxel_grids_initial,
            vox_des=voxel_grids_desired,
            initial_diffs=initial_diffs,
            initial_diff_counts=torch.tensor(init_diff_counts),
            scene_id=scene_idx,
            batch_dim=scene_gen_count.item(),
            reward_history=RewardHistory(batch_dim=scene_gen_count),
            step_count=torch.zeros(scene_gen_count, dtype=torch.int),
            task_ids=task_ids,
        )  # type: ignore # inttensor and tensor.
        scene_config_batches.append(this_scene_configs)

        if save:
            # get them later by get_graph_save_paths
            starting_sim_state.save(save_path, scene_idx, init=True)
            desired_sim_state.save(save_path, scene_idx, init=False)
            torch.save(
                torch.tensor(init_diff_counts),
                save_path / ("scene_" + str(scene_idx)) / "initial_diff_counts.pt",
            )
            torch.save(
                initial_diffs,  # may want
                save_path / ("scene_" + str(scene_idx)) / "initial_diffs.pt",
            )

    # note: RepairsSimState comparison won't work without moving the desired physical state by `move_by` from base env.
    return scene_config_batches, (mesh_file_names if save else {})


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
    desired_sim_state: RepairsSimState,
    mesh_file_names: dict[str, str],
    batch_dim: int,
    random_textures: bool = False,
):
    # for starting scene, move it to an appropriate position #no, not here...
    # create a FIRST genesis scene for starting state from desired state; it is to be discarded, however.
    first_desired_scene, initial_gs_entities = translate_state_to_genesis_scene(
        scene, desired_sim_state, mesh_file_names, random_textures
    )

    # initiate cameras and others in genesis scene:
    first_desired_scene, cameras, franka = add_base_scene_geometry(first_desired_scene)
    initial_gs_entities["franka@control"] = franka

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
            fixed=True,
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
    _plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, -0.2)))
    return scene, [camera_1, camera_2], franka


def move_entities_to_pos(
    gs_entities: dict[str, RigidEntity],
    starting_sim_state: RepairsSimState,
    env_idx: torch.Tensor | None = None,
):
    """Move parts to their necessary positions. Can be used both in reset and init."""
    if env_idx is None:
        env_idx = torch.arange(len(starting_sim_state.physical_state))
    # batch collect all positions (cm to meters) across environments
    all_positions = (
        torch.stack(
            [
                torch.tensor(s.graph.position, device=env_idx.device)
                for s in starting_sim_state.physical_state
            ],
            dim=0,
        )
        / 100
    )

    # set positions for each entity in batch
    for gs_entity_name, entity_idx in starting_sim_state.physical_state[
        env_idx[0]
    ].body_indices.items():
        entity = gs_entities[gs_entity_name]
        entity_pos = all_positions[env_idx, entity_idx]
        # No need to move because already centered
        entity.set_pos(entity_pos, envs_idx=env_idx)


# TODO why not used?
def normalize_to_center(compound: Compound) -> Compound:
    bbox = compound.bounding_box()
    center = bbox.center()
    return compound.move(Pos(-center.x, -center.y, -center.z / 2))


# mesh save utils
def generate_scene_meshes():
    "A function to generate all  meshes for all the scenes."
    if not tooling_stand_plate.export_path().exists():
        print("Tooling stand mesh not found. Generating...")
        tooling_stand_plate.plate_env_bd_geometry(export_geom_gltf=True)


def persist_meshes_and_mjcf(
    b123d_compound: Compound, save_dir: Path, scene_id: int, export_format="gltf"
):
    assert export_format in ("gltf", "stl")

    mesh_file_names = {}

    # flatten the array
    children = b123d_compound.descendants

    # export mesh
    for child in children:
        assert child.label is not None, "Child must have a label"
        assert "@" in child.label, "Part label must have a type delimiter."
        _part_name, part_type = child.label.split("@")
        center = child.center(CenterOf.BOUNDING_BOX)
        if export_format == "gltf":
            mesh_file_name = f"{child.label}.gltf"
            export_gltf(
                child.moved(Pos(-center)),
                save_dir / f"scene_{scene_id}" / mesh_file_name,
                unit=Unit.CM,
            )
        elif export_format == "stl":
            # fixme: stl does not resize? hmm. (resize units)
            mesh_file_name = f"{child.label}.stl"
            export_stl(child, file_path=save_dir / f"scene_{scene_id}" / mesh_file_name)

        # FIXME: deprecate mjcf!
        if part_type in ("connector", "button", "led", "switch"):
            connector: Connector = sim_state.electronics_state.components[label]  # type: ignore
            mjcf = connector.get_mjcf()
            with open(save_dir / f"scene_{scene_id}" / f"{child.label}.xml", "w") as f:
                f.write(mjcf)

        mesh_file_names[child.label] = mesh_file_name
    return mesh_file_names
