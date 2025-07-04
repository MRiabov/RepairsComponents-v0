"""Debug rendering pipeline for Genesis scene.

This file contains all logic that touches the ``gs.Scene`` instance or the
``gs_entities`` dictionary from the moment a scene is created up to the final
camera rendering helper ``_render_all_cameras``.

External / undefined symbols that need to be provided by the caller:
-------------------------------------------------------------------
1. ``translate_state_to_genesis_scene(scene, desired_sim_state, mesh_file_names, random_textures)``
   Converts a high-level ``RepairsSimState`` description into concrete Genesis
   entities added to ``scene`` and returns the populated scene together with a
   ``dict[str, genesis.engine.entities.RigidEntity]`` mapping.
2. ``RepairsSimState`` –  dataclass describing the desired simulation state.
3. ``tooling_stand_plate.export_path(base_dir)`` – helper returning a path to
   the tooling-stand mesh.  Either stub it or provide the actual implementation
   from ``repairs_components.geometry.base_env.tooling_stand_plate``.
4. The MJCF file referenced in ``add_base_scene_geometry``
   (``"xml/franka_emika_panda/panda.xml"``) must exist on disk.

Adjust these symbols or replace the corresponding calls if you are running this
file in isolation.
"""

from __future__ import annotations
from repairs_components.processing.translation import translate_state_to_genesis_scene
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import RepairsSimState

import time
from pathlib import Path
from typing import Any, Dict, List

from repairs_components.save_and_load.multienv_dataloader import (
    RepairsEnvDataLoader,
)
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
)
from repairs_components.geometry.base_env import tooling_stand_plate

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.vis.visualizer import Camera

################################################################################
# Rendering helpers                                                             #
################################################################################


def obs_to_int8(rgb: np.ndarray, depth: np.ndarray, normal: np.ndarray) -> torch.Tensor:
    """Pack RGB, depth and surface normal buffers into a single uint8 tensor.

    The function mirrors the behaviour used inside *gym_env.py* to create the
    per-camera observation tensor that is later handed to ML code.  Depth is
    first normalised into the [0, 1] range, normals are mapped from [-1, 1] to
    [0, 1].  All three channels are then scaled to [0, 255] and concatenated
    along the last (channel) dimension.
    """
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)[..., None]

    normal_uint8 = ((normal * 0.5 + 0.5) * 255).astype(np.uint8)

    packed = np.concatenate([rgb_uint8, depth_uint8, normal_uint8], axis=-1)
    return torch.from_numpy(packed).cuda()


def _render_all_cameras(cameras: List[Camera]) -> torch.Tensor:
    """Render *all* cameras and return a stacked `(batch, cam, H, W, C)` tensor."""
    env_obs: List[torch.Tensor] = []
    for camera in cameras:
        # Genesis returns (H, W, 3/1/3) numpy arrays in float32, range [0, 1]
        rgb, depth, _seg, normal = camera.render(rgb=True, depth=True, normal=True)
        env_obs.append(obs_to_int8(rgb, depth, normal))  # (B, H, W, 7)

    # -> (B, num_cams, H, W, 7)
    return torch.stack(env_obs, dim=1)


################################################################################
# Scene-construction helpers                                                    #
################################################################################


def add_base_scene_geometry(scene: gs.Scene, base_dir: Path):
    """Insert the tooling-stand, the Franka robot and two cameras into *scene*."""
    # NOTE: tooling stand is moved so that its top surface sits at z == 0.
    tooling_stand: RigidEntity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(tooling_stand_plate.export_path(base_dir)),  # type: ignore
            scale=1.0,  # CAD files are already in cm, Genesis works in metres
            pos=(0, -(0.64 / 2 + 0.2), -0.2),
            euler=(90, 0, 0),
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 0.7, 0.3, 0.3)),
    )

    # Franka Panda manipulator – make sure mjcf path is correct for your setup
    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.3, -(0.64 / 2 + 0.2 / 2), 0.0),
        )
    )

    # Two static RGB-D cameras looking at the working area
    cam_1 = scene.add_camera(
        pos=(1.0, 2.5, 3.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
    )

    cam_2 = scene.add_camera(
        pos=(-2.5, 1.5, 1.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
        GUI=False,
    )

    _ground = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.2)))

    return scene, [cam_1, cam_2], franka


def initialize_and_build_scene(
    scene: gs.Scene,
    desired_sim_state: RepairsSimState,
    mesh_file_names: Dict[str, str],
    batch_dim: int,
    *,
    base_dir: Path,
    scene_id: int = 0,
    random_textures: bool = False,
):
    """Populate *scene* using *desired_sim_state* and build a batched sim.

    1.   Convert the high-level *desired_sim_state* to Genesis entities via
         ``translate_state_to_genesis_scene``.
    2.   Add the base environment geometry (tooling stand, robot, cameras).
    3.   Build the scene with *batch_dim* environments in parallel.
    4.   Configure PD gains and torque limits on the robot.
    """
    # ---------------------------------------------------------------------
    # 1. CAD → Genesis entities
    # ---------------------------------------------------------------------
    # scene, gs_entities = translate_state_to_genesis_scene(  # type: ignore
    #     scene, desired_sim_state, mesh_file_names, random_textures
    # )

    # num_random_boxes = 5
    # sizes = torch.rand(num_random_boxes, 3) * 0.2 + 0.05
    # positions = torch.rand(num_random_boxes, 3) * 0.4 + 0.3
    # box_entities = {}
    # # for i, (size, pos) in enumerate(zip(sizes, positions)):
    # #     box_entities["box_" + str(i)] = scene.add_entity(
    # #         gs.morphs.Box(pos=tuple(pos), size=tuple(size))
    # #     )

    sizes = torch.tensor([0.1, 0.1, 0.1])
    positions = torch.tensor([0.3, 0.3, 0.3])
    gs_entities = {}
    gs_entities["box@solid"] = scene.add_entity(
        gs.morphs.Box(pos=tuple(positions), size=tuple(sizes))
    )

    # ---------------------------------------------------------------------
    # 2. Additional static environment geometry & cameras
    # ---------------------------------------------------------------------
    scene, cameras, franka = add_base_scene_geometry(scene, base_dir)
    # gs_entities["franka@control"] = franka  # make robot accessible to caller

    # ---------------------------------------------------------------------
    # 3. Finalise scene & build
    # ---------------------------------------------------------------------
    t0 = time.time()
    print(f"[Scene {scene_id}] building …", flush=True)
    scene.build(n_envs=batch_dim)
    print(f"[Scene {scene_id}] built in {time.time() - t0:.2f} s", flush=True)

    # ---------------------------------------------------------------------
    # 4. Robot controller tuning – values copied from gym_env.py
    # ---------------------------------------------------------------------
    franka.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
    )
    franka.set_dofs_kv(kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    return scene, cameras, gs_entities, franka


################################################################################
# Utility                                                                      #
################################################################################


def move_entities_to_pos(
    gs_entities: Dict[str, RigidEntity],
    starting_sim_state: "RepairsSimState",
    env_idx: torch.Tensor | None = None,
):
    """Re-position entities according to *starting_sim_state*.

    This helper is used both during the initial reset and for per-episode
    environment resets.  Positions in *starting_sim_state* are expressed in
    centimetres, Genesis expects metres – hence the `/ 100` conversion.
    """
    if env_idx is None:
        env_idx = torch.arange(len(starting_sim_state.physical_state))

    all_pos_cm = torch.stack(
        [torch.tensor(s.graph.position) for s in starting_sim_state.physical_state],
        dim=0,
    )
    all_pos_m = all_pos_cm / 100.0  # cm → m

    for name, body_idx in starting_sim_state.physical_state[
        env_idx[0]
    ].body_indices.items():
        entity = gs_entities[name]
        entity.set_pos(all_pos_m[env_idx, body_idx], envs_idx=env_idx)


################################################################################
# Data-loader helper                                                           #
################################################################################


def load_scene_config(
    base_dir: Path, env_setup_id: int = 0, batch_dim: int = 16
) -> tuple[ConcurrentSceneData, Dict[str, str]]:
    """Grab ONE `ConcurrentSceneData` from the offline dataloader.

    This assumes that your dataset lives under *base_dir* (same structure that
    multienv_dataloader expects) and that `env_setup_id` exists there. If the
    dataset or the id is missing the function raises and the caller can fall
    back to an empty stub.
    """
    dataloader = RepairsEnvDataLoader(
        online=False,
        env_setup_ids=[env_setup_id],
        offline_data_dir=base_dir,
        prefetch_memory_size=4,
    )

    # Request exactly one config for this scene
    request = torch.tensor([batch_dim], dtype=torch.int16)
    batches = dataloader.get_processed_data(request)
    scene_cfg = batches[0]  # first (and only) env_setup
    if scene_cfg is None:
        raise RuntimeError("No configs returned by dataloader – check dataset path/id")

    # In offline mode the mesh mapping is stored in `mesh_file_names` attr
    mesh_files = getattr(scene_cfg, "mesh_file_names", {})  # type: ignore
    return scene_cfg, mesh_files


################################################################################
# Runnable example                                                             #
################################################################################

if __name__ == "__main__":
    """Minimal standalone demo.

    1.  Creates an *empty* Genesis scene (no parts) plus the base geometry &
        cameras.
    2.  Steps the physics for ~2 seconds while orbiting the first camera.
    3.  Saves a ``video.mp4`` timelapse in the current directory.

    Replace the *stub* ``RepairsSimState`` below with a real one to visualise a
    populated assembly.
    """
    base_dir = Path("/workspace/data")
    gs.init(debug=True)

    # ------------------------------------------------------------------
    # 1. Try to pull a populated config from offline dataset
    # ------------------------------------------------------------------
    try:
        scene_cfg, mesh_files_stub = load_scene_config(
            base_dir, env_setup_id=0, batch_dim=10
        )
        desired_state = scene_cfg.desired_state
        batch_dim = scene_cfg.batch_dim
        print("Loaded populated scene from dataset.")
    except Exception as err:
        raise RuntimeError("[debug_scene_render] Falling back to empty stub:", err)

    desired_state = scene_cfg.desired_state
    mesh_files_stub = getattr(scene_cfg, "mesh_file_names", {})  # type: ignore

    scene_1 = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, substeps=2),
        show_viewer=False,
        show_FPS=False,
        vis_options=gs.options.VisOptions(
            env_separate_rigid=True,
            shadow=True,
        ),
    )
    scene_2 = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, substeps=2),
        show_viewer=False,
        show_FPS=False,
        vis_options=gs.options.VisOptions(
            env_separate_rigid=True,
            shadow=True,
        ),
    )

    # Build scene with our helper (adds robot + cameras)
    scene_1, cameras_1, gs_entities_1, franka_1 = initialize_and_build_scene(
        scene_1,
        desired_state,
        mesh_files_stub,
        batch_dim,
        base_dir=base_dir,
        scene_id=0,
        random_textures=False,
    )
    scene_2, cameras_2, gs_entities_2, franka_2 = initialize_and_build_scene(
        scene_2,
        desired_state,
        mesh_files_stub,
        batch_dim,
        base_dir=base_dir,
        scene_id=1,
        random_textures=False,
    )

    # ------------------------------------------------------------------
    # 2. Record a short orbit movie using the first camera
    # ------------------------------------------------------------------

    cameras_scene_1 = scene_1.visualizer.cameras
    cameras_scene_2 = scene_2.visualizer.cameras
    cameras_scene_1[0].start_recording()
    cameras_scene_2[0].start_recording()

    for i in range(120):  # ~2 s at 60 fps
        scene_1.step(update_visualizer=False, refresh_visualizer=False)
        scene_2.step(update_visualizer=False, refresh_visualizer=False)

        if i % 10 == 0 and i > 0:
            move_entities_to_pos(gs_entities_1, desired_state)
            move_entities_to_pos(gs_entities_2, desired_state)

                        # update the scene
            # Reset robot joint positions to default
            dof_pos = self.default_dof_pos.expand(reset_scene.batch_dim, -1)

            self.concurrent_scenes_data[scene_id].gs_entities[
                "franka@control"
            ].set_dofs_position(position=dof_pos, envs_idx=reset_env_ids_this_scene)

        print("render!")
        vid_obs_1 = _render_all_cameras(cameras_scene_1)
        vid_obs_2 = _render_all_cameras(cameras_scene_2)

    cameras_scene_1[0].stop_recording(save_to_filename="video_1.mp4", fps=60)
    cameras_scene_2[0].stop_recording(save_to_filename="video_2.mp4", fps=60)

    # Quick sanity-print of the per-camera observation tensor
    print(
        "Video obs shape:",
        vid_obs_1.shape if hasattr(vid_obs_1, "shape") else len(vid_obs_1),
    )
