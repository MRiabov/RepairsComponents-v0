from dataclasses import dataclass
from genesis.vis.camera import Camera
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity
from repairs_components.training_utils.sim_state_global import RepairsSimState


@dataclass
class ConcurrentSceneData:
    """Dataclass for concurrent scene data.
    Each entry is batched with a single batch dim, calculated
    as batch_dim // concurrent_scenes (except scene, gs_entities and cameras,
    since they are singletons.)"""

    scene: gs.Scene
    gs_entities: list[RigidEntity]
    cameras: tuple[Camera, Camera]
    current_state: RepairsSimState
    desired_state: RepairsSimState
    vox_init: torch.Tensor
    vox_des: torch.Tensor
    initial_diff: dict[str, torch.Tensor]
    initial_diff_count: torch.Tensor  # shape: (batch_dim // concurrent_scenes,)
    scene_id: int
    "A safety int to ensure we don't access the wrong scene."
