import copy
from dataclasses import dataclass
from genesis.vis.camera import Camera
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity
from repairs_components.training_utils.sim_state_global import (
    RepairsSimState,
    merge_global_states,
    merge_global_states_at_idx,
)


@dataclass
class ConcurrentSceneData:
    """Dataclass for concurrent scene data.
    Each entry is batched with a single batch dim, calculated
    as batch_dim // concurrent_scenes (except scene, gs_entities and cameras,
    since they are singletons.)"""

    scene: gs.Scene
    gs_entities: dict[str, RigidEntity]
    cameras: tuple[Camera, Camera]
    current_state: RepairsSimState
    desired_state: RepairsSimState
    vox_init: torch.Tensor  # sparse tensor!
    vox_des: torch.Tensor  # sparse tensor!
    initial_diffs: dict[str, torch.Tensor]  # feature diffs and node diffs.
    initial_diff_counts: torch.Tensor  # shape: (batch_dim // concurrent_scenes,)
    scene_id: int
    "A safety int to ensure we don't access the wrong scene."
    reward_history: RewardHistory
    batch_dim: int
    """Batch dimension of this scene, must match length of all lists/tensors in this dataclass.
    Primarily for sanity checks."""
    step_count: torch.IntTensor = torch.zeros((), dtype=torch.int)
    "Step count in every scene. I don't think this should be diffed."

    # debug
    def __post_init__(self):
        assert isinstance(self.vox_des, torch.Tensor), "vox_des must be a torch tensor"
        assert isinstance(self.vox_init, torch.Tensor), (
            "vox_init must be a torch tensor"
        )
        assert self.vox_des.shape == self.vox_init.shape, (
            "vox_des and vox_init must have the same shape"
        )
        assert self.vox_init.ndim == 4, (
            f"vox_init must be a 4D tensor, but got {self.vox_init.shape}"
        )
        assert self.vox_init.shape[0] == self.batch_dim, (
            f"vox_init must have the same batch dimension as batch_dim, but got {self.vox_init.shape[0]} and {self.batch_dim}"
        )
        # reward history
        assert isinstance(self.reward_history, RewardHistory), (
            "reward_history must be a RewardHistory object"
        )
        assert self.reward_history.batch_dim == self.batch_dim, (
            f"reward_history must have the same batch dimension as batch_dim, but got {self.reward_history.batch_dim} and {self.batch_dim}"
        )
        if self.step_count.ndim == 0:
            self.step_count = torch.zeros(self.batch_dim, dtype=torch.int)
        else:
            assert self.step_count.shape[0] == self.batch_dim, (
                f"step_count must have the same batch dimension as batch_dim, but got {self.step_count.shape[0]} and {self.batch_dim}"
            )


def merge_concurrent_scene_configs(scene_configs: list[ConcurrentSceneData]):
    # assert the scenes in configs are equivalent.
    assert all(scene_configs[0].scene == scene_cfg.scene for scene_cfg in scene_configs)
    assert all(
        scene_configs[0].scene_id == scene_cfg.scene_id for scene_cfg in scene_configs
    )
    assert all(  # handle partial configs.
        scene_configs[0].gs_entities is None
        or set(scene_configs[0].gs_entities.keys()) == set(scene_cfg.gs_entities.keys())
        for scene_cfg in scene_configs
    )
    assert all(  # handle partial configs.
        scene_configs[0].cameras is None
        or len(scene_configs[0].cameras) == len(scene_cfg.cameras)
        for scene_cfg in scene_configs
    )

    # create a single big RewardHistorys
    reward_history = RewardHistory(
        batch_dim=sum([data.batch_dim for data in scene_configs])
    )
    for data in scene_configs:  # and just merge it in.
        reward_history.merge_at_idx(data.reward_history, torch.arange(data.batch_dim))

    # Cat voxel tensors
    vox_init = torch.cat([data.vox_init for data in scene_configs], dim=0)
    vox_des = torch.cat([data.vox_des for data in scene_configs], dim=0)

    # TODO: gs entities should not be there... or at least I don't know how to merge them.
    # same for scene and desired state. they should be None.

    # Extend tensors and RepairsSimState with items from other scene_configs

    new_scene_config = ConcurrentSceneData(
        vox_init=vox_init,
        vox_des=vox_des,
        initial_diffs={
            k: [data.initial_diffs[k] for data in scene_configs]
            for k in scene_configs[0].initial_diffs.keys()
        },
        initial_diff_counts=torch.cat(
            [data.initial_diff_counts for data in scene_configs], dim=0
        ),
        current_state=merge_global_states(
            [scene_cfg.current_state for scene_cfg in scene_configs]
        ),
        desired_state=merge_global_states(
            [scene_cfg.desired_state for scene_cfg in scene_configs]
        ),
        scene_id=scene_configs[0].scene_id,
        scene=scene_configs[0].scene,
        gs_entities=scene_configs[0].gs_entities,
        cameras=scene_configs[0].cameras,
        batch_dim=sum([data.batch_dim for data in scene_configs]),  # right?
        reward_history=reward_history,
    )
    return new_scene_config


def merge_scene_configs_at_idx(
    old_scene_config: "ConcurrentSceneData",
    new_scene_config: "ConcurrentSceneData",
    reset_configs: torch.BoolTensor,
) -> "ConcurrentSceneData":
    """Insert new scene configs at indices indicated by bool tensor `reset_configs`.

    Args:
        old_scene_config: The original scene configuration
        new_scene_config: The new scene configuration to merge from
        reset_configs: Boolean tensor indicating which indices to update from new_scene_config

    Returns:
        A new ConcurrentSceneData with merged values
    """
    batch_dim = len(old_scene_config.vox_init)
    assert len(reset_configs) == batch_dim, (
        "Reset states must have the same length as batch dimension."
    )
    assert reset_configs.dtype == torch.bool, "Reset states must be a bool tensor."
    assert reset_configs.ndim == 1, "Reset states must be a 1D tensor."

    # Create a new scene config with the same structure as the old one
    merged_scene_config = ConcurrentSceneData(
        vox_init=old_scene_config.vox_init.clone(),
        vox_des=old_scene_config.vox_des.clone(),
        initial_diffs=copy.deepcopy(old_scene_config.initial_diffs),
        initial_diff_counts=old_scene_config.initial_diff_counts.clone(),
        current_state=merge_global_states_at_idx(
            old_scene_config.current_state,
            new_scene_config.current_state,
            reset_configs,
        ),
        desired_state=merge_global_states_at_idx(
            old_scene_config.desired_state,
            new_scene_config.desired_state,
            reset_configs,
        ),
        scene_id=old_scene_config.scene_id,
        scene=old_scene_config.scene,
        gs_entities=old_scene_config.gs_entities,
        cameras=old_scene_config.cameras,
        batch_dim=old_scene_config.batch_dim,
        reward_history=old_scene_config.reward_history.merge_at_idx(
            new_scene_config.reward_history, reset_configs
        ),
    )

    # Update vox_init and vox_des for reset states
    if torch.any(reset_configs):
        merged_scene_config.vox_init[reset_configs] = new_scene_config.vox_init[
            reset_configs
        ]
        merged_scene_config.vox_des[reset_configs] = new_scene_config.vox_des[
            reset_configs
        ]

        # Update initial_diffs
        for i, reset in enumerate(reset_configs):
            if reset:
                for k in new_scene_config.initial_diffs.keys():
                    merged_scene_config.initial_diffs[k][i] = (
                        new_scene_config.initial_diffs[k][i]
                    )

        # Update initial_diff_counts
        merged_scene_config.initial_diff_counts[reset_configs] = (
            new_scene_config.initial_diff_counts[reset_configs]
        )

    return merged_scene_config
