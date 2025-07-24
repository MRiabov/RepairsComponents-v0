from dataclasses import dataclass

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.vis.camera import Camera
from torch_geometric.data import Data

from repairs_components.processing.voxel_export import sparse_arr_put
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
from repairs_components.training_utils.sim_state_global import (
    RepairsSimState,
    merge_global_states,
    merge_global_states_at_idx,
)


@dataclass
class ConcurrentSceneData:
    """Dataclass for concurrent scene data.
    Each entry is batched with a single batch dim, calculated
    as batch_dim // concurrent_scenes (except scene and gs_entities
    since they are singletons.)"""

    scene: gs.Scene
    gs_entities: dict[str, RigidEntity]
    init_state: RepairsSimState
    current_state: RepairsSimState
    desired_state: RepairsSimState
    vox_init: torch.Tensor  # sparse tensor!
    vox_des: torch.Tensor  # sparse tensor!
    initial_diffs: dict[str, list[Data]]
    initial_diff_counts: torch.Tensor  # shape: (batch_dim // concurrent_scenes,)
    scene_id: int
    "A safety int to ensure we don't access the wrong scene."
    reward_history: RewardHistory
    batch_dim: int
    """Batch dimension of this scene, must match length of all lists/tensors in this dataclass.
    Primarily for sanity checks."""
    step_count: torch.IntTensor
    "Step count in every scene. I don't think this should be diffed."
    task_ids: torch.IntTensor
    "Task ids in for every scene."

    # holes
    starting_hole_positions: torch.Tensor
    """Starting hole positions for every part, batched with part_hole_batch. 
    Equal over the batch. Shape: (H, 3)"""
    starting_hole_quats: torch.Tensor
    """Starting hole quats for every part, batched with part_hole_batch. 
    Equal over the batch. Shape: (H, 4)"""
    hole_depth: torch.Tensor
    """Hole depths for every part.
    Equal over the batch. Shape: (H)"""
    hole_is_through: torch.Tensor
    """Boolean mask indicating whether each hole is through (True) or blind (False).
    Equal over the batch. Shape: (H)"""
    part_hole_batch: torch.Tensor
    "Part index for every hole. Equal over the batch. Shape: (H)"

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
        assert self.task_ids.shape[0] == self.batch_dim, (
            f"task_ids must have the same batch dimension as batch_dim, but got {self.task_ids.shape[0]} and {self.batch_dim}"
        )
        if self.step_count.ndim == 0:
            self.step_count = torch.zeros(self.batch_dim, dtype=torch.int)
        else:
            assert self.step_count.shape[0] == self.batch_dim, (
                f"step_count must have the same batch dimension as batch_dim, but got {self.step_count.shape[0]} and {self.batch_dim}"
            )
        # diffs
        assert isinstance(self.initial_diffs, dict), "initial_diffs must be a dict"
        assert isinstance(self.initial_diffs["physical_diff"], list), (
            "initial_diffs must be a list"
        )
        assert isinstance(self.initial_diffs["electronics_diff"], list), (
            "initial_diffs must be a list"
        )
        assert len(self.initial_diffs["physical_diff"]) == self.batch_dim, (
            f"initial_diffs must have the same length as batch_dim={self.batch_dim}, but got {len(self.initial_diffs)}"
        )
        assert len(self.initial_diffs["electronics_diff"]) == self.batch_dim, (
            f"initial_diffs must have the same length as batch_dim={self.batch_dim}, but got {len(self.initial_diffs)}"
        )
        assert isinstance(self.initial_diff_counts, torch.Tensor), (
            "initial_diff_counts must be a torch tensor"
        )
        assert self.initial_diff_counts.shape[0] == self.batch_dim, (
            f"initial_diff_counts must have the same batch dimension as batch_dim, but got {self.initial_diff_counts.shape[0]} and {self.batch_dim}"
        )
        # holes
        self.hole_count = self.starting_hole_positions.shape[1]
        assert self.starting_hole_positions.shape == (
            self.batch_dim,
            self.hole_count,
            3,
        ), (
            f"Starting hole positions must have shape ({self.batch_dim}, {self.hole_count}, 3), but got {self.starting_hole_positions.shape}"
        )
        assert self.starting_hole_quats.shape == (
            self.batch_dim,
            self.hole_count,
            4,
        ), (
            f"Starting hole quats must have shape ({self.batch_dim}, {self.hole_count}, 4), but got {self.starting_hole_quats.shape}"
        )
        assert self.part_hole_batch.shape == (self.hole_count,), (
            "Part hole batch must have shape (H,)"
        )
        assert self.hole_depth.shape == (self.hole_count,), (
            "Hole depths must have shape (H,)"
        )
        assert (self.hole_depth > 0).all(), "Hole depths must be positive."
        assert self.hole_is_through.shape == (self.hole_count,), (
            "Hole is through must have shape (H,)"
        )

        # init state != current state
        # assert self.init_state != self.current_state, (
        #     "init_state must not be equal to current_state. If it is, use `copy.copy` to create a new object."
        # ) # note: it fails because we copy the initial state, but use copy anyway. deepcopy maybe?


def merge_concurrent_scene_configs(scene_configs: list[ConcurrentSceneData]):
    assert scene_configs is not None and len(scene_configs) > 0, (
        "Can't merge 0 scene configs"
    )
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
    assert all(
        scene_configs[0].starting_hole_positions.keys()
        == scene_cfg.starting_hole_positions.keys()
        for scene_cfg in scene_configs
    )
    assert all(
        scene_configs[0].starting_hole_quats.keys()
        == scene_cfg.starting_hole_quats.keys()
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

    new_batch_dim = sum([data.batch_dim for data in scene_configs])
    # Cat diffs
    physical_diffs = []
    electronics_diffs = []
    fluid_diffs = []
    for data in scene_configs:
        physical_diffs.extend(data.initial_diffs["physical_diff"])
        electronics_diffs.extend(data.initial_diffs["electronics_diff"])
        fluid_diffs.extend(data.initial_diffs["fluid_diff"])
    initial_diffs = {
        "physical_diff": physical_diffs,
        "electronics_diff": electronics_diffs,
        "fluid_diff": fluid_diffs,
    }
    starting_hole_positions = {}
    starting_hole_quats = {}

    for data in scene_configs:
        # cat per key
        for key in data.starting_hole_positions.keys():
            starting_hole_positions[key] = torch.cat(
                [data.starting_hole_positions[key] for data in scene_configs], dim=0
            )
            starting_hole_quats[key] = torch.cat(
                [data.starting_hole_quats[key] for data in scene_configs], dim=0
            )

    # Extend tensors and RepairsSimState with items from other scene_configs

    new_scene_config = ConcurrentSceneData(
        vox_init=vox_init,
        vox_des=vox_des,
        initial_diffs=initial_diffs,
        initial_diff_counts=torch.cat(
            [data.initial_diff_counts for data in scene_configs], dim=0
        ),
        init_state=merge_global_states(
            [scene_cfg.init_state for scene_cfg in scene_configs]
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
        batch_dim=new_batch_dim,
        reward_history=reward_history,
        step_count=torch.zeros(new_batch_dim, dtype=torch.int),
        task_ids=torch.cat([data.task_ids for data in scene_configs], dim=0),
        starting_hole_positions=starting_hole_positions,
        starting_hole_quats=starting_hole_quats,
        hole_depth=scene_configs[0].hole_depth,
        part_hole_batch=scene_configs[0].part_hole_batch,
        hole_is_through=scene_configs[0].hole_is_through,
    )
    return new_scene_config


def merge_scene_configs_at_idx(
    old_scene_config: ConcurrentSceneData,
    new_scene_config: ConcurrentSceneData,
    reset_mask: torch.BoolTensor,
) -> "ConcurrentSceneData":
    """Insert new scene configs at indices indicated by bool tensor `reset_mask`.

    Args:
        old_scene_config: The original scene configuration
        new_scene_config: The new scene configuration to merge from
        reset_mask: Boolean tensor indicating which indices to update from new_scene_config

    Returns:
        A new ConcurrentSceneData with merged values
    """
    batch_dim = len(old_scene_config.vox_init)
    assert len(reset_mask) == batch_dim, (
        "Reset states must have the same length as batch dimension."
    )
    assert reset_mask.dtype == torch.bool, "Reset states must be a bool tensor."
    assert reset_mask.ndim == 1, "Reset states must be a 1D tensor."
    assert new_scene_config.batch_dim == reset_mask.int().sum(), (
        "Count of reset configs must be equal to the batch dimension of the incoming configs."
    )
    assert (
        old_scene_config.starting_hole_positions.keys()
        == new_scene_config.starting_hole_positions.keys()
    ), "Starting hole positions must have the same keys."
    if torch.any(reset_mask):
        old_idx = torch.nonzero(reset_mask).squeeze(1)
        new_idx = torch.arange(len(old_idx))
        # Create a new scene config with the same structure as the old one
        merged_scene_config = ConcurrentSceneData(
            vox_init=old_scene_config.vox_init.clone(),
            vox_des=old_scene_config.vox_des.clone(),
            initial_diffs=old_scene_config.initial_diffs,
            initial_diff_counts=old_scene_config.initial_diff_counts.clone(),
            init_state=merge_global_states_at_idx(
                old_scene_config.init_state,
                new_scene_config.init_state,
                reset_mask,
            ),
            current_state=merge_global_states_at_idx(
                old_scene_config.current_state,
                new_scene_config.current_state,
                reset_mask,
            ),
            desired_state=merge_global_states_at_idx(
                old_scene_config.desired_state,
                new_scene_config.desired_state,
                reset_mask,
            ),
            scene_id=old_scene_config.scene_id,
            scene=old_scene_config.scene,
            gs_entities=old_scene_config.gs_entities,
            batch_dim=old_scene_config.batch_dim,
            reward_history=old_scene_config.reward_history.merge_at_idx(
                new_scene_config.reward_history, old_idx
            ),
            step_count=old_scene_config.step_count.clone(),
            task_ids=old_scene_config.task_ids.clone(),
            starting_hole_positions=old_scene_config.starting_hole_positions,
            starting_hole_quats=old_scene_config.starting_hole_quats,
            hole_depth=old_scene_config.hole_depth,
            part_hole_batch=old_scene_config.part_hole_batch,
            hole_is_through=old_scene_config.hole_is_through,
        )
        # Update vox_init and vox_des for reset states
        merged_scene_config.vox_init = sparse_arr_put(
            merged_scene_config.vox_init, new_scene_config.vox_init, old_idx, dim=0
        )
        merged_scene_config.vox_des = sparse_arr_put(
            merged_scene_config.vox_des, new_scene_config.vox_des, old_idx, dim=0
        )

        # Update initial_diffs
        for new_id, old_id in enumerate(old_idx):
            for k in new_scene_config.initial_diffs:
                merged_scene_config.initial_diffs[k][old_id] = (
                    new_scene_config.initial_diffs[k][new_id]
                )

        # Update initial_diff_counts
        merged_scene_config.initial_diff_counts[old_idx] = (
            new_scene_config.initial_diff_counts.to(
                old_scene_config.initial_diff_counts.device
            )
        )
        merged_scene_config.step_count[old_idx] = new_scene_config.step_count.to(
            old_scene_config.step_count.device
        )
        merged_scene_config.task_ids[old_idx] = new_scene_config.task_ids.to(
            old_scene_config.task_ids.device
        )
        # holes are unupdated because they are static throughout scene.
    return merged_scene_config


def split_scene_config(scene_config: ConcurrentSceneData):
    batch = scene_config.batch_dim
    cfg_list: list[ConcurrentSceneData] = []
    for i in range(batch):
        # slice global states
        orig_curr = scene_config.current_state
        curr = RepairsSimState(batch_dim=1)
        curr.electronics_state = [orig_curr.electronics_state[i]]
        curr.physical_state = [orig_curr.physical_state[i]]
        curr.fluid_state = [orig_curr.fluid_state[i]]
        curr.tool_state = [orig_curr.tool_state[i]]
        curr.has_electronics = orig_curr.has_electronics
        curr.has_fluid = orig_curr.has_fluid
        # sanity check: ensure single-item state
        assert curr.scene_batch_dim == 1, (
            f"Expected batch_dim=1, got {curr.scene_batch_dim}"
        )

        orig_des = scene_config.desired_state
        des = RepairsSimState(batch_dim=1)
        des.electronics_state = [orig_des.electronics_state[i]]
        des.physical_state = [orig_des.physical_state[i]]
        des.fluid_state = [orig_des.fluid_state[i]]
        des.tool_state = [orig_des.tool_state[i]]
        des.has_electronics = orig_des.has_electronics
        des.has_fluid = orig_des.has_fluid

        orig_init = scene_config.init_state
        init = RepairsSimState(batch_dim=1)
        init.electronics_state = [orig_init.electronics_state[i]]
        init.physical_state = [orig_init.physical_state[i]]
        init.fluid_state = [orig_init.fluid_state[i]]
        init.tool_state = [orig_init.tool_state[i]]
        init.has_electronics = orig_init.has_electronics
        init.has_fluid = orig_init.has_fluid
        # sanity check: ensure single-item state
        assert des.scene_batch_dim == 1, (
            f"Expected batch_dim=1, got {des.scene_batch_dim}"
        )

        # slice voxel and diffs
        vox_init_i = scene_config.vox_init[i].unsqueeze(0)
        vox_des_i = scene_config.vox_des[i].unsqueeze(0)
        diffs_i = (
            {k: [scene_config.initial_diffs[k][i]] for k in scene_config.initial_diffs}
            if scene_config.initial_diffs is not None
            else None
        )  # ^note: put to list.
        diff_counts_i = (
            scene_config.initial_diff_counts[i : i + 1]
            if scene_config.initial_diff_counts is not None
            else None
        )

        cfg_list.append(
            ConcurrentSceneData(
                scene=scene_config.scene,
                gs_entities=scene_config.gs_entities,
                init_state=init,
                current_state=curr,
                desired_state=des,
                vox_init=vox_init_i,
                vox_des=vox_des_i,
                initial_diffs=diffs_i,
                initial_diff_counts=diff_counts_i,
                scene_id=scene_config.scene_id,
                batch_dim=1,
                reward_history=RewardHistory(batch_dim=1),
                step_count=scene_config.step_count[i],
                task_ids=scene_config.task_ids[i : i + 1],
                # holes are uniform throughout the batch
                starting_hole_positions=scene_config.starting_hole_positions,
                starting_hole_quats=scene_config.starting_hole_quats,
                hole_depth=scene_config.hole_depth,
                part_hole_batch=scene_config.part_hole_batch,
                hole_is_through=scene_config.hole_is_through,
            )
        )
    return cfg_list
