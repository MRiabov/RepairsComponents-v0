import copy
import json
from pathlib import Path
from typing import List

import numpy as np

import torch

from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
    split_scene_config,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
from repairs_components.training_utils.sim_state_global import (
    get_state_and_info_save_paths,
    RepairsSimState,
    RepairsSimInfo,
)

# During the loading of the data we load:
# 1. Graphs
# 2. Voxels

# We reconstruct:
# Reverse indices at electronics and mechanical states
# Genesis scene (from RepairsEnvState, which is in turn reconstructed from graphs.)


class OfflineDataloader:
    """Offline dataset for loading pre-generated environment configurations."""

    # note: I know that technically this isn't the most efficient way to do it - better
    # to do dataloader, prefetch etc; however with simulator step dominating the runtime,
    # I'd rather use the convenient API of getting by batch using num_envs_per_scene.
    # This is more practical because it's easy to reset the environment this way (in RL batch sizes vary greatly).
    # Plus it is consistent with the online dataloader.

    def __init__(self, data_dir: str | Path, scene_ids: List[int]) -> None:
        """Initialize the OfflineDataset.

        Args:
            data_dir: Directory containing the saved environment configurations.
            scene_ids: List of scene IDs to load.
        """  # TODO loading per separate tasks by name is not supported yet, but should be easy to implement via preprocess filtering.
        self.data_dir = Path(data_dir)
        self.scene_ids = scene_ids

        # Cached full states and info per scene
        self.loaded_init_states: dict[int, RepairsSimState] = {}
        self.loaded_des_states: dict[int, RepairsSimState] = {}
        self.loaded_sim_info: dict[int, RepairsSimInfo] = {}

        self.vox_init_dict = {}
        self.vox_des_dict = {}

        self.initial_diff_dict = {}  # dict[str(scene_id), dict[str(diff_type), Data]]
        self.initial_diff_count_dict = {}  # dict[str(scene_id), torch.Tensor]

        self.metadata = {}

        # Load metadata
        for scene_id in scene_ids:
            metadata_path = self.data_dir / f"scene_{scene_id}" / "metadata.json"
            assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"

            with open(metadata_path, "r") as f:
                self.metadata["scene_" + str(scene_id)] = json.load(f)
        # Load diffs

        for scene_id in scene_ids:
            self.initial_diff_dict[scene_id] = torch.load(
                self.data_dir / f"scene_{scene_id}" / "initial_diffs.pt"
            )
            self.initial_diff_count_dict[scene_id] = torch.load(
                self.data_dir / f"scene_{scene_id}" / "initial_diff_counts.pt"
            )

        # Load all scene configs
        self.scene_configs = []
        for scene_id in scene_ids:
            # get this scene metadata
            scene_metadata = self.metadata["scene_" + str(scene_id)]
            init_scene_config = self.load_scene_config(scene_metadata)
            self.scene_configs.append(init_scene_config)

    def get_processed_offline_data(
        self, num_envs_per_scene: torch.Tensor
    ) -> List[ConcurrentSceneData]:
        """Get a list of batched scene configurations as ConcurrentSceneData objects.

        Args:
            num_envs_per_scene: Tensor containing the number of environments to load per scene id.

        Returns:
            A list of ConcurrentSceneData objects containing the scene configurations.

        Raises:
            RuntimeError: If the scene configuration is invalid or missing required fields.
        """
        assert len(num_envs_per_scene) == len(self.scene_configs), (
            "num_envs_per_scene must have the same length as self.scene_configs"
        )

        # note: I'm certain there was a better way to do this, but I can't find it.
        all_cfgs = []
        for scene_id, num_envs_to_get in enumerate(num_envs_per_scene):
            if num_envs_to_get == 0:
                all_cfgs.append(None)
                continue
            config = self.scene_configs[scene_id]
            all_cfgs_this_scene = split_scene_config(config)
            env_ids = np.random.randint(
                low=0, high=config.batch_dim, size=(num_envs_to_get.item(),)
            )
            cfgs_this_scene = []
            for env_id in env_ids:
                cfgs_this_scene.append(all_cfgs_this_scene[env_id])
            all_cfgs.append(cfgs_this_scene)
        return all_cfgs

    def load_scene_config(self, scene_metadata: dict) -> ConcurrentSceneData:
        """Load a single scene configuration from disk."""
        scene_id = scene_metadata["scene_id"]

        # voxels
        vox_init_path = self.data_dir / "voxels" / f"vox_init_{scene_id}.pt"
        vox_des_path = self.data_dir / "voxels" / f"vox_des_{scene_id}.pt"

        # Load sparse tensors # note: small enough to fit into memory, yet. (36mb estimated)

        self.vox_init_dict[scene_id] = torch.load(vox_init_path)
        self.vox_des_dict[scene_id] = torch.load(vox_des_path)

        print(
            f"Loaded vox init dict for scene_id {scene_id} with shape: {self.vox_init_dict[scene_id].shape}",
        )
        # ^ memory calculation: 100k samples*max 15 items * 10 datapoints * float16 =36mil = 36mbytes
        # states and info
        init_state, des_state, sim_info = self._load_sim_states(scene_id)

        # Use loaded full states directly
        init_sim_state = init_state
        des_sim_state = des_state

        batch_dim = init_sim_state.batch_size

        sim_state = ConcurrentSceneData(
            scene=None,
            gs_entities=None,
            init_state=init_sim_state,
            current_state=copy.deepcopy(init_sim_state),
            desired_state=des_sim_state,
            vox_init=self.vox_init_dict[scene_id],
            vox_des=self.vox_des_dict[scene_id],
            initial_diffs=self.initial_diff_dict[scene_id],
            initial_diff_counts=self.initial_diff_count_dict[scene_id],
            scene_id=scene_id,
            reward_history=RewardHistory(batch_dim=batch_dim),
            batch_dim=batch_dim,
            task_ids=torch.zeros(batch_dim, dtype=torch.int32),
            step_count=torch.zeros(batch_dim, dtype=torch.int32),
            sim_info=sim_info,
        )

        return sim_state

    def _load_sparse_tensor(
        self,
        path: Path,
        expected_batch_dim: int | None = None,
    ) -> torch.Tensor:
        """Load a sparse tensor from disk."""
        tensor = torch.load(path)
        assert isinstance(tensor, torch.Tensor), "Tensor must be a torch tensor"
        assert expected_batch_dim is None or tensor.shape[0] == expected_batch_dim, (
            "Tensor must have the expected batch dim"
        )
        return tensor

    def _load_sim_states(
        self, scene_id: int, env_idx: torch.Tensor | None = None
    ) -> tuple[RepairsSimState, RepairsSimState, RepairsSimInfo]:
        """Load full sim states and info for a scene."""
        if (
            scene_id not in self.loaded_init_states
            or scene_id not in self.loaded_des_states
        ):
            state_init_path, info_path = get_state_and_info_save_paths(
                self.data_dir, scene_id, init=True
            )
            state_des_path, _ = get_state_and_info_save_paths(
                self.data_dir, scene_id, init=False
            )

            assert state_init_path.exists(), (
                f"State file not found at {state_init_path}"
            )
            assert state_des_path.exists(), f"State file not found at {state_des_path}"
            assert info_path.exists(), f"Info file not found at {info_path}"

            init_state = torch.load(state_init_path)
            des_state = torch.load(state_des_path)
            sim_info = torch.load(info_path)

            assert isinstance(init_state, RepairsSimState), (
                f"Expected RepairsSimState at {state_init_path}, got {type(init_state)}"
            )
            assert isinstance(des_state, RepairsSimState), (
                f"Expected RepairsSimState at {state_des_path}, got {type(des_state)}"
            )
            assert isinstance(sim_info, RepairsSimInfo), (
                f"Expected RepairsSimInfo at {info_path}, got {type(sim_info)}"
            )

            self.loaded_init_states[scene_id] = init_state
            self.loaded_des_states[scene_id] = des_state
            self.loaded_sim_info[scene_id] = sim_info

        init_state = self.loaded_init_states[scene_id]
        des_state = self.loaded_des_states[scene_id]
        sim_info = self.loaded_sim_info[scene_id]

        if env_idx is None:
            return init_state, des_state, sim_info
        else:
            # Slice states by env_idx; sim_info is global/static
            return init_state[env_idx], des_state[env_idx], sim_info


def check_if_data_exists(
    scene_ids: list[int],
    data_dir: Path,
    count_envs_per_scene: torch.Tensor,
    env_setups: list[EnvSetup],
) -> bool:
    """
    Check if all required data files exist for each scene_id in data_dir.
    Returns True if all exist, else False.
    """
    data_dir = Path(data_dir)

    # Check per-scene files
    for scene_id in scene_ids:
        metadata_path = data_dir / f"scene_{scene_id}" / "metadata.json"
        if not metadata_path.exists():
            return False
        # States and info
        state_dir = data_dir / "state"
        graph_files = [
            state_dir / f"state_{scene_id}_init.pt",
            state_dir / f"info_{scene_id}_init.pt",
            state_dir / f"state_{scene_id}_des.pt",
            state_dir / f"info_{scene_id}_des.pt",
        ]
        # Voxels
        voxels_dir = data_dir / "voxels"
        voxel_files = [
            voxels_dir / f"vox_init_{scene_id}.pt",
            voxels_dir / f"vox_des_{scene_id}.pt",
        ]
        # holes
        holes_dir = data_dir / f"scene_{scene_id}"
        holes_files = [
            holes_dir / "starting_hole_positions.pt",
            holes_dir / "starting_hole_quats.pt",
            holes_dir / "hole_depth.pt",
            holes_dir / "part_hole_batch.pt",
            holes_dir / "hole_is_through.pt",
        ]
        # Check all
        if not all(
            file_path.exists() for file_path in graph_files + voxel_files + holes_files
        ):
            return False

        # assert that there is enough data:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if metadata["count_generated_envs"] < count_envs_per_scene[scene_id]:
            print(
                f"Found less data than expected for scene_id: {scene_id}. Found {metadata['count_generated_envs']}, expected {count_envs_per_scene[scene_id]}. Regenerating..."
            )
            return False
        elif metadata["count_generated_envs"] > count_envs_per_scene[scene_id] * 3:
            print(
                "Note: found at least 3 times more data than requested. This may strain the memory. "
                "Consider regenerating data with a requested size."
            )
        if metadata["env_setup_name"] != env_setups[scene_id].__class__.__name__:
            print(
                f"Found data for different environment setup than requested. Found {metadata['env_setup_name']}, expected {env_setups[scene_id].__class__.__name__}. Regenerating..."
            )
            return False

    return True  # TODO it would be ideal to not regenerate data for all environments every time, but it's quick enough (for one-time op)


def get_scene_mesh_file_names(
    scene_ids: list[int], data_dir: Path, append_path: bool = True
) -> list[dict[str, str]]:
    """Get the file names of the meshes to their scene names for the given scene ids."""
    data_dir = Path(data_dir)
    mesh_file_names = []

    # note: all these jsons are probably (?) better be done as a single file.
    for scene_id in scene_ids:
        metadata_path = data_dir / f"scene_{scene_id}" / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # append full path if append_path is True
        if append_path:
            mesh_file_names.append(
                {
                    k: data_dir / f"scene_{scene_id}" / v
                    for k, v in metadata["mesh_file_names"].items()
                }
            )  # return path
        else:
            mesh_file_names.append(metadata["mesh_file_names"])

    return mesh_file_names
