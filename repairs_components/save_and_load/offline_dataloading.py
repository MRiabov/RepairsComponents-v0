from pathlib import Path
from typing import List

import numpy as np
import torch

from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.geometry.fasteners import (
    get_fastener_save_path_from_name,
    get_fastener_singleton_name,
)
from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    split_scene_config,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
from repairs_components.training_utils.sim_state_global import (
    RepairsSimInfo,
    RepairsSimState,
    get_state_and_info_save_paths,
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
        # Load diffs

        for scene_id in scene_ids:
            self.initial_diff_dict[scene_id] = torch.load(
                self.data_dir / f"scene_{scene_id}" / "initial_diffs.pt"
            )
            self.initial_diff_count_dict[scene_id] = torch.load(
                self.data_dir / f"scene_{scene_id}" / "initial_diff_counts.pt"
            )

        # Load all scene configs (no metadata.json; scene_id is sufficient)
        self.scene_configs = []
        for scene_id in scene_ids:
            init_scene_config = self.load_scene_config(scene_id)
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

    def load_scene_config(self, scene_id: int) -> ConcurrentSceneData:
        """Load a single scene configuration from disk."""

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

        # Normalize batch_dim to plain int for downstream constructors (e.g., RewardHistory, torch.arange)
        batch_dim = (
            int(init_sim_state.batch_size)
            if isinstance(init_sim_state.batch_size, int)
            else int(init_sim_state.batch_size[0])
        )

        sim_state = ConcurrentSceneData(
            scene=None,
            gs_entities=None,
            init_state=init_sim_state,
            current_state=init_sim_state.clone(recurse=True),
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

        # # Backward compatibility: ensure mesh_file_names live under physical_info
        # # 1) Migrate from old RepairsSimInfo.mesh_file_names if present
        # if hasattr(sim_info, "mesh_file_names") and sim_info.mesh_file_names:
        #     sim_info.physical_info.mesh_file_names = dict(sim_info.mesh_file_names)
        #     # do not delete attribute to keep unpickling stability

        # # 2) If still empty, reconstruct from disk
        # if not sim_info.physical_info.mesh_file_names:
        #     mapping_list = get_scene_mesh_file_names([scene_id], self.data_dir)
        #     sim_info.physical_info.mesh_file_names = (
        #         mapping_list[0] if mapping_list else {}
        #     )

        assert (
            sim_info.physical_info.mesh_file_names is not None
            and len(sim_info.physical_info.mesh_file_names) > 0
        ), f"Mesh file names not found for scene {scene_id}"

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
        # States and info
        state_dir = data_dir / "state"
        state_and_info_files = [
            state_dir / f"state_{scene_id}_init.pt",
            state_dir / f"state_{scene_id}_des.pt",
            state_dir / f"info_{scene_id}.pt",
        ]
        # Voxels
        voxels_dir = data_dir / "voxels"
        voxel_files = [
            voxels_dir / f"vox_init_{scene_id}.pt",
            voxels_dir / f"vox_des_{scene_id}.pt",
        ]
        # Check all
        if not all(
            file_path.exists() for file_path in state_and_info_files + voxel_files
        ):
            return False

        # Verify batch count using saved init state (metadata.json deprecated)
        init_state_path = state_dir / f"state_{scene_id}_init.pt"
        with open(init_state_path, "rb") as f:
            init_state = torch.load(f)
            sim_info = torch.load(state_dir / f"info_{scene_id}.pt")

        assert init_state.ndim == 1, "Expected init_state to be persisted as 1ndim."
        batch_dim = int(init_state.batch_size[0])

        if batch_dim < int(count_envs_per_scene[scene_id]):
            print(
                f"Found less data than expected for scene_id: {scene_id}. Found {batch_dim}, expected {int(count_envs_per_scene[scene_id])}. Regenerating..."
            )
            return False
        elif batch_dim > int(count_envs_per_scene[scene_id]) * 3:
            print(
                "Note: found at least 3 times more data than requested. This may strain the memory. "
                "Consider regenerating data with a requested size."
            )
        if sim_info.env_setup_name != env_setups[scene_id].__class__.__name__:
            print(
                f"Found data for different environment setup than requested. Found {sim_info.env_setup_name}, expected {env_setups[scene_id].__class__.__name__}. Regenerating..."
            )
            return False

    return True  # TODO it would be ideal to not regenerate data for all environments every time, but it's quick enough (for one-time op)


def get_scene_mesh_file_names(
    scene_ids: list[int], data_dir: Path, append_path: bool = True
) -> list[dict[str, str]]:
    """Get the file names of the meshes to their scene names for the given scene ids."""
    data_dir = Path(data_dir)
    mesh_file_names: list[dict[str, str]] = []

    for scene_id in scene_ids:
        # Load sim_info saved once per scene
        # 'init' flag does not affect info path (info is shared), but function requires it
        # Pass base data_dir directly; get_state_and_info_save_paths appends 'state' internally
        _, info_path = get_state_and_info_save_paths(data_dir, scene_id, init=True)
        sim_info: RepairsSimInfo = torch.load(
            info_path, map_location="cpu", weights_only=False
        )
        physical_info = sim_info.physical_info

        mapping: dict[str, Path] = {}

        # Bodies and connectors
        for body_name in physical_info.body_indices.keys():
            assert "@" in body_name, "Label must contain '@'"
            part_name, part_type = body_name.split("@", 1)
            part_type = part_type.lower()

            if part_type in ("solid", "fixed_solid"):
                p = data_dir / f"scene_{scene_id}" / f"{body_name}.glb"
            elif part_type == "connector":
                p = Connector.save_path_from_name(data_dir, body_name, suffix="glb")
            elif part_type in ("button", "led", "switch"):
                # electronics are exported as MJCF
                p = data_dir / f"scene_{scene_id}" / f"{body_name}.xml"
            elif part_type == "fastener":
                # fasteners are singleton assets and handled separately below
                continue
            else:
                raise NotImplementedError(
                    f"Not implemented for part type in name: {body_name}"
                )

            mapping[body_name] = p

        # Fastener singleton assets (dimensions are static per scene)
        diam = physical_info.fasteners_diam
        length = physical_info.fasteners_length
        if isinstance(diam, torch.Tensor) and diam.numel() > 0:
            # fastener metadata tensors are singleton 1D
            assert diam.ndim == 1 and length.ndim == 1
            assert diam.shape == length.shape
            for i in range(diam.shape[0]):
                name = get_fastener_singleton_name(
                    float(diam[i].item() * 1000),
                    float(length[i].item() * 1000),
                )
                mapping[name] = get_fastener_save_path_from_name(name, data_dir)

        # Append as absolute or relative paths
        if append_path:
            mesh_file_names.append({k: str(v) for k, v in mapping.items()})
        else:

            def _rel(p: Path) -> str:
                try:
                    return str(p.relative_to(data_dir))
                except Exception:
                    return str(p)

            mesh_file_names.append({k: _rel(v) for k, v in mapping.items()})

    return mesh_file_names
