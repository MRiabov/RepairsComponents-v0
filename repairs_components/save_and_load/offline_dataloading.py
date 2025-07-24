import copy
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
    split_scene_config,
)
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.progressive_reward_calc import RewardHistory
from repairs_components.training_utils.sim_state_global import (
    get_graph_save_paths,
    reconstruct_sim_state,
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

        self.loaded_pyg_batch_dict_mech_init = {}
        self.loaded_pyg_batch_dict_elec_init = {}
        self.loaded_pyg_batch_dict_mech_des = {}
        self.loaded_pyg_batch_dict_elec_des = {}

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
        # graphs
        mech_graphs_init, elec_graphs_init, mech_graphs_des, elec_graphs_des = (
            self._load_graphs(scene_id)
        )

        # tool idx
        tool_idx_path_init = self.data_dir / f"tool_idx_{scene_id}.pt"
        tool_idx_path_des = self.data_dir / f"tool_idx_{scene_id}.pt"
        tool_data_init = torch.load(tool_idx_path_init)
        tool_data_des = torch.load(tool_idx_path_des)

        # holes
        starting_hole_positions = torch.load(
            self.data_dir / f"scene_{scene_id}" / "starting_hole_positions.pt"
        )
        starting_hole_quats = torch.load(
            self.data_dir / f"scene_{scene_id}" / "starting_hole_quats.pt"
        )
        hole_depth = torch.load(self.data_dir / f"scene_{scene_id}" / "hole_depth.pt")
        part_hole_batch = torch.load(
            self.data_dir / f"scene_{scene_id}" / "part_hole_batch.pt"
        )
        hole_is_through = torch.load(
            self.data_dir / f"scene_{scene_id}" / "hole_is_through.pt"
        )
        # TODO update them during load based on pos.

        # load RepairsEnvState
        init_sim_state = reconstruct_sim_state(
            elec_graphs_init,
            mech_graphs_init,
            scene_metadata["electronics_indices"],
            scene_metadata["mechanical_indices"],
            tool_data_init,
            starting_hole_positions,
            starting_hole_quats,
        )
        des_sim_state = reconstruct_sim_state(
            elec_graphs_des,
            mech_graphs_des,
            scene_metadata["electronics_indices"],
            scene_metadata["mechanical_indices"],
            tool_data_des,
            starting_hole_positions,
            starting_hole_quats,
        )

        batch_dim = init_sim_state.scene_batch_dim

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
            starting_hole_positions=starting_hole_positions,
            starting_hole_quats=starting_hole_quats,
            hole_depth=hole_depth,
            part_hole_batch=part_hole_batch,
            hole_is_through=hole_is_through,
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

    def _load_graphs(
        self, scene_id: int, env_idx: torch.Tensor | None = None
    ) -> tuple[list[Data], list[Data], list[Data], list[Data]]:
        # ^ no point in converting lists of data to batches.
        """Load the graphs for a scene."""
        if (
            scene_id not in self.loaded_pyg_batch_dict_mech_init
            or scene_id not in self.loaded_pyg_batch_dict_elec_init
        ):
            # load initial and desired graphs: mech init, mech des, elec init, elec des
            # note: not in a for loop, but it's OK.
            mech_graph_init_path, elec_graph_init_path = get_graph_save_paths(
                self.data_dir, scene_id, init=True
            )
            mech_graph_des_path, elec_graph_des_path = get_graph_save_paths(
                self.data_dir, scene_id, init=False
            )

            graphs = []
            for path in [
                mech_graph_init_path,
                elec_graph_init_path,
                mech_graph_des_path,
                elec_graph_des_path,
            ]:
                assert path.exists(), f"Graph file not found at {path}"
                batch = torch.load(path)
                assert isinstance(batch, Batch), (
                    f"Graph under path {path} is not a Batch, it is {type(batch)}"
                )
                graphs.append(batch)

                assert batch.num_graphs == graphs[0].num_graphs, (
                    "Num graphs is expected to be equal for all graph batches."
                )
            (
                self.loaded_pyg_batch_dict_mech_init[scene_id],
                self.loaded_pyg_batch_dict_elec_init[scene_id],
                self.loaded_pyg_batch_dict_mech_des[scene_id],
                self.loaded_pyg_batch_dict_elec_des[scene_id],
            ) = graphs

        if env_idx is None:
            return (
                self.loaded_pyg_batch_dict_mech_init[scene_id].to_data_list(),
                self.loaded_pyg_batch_dict_elec_init[scene_id].to_data_list(),
                self.loaded_pyg_batch_dict_mech_des[scene_id].to_data_list(),
                self.loaded_pyg_batch_dict_elec_des[scene_id].to_data_list(),
            )
        else:
            return (
                self.loaded_pyg_batch_dict_mech_init[scene_id][env_idx],
                self.loaded_pyg_batch_dict_elec_init[scene_id][env_idx],
                self.loaded_pyg_batch_dict_mech_des[scene_id][env_idx],
                self.loaded_pyg_batch_dict_elec_des[scene_id][env_idx],
            )


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
        # Graphs
        graphs_dir = data_dir / "graphs"
        graph_files = [
            graphs_dir / f"mechanical_graphs_{scene_id}_init.pt",
            graphs_dir / f"electronics_graphs_{scene_id}_init.pt",
            graphs_dir / f"mechanical_graphs_{scene_id}_des.pt",
            graphs_dir / f"electronics_graphs_{scene_id}_des.pt",
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
