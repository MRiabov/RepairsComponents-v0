import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch_geometric.data import Batch

from repairs_components.training_utils.concurrent_scene_dataclass import (
    ConcurrentSceneData,
    merge_concurrent_scene_configs,
)
from repairs_components.training_utils.sim_state_global import (
    RepairsSimState,
)

# During the loading of the data we load:
# 1. Graphs
# 2. Voxels

# We reconstruct:
# Reverse indices at electronics and mechanical states
# Genesis scene (from RepairsEnvState, which is in turn reconstructed from graphs.)


class OfflineDataset:
    """Offline dataset for loading pre-generated environment configurations."""

    # note: I know that technically this isn't the most efficient way to do it - better
    # to do dataloader, prefetch etc; however with simulator step dominating the runtime,
    # I'd rather use the convenient API of getting by batch using num_envs_per_scene.
    # This is more practical because it's easy to reset the environment this way (in RL batch sizes vary greatly).
    # Plus it is consistent with the online dataloader.

    def __init__(
        self,
        data_dir: Union[str, Path],
        scene_ids: List[int],
    ) -> None:
        """Initialize the OfflineDataset.

        Args:
            data_dir: Directory containing the saved environment configurations.
            scene_ids: List of scene IDs to load.
        """  # TODO loading per separate tasks by name is not supported yet, but should be easy to implement via preprocess filtering.
        self.data_dir = Path(data_dir)
        self.scene_ids = scene_ids

        # Load metadata
        metadata_path = self.data_dir / "scenes_metadata.json"
        assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"

        self.loaded_pyg_batch_dict_mech = {}
        self.loaded_pyg_batch_dict_elec = {}

        self.vox_init_dict = {}
        self.vox_des_dict = {}

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load all scene configs
        self.scene_configs = []
        for scene_id in scene_ids:
            # get this scene metadata
            scene_metadata = self.metadata["scene_" + str(scene_id)]
            scene_config = self.load_scene_config(scene_metadata)
            self.scene_configs.append(scene_config)

    def get_processed_data(
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
            all_cfgs_this_scene = self.scene_configs[scene_id]
            env_ids = np.random.randint(
                low=0, high=len(all_cfgs_this_scene), size=num_envs_to_get
            )
            cfgs_this_scene = []
            for env_id in env_ids:
                cfgs_this_scene.append(all_cfgs_this_scene[env_id])
            all_cfgs.append(merge_concurrent_scene_configs(cfgs_this_scene))
        return all_cfgs

    def load_scene_config(self, scene_metadata: dict) -> Dict[str, Any]:
        """Load a single scene configuration from disk."""
        scene_id = scene_metadata["scene_id"]
        vox_init_path = self.data_dir / "voxels" / f"vox_init_{scene_id}.npz"
        vox_des_path = self.data_dir / "voxels" / f"vox_des_{scene_id}.npz"
        # Load sparse tensors # note: small enough to fit into memory, yet. (36mb estimated)

        self.vox_init_dict[scene_id] = self._load_sparse_tensor(vox_init_path)
        self.vox_des_dict[scene_id] = self._load_sparse_tensor(vox_des_path)
        # ^ memory calculation: 100k samples*max 15 items * 10 datapoints * float16 =36mil = 36mbytes
        scene_data_path = self.data_dir / ("scene_data_" + str(scene_id))

        # load RepairsEnvState
        RepairsEnvState()

        assert scene_data["batch_dim"] == self.vox_init_dict[scene_id].shape[0], (
            f"Batch dim of vox_init and scene_data do not match. "
            f"Expected {self.vox_init_dict[scene_id].shape[0]}, got {scene_data['batch_dim']}"
        )

        return scene_data

    def _load_sparse_tensor(
        self, npz_path: Path, expected_batch_dim: int | None = None, dtype=torch.float16
    ) -> torch.Tensor:
        """Load a sparse tensor from disk."""
        import numpy

        arrays_npz = numpy.load(npz_path)
        indices = arrays_npz["indices"]
        values = arrays_npz["values"]
        tensor_size = (indices.shape[0], *arrays_npz["tensor_size"])
        if expected_batch_dim is not None:
            assert indices.shape[0] == expected_batch_dim, (
                f"Expected batch dim {expected_batch_dim}, got {indices.shape[0]}"
            )
            assert numpy.unique(indices[:, 0]).shape[0] == expected_batch_dim, (
                f"Expected batch dim {expected_batch_dim}, got {numpy.unique(indices[:, 0]).shape[0]}"
            )

        indices = torch.from_numpy(indices)
        values = torch.from_numpy(values)

        return torch.sparse_coo_tensor(
            indices, values, size=tensor_size, dtype=dtype
        ).coalesce()

    def _load_graphs(
        self, scene_id: int, env_idx: torch.Tensor | None
    ) -> tuple[Batch, Batch]:
        """Load the graphs for a scene."""
        if (
            scene_id not in self.loaded_pyg_batch_dict_mech
            or scene_id not in self.loaded_pyg_batch_dict_elec
        ):
            mech_graph_path = (
                self.data_dir / "graphs" / f"mechanical_graphs_{scene_id}.pt"
            )
            elec_graph_path = (
                self.data_dir / "graphs" / f"electronic_graphs_{scene_id}.pt"
            )
            assert mech_graph_path.exists(), (
                f"Mechanical graph file not found at {mech_graph_path}"
            )
            assert elec_graph_path.exists(), (
                f"Electronic graph file not found at {elec_graph_path}"
            )

            loaded_graphs_mech = Batch.from_data_list(torch.load(mech_graph_path))
            loaded_graphs_elec = Batch.from_data_list(torch.load(elec_graph_path))

            assert loaded_graphs_mech.num_graphs == loaded_graphs_elec.num_graphs, (
                "Num graphs in mechanical and electronic batches do not match."
            )
            assert loaded_graphs_mech[0].scene_id == scene_id, (
                f"The loaded (mechanical) dataset graph does not seem to belong to this scene. "
                f"Expected {scene_id}, got {loaded_graphs_mech[0].scene_id}"
            )
            assert loaded_graphs_elec[0].scene_id == scene_id, (
                f"The loaded (electronic) dataset graph does not seem to belong to this scene. "
                f"Expected {scene_id}, got {loaded_graphs_elec[0].scene_id}"
            )

            self.loaded_pyg_batch_dict_mech[scene_id] = loaded_graphs_mech
            self.loaded_pyg_batch_dict_elec[scene_id] = loaded_graphs_elec

        if env_idx is not None:
            loaded_graphs_mech = self.loaded_pyg_batch_dict_mech[scene_id][env_idx]
            loaded_graphs_elec = self.loaded_pyg_batch_dict_elec[scene_id][env_idx]

        return loaded_graphs_mech, loaded_graphs_elec
