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
from repairs_components.training_utils.sim_state_global import reconstruct_sim_state

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

        self.loaded_pyg_batch_dict_mech_init = {}
        self.loaded_pyg_batch_dict_elec_init = {}
        self.loaded_pyg_batch_dict_mech_des = {}
        self.loaded_pyg_batch_dict_elec_des = {}

        self.vox_init_dict = {}
        self.vox_des_dict = {}

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load all scene configs
        self.scene_configs = []
        for scene_id in scene_ids:
            # get this scene metadata
            scene_metadata = self.metadata["scene_" + str(scene_id)]
            init_scene_config = self.load_scene_config(scene_metadata)
            self.scene_configs.append(scene_config)

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
            all_cfgs_this_scene = self.scene_configs[scene_id]
            env_ids = np.random.randint(
                low=0, high=len(all_cfgs_this_scene), size=num_envs_to_get
            )
            cfgs_this_scene = []
            for env_id in env_ids:
                cfgs_this_scene.append(all_cfgs_this_scene[env_id])
            all_cfgs.append(merge_concurrent_scene_configs(cfgs_this_scene))
        return all_cfgs

    def load_scene_config(self, scene_metadata: dict) -> ConcurrentSceneData:
        """Load a single scene configuration from disk."""
        scene_id = scene_metadata["scene_id"]

        # voxels
        vox_init_path = self.data_dir / "voxels" / f"vox_init_{scene_id}.npz"
        vox_des_path = self.data_dir / "voxels" / f"vox_des_{scene_id}.npz"
        # Load sparse tensors # note: small enough to fit into memory, yet. (36mb estimated)

        self.vox_init_dict[scene_id] = self._load_sparse_tensor(vox_init_path)
        self.vox_des_dict[scene_id] = self._load_sparse_tensor(vox_des_path)
        # ^ memory calculation: 100k samples*max 15 items * 10 datapoints * float16 =36mil = 36mbytes
        # graphs
        elec_graphs_init, mech_graphs_init, elec_graphs_des, mech_graphs_des = (
            self._load_graphs(scene_id)
        )

        # tool idx
        tool_idx_path_init = self.data_dir / "tool_idx_" + str(scene_id) + ".pt"
        tool_idx_path_des = self.data_dir / "tool_idx_" + str(scene_id) + ".pt"
        tool_data_init = torch.load(tool_idx_path_init)
        tool_data_des = torch.load(tool_idx_path_des)

        # load RepairsEnvState
        init_sim_state = reconstruct_sim_state(
            elec_graphs_init,
            mech_graphs_init,
            scene_metadata["electronics_indices"],
            scene_metadata["mechanical_indices"],
            tool_data_init,
        )
        des_sim_state = reconstruct_sim_state(
            elec_graphs_des,
            mech_graphs_des,
            scene_metadata["electronics_indices"],
            scene_metadata["mechanical_indices"],
            tool_data_des,
        )
        sim_state = ConcurrentSceneData(
            scene=None,
            gs_entities=None,
            cameras=None,
            current_state=init_sim_state,
            desired_state=des_sim_state,
            vox_init=self.vox_init_dict[scene_id],
            vox_des=self.vox_des_dict[scene_id],
            initial_diffs=None,
            initial_diff_counts=None,
            scene_id=scene_id,
            reward_history=None,
            batch_dim=scene_metadata["batch_dim"],
            step_count=torch.zeros(scene_metadata["batch_dim"], dtype=torch.int32),
        )

        return sim_state

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
            scene_id not in self.loaded_pyg_batch_dict_mech_init
            or scene_id not in self.loaded_pyg_batch_dict_elec_init
        ):
            # load initial and desired graphs: mech init, mech des, elec init, elec des
            # note: not in a for loop, but it's OK.
            mech_graph_init_path = (
                self.data_dir / "graphs" / f"mechanical_graphs_{scene_id}.pt"
            )
            mech_graph_des_path = (
                self.data_dir / "graphs" / f"mechanical_graphs_{scene_id}.pt"
            )
            elec_graph_init_path = (
                self.data_dir / "graphs" / f"electronic_graphs_{scene_id}.pt"
            )
            elec_graph_des_path = (
                self.data_dir / "graphs" / f"electronic_graphs_{scene_id}.pt"
            )
            assert mech_graph_init_path.exists(), (
                f"Mechanical graph file not found at {mech_graph_init_path}"
            )
            assert elec_graph_init_path.exists(), (
                f"Electronic graph file not found at {elec_graph_init_path}"
            )
            assert mech_graph_des_path.exists(), (
                f"Mechanical graph file not found at {mech_graph_des_path}"
            )
            assert elec_graph_des_path.exists(), (
                f"Electronic graph file not found at {elec_graph_des_path}"
            )

            loaded_graphs_mech_init = Batch.from_data_list(
                torch.load(mech_graph_init_path)
            )
            loaded_graphs_elec_init = Batch.from_data_list(
                torch.load(elec_graph_init_path)
            )
            loaded_graphs_mech_des = Batch.from_data_list(
                torch.load(mech_graph_des_path)
            )
            loaded_graphs_elec_des = Batch.from_data_list(
                torch.load(elec_graph_des_path)
            )

            assert (
                loaded_graphs_mech_init.num_graphs == loaded_graphs_elec_init.num_graphs
            ), "Num graphs in mechanical and electronic batches do not match."
            assert loaded_graphs_mech_init[0].scene_id == scene_id, (
                f"The loaded (mechanical) dataset graph does not seem to belong to this scene. "
                f"Expected {scene_id}, got {loaded_graphs_mech_init[0].scene_id}"
            )
            assert loaded_graphs_elec_init[0].scene_id == scene_id, (
                f"The loaded (electronic) dataset graph does not seem to belong to this scene. "
                f"Expected {scene_id}, got {loaded_graphs_elec_init[0].scene_id}"
            )

            self.loaded_pyg_batch_dict_mech_init[scene_id] = loaded_graphs_mech_init
            self.loaded_pyg_batch_dict_mech_des[scene_id] = loaded_graphs_mech_des
            self.loaded_pyg_batch_dict_elec_init[scene_id] = loaded_graphs_elec_init
            self.loaded_pyg_batch_dict_elec_des[scene_id] = loaded_graphs_elec_des

        if env_idx is not None:
            loaded_graphs_mech_init = self.loaded_pyg_batch_dict_mech_init[scene_id][
                env_idx
            ]
            loaded_graphs_elec_init = self.loaded_pyg_batch_dict_elec_init[scene_id][
                env_idx
            ]
            loaded_graphs_mech_des = self.loaded_pyg_batch_dict_mech_des[scene_id][
                env_idx
            ]
            loaded_graphs_elec_des = self.loaded_pyg_batch_dict_elec_des[scene_id][
                env_idx
            ]

        return (
            loaded_graphs_mech_init,
            loaded_graphs_elec_init,
            loaded_graphs_mech_des,
            loaded_graphs_elec_des,
        )
