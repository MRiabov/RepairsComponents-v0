from dataclasses import dataclass, field
from pathlib import Path
import torch
from repairs_components.logic.electronics.electronics_state import (
    ElectronicsState,
    ElectronicsInfo,
)
from repairs_components.logic.physical_state import PhysicalState, PhysicalStateInfo
from repairs_components.logic.tools.tools_state import ToolState, ToolInfo
from repairs_components.training_utils.sim_state import SimState
from torch_geometric.data import Data


@dataclass
class RepairsSimInfo:
    "Singleton, meta information about the sim state."

    component_info: ElectronicsInfo = field(default_factory=ElectronicsInfo)
    physical_info: PhysicalStateInfo = field(default_factory=PhysicalStateInfo)
    tool_info: ToolInfo = field(default_factory=ToolInfo)

    fastener_link_id_cache: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    "Cache of fastener links for fastener constraint creation without dict querying."
    env_setup_name: str = (
        "env_setup"  # TODO: move to a separate persistence state info.
    )
    # to prevent computation of certain objects if they are not present.

    # Mechanical linkage metadata (rigid groups). These are optional and populated
    # by translation when linked groups are provided. Names must correspond to
    # `physical_info.body_indices` keys; indices are resolved from those names.
    mech_linked_groups_names: tuple[list[str], ...] = field(default_factory=tuple)
    mech_linked_groups_indices: tuple[list[int], ...] = field(default_factory=tuple)


class RepairsSimState(SimState):  # type: ignore
    """A SimState class holding all information about the scene.
    Is a TensorClass.

    RepairsSimState will always have to be instantiated with either RepairsSimState(device=device).unsqueeze(0)
    or torch.stack([RepairsSimState(device=device)]*B). This is because they are expected to be batched with a leading dimension."""

    # the main states.
    electronics_state: ElectronicsState = field(default_factory=ElectronicsState)
    physical_state: PhysicalState = field(
        default_factory=PhysicalState
    )  # Single TensorClass instance
    # fluid_state: list[FluidState] = field(default_factory=list)
    tool_state: ToolState = field(default_factory=ToolState)

    def diff(self, other: "RepairsSimState", sim_info: RepairsSimInfo):  # batched diff.
        assert len(self.electronics_state) == len(other.electronics_state), (
            "Batch dim mismatch in sim state diff!"
        )
        assert self.batch_size == other.batch_size, (
            "Batch dim mismatch in sim state diff!"
        )
        assert sim_info.component_info == sim_info.component_info, (
            "Electronics component info mismatch in sim state diff."
        )
        assert sim_info.component_info is not None, (
            "Electronics component info is not set on RepairsSimState."
        )
        assert (
            self.electronics_state.net_ids.shape
            == other.electronics_state.net_ids.shape
        ), "Electronics net ids shape mismatch in sim state diff."
        electronics_diffs = []
        electronics_diff_counts = []
        physical_diffs = []
        physical_diff_counts = []
        total_diff_counts = []
        # Normalize batch size to an int (TensorClass may expose it as int or tuple)
        bs: int = (
            int(self.batch_size)
            if isinstance(self.batch_size, int)
            else int(self.batch_size[0])
        )
        for i in range(bs):
            # Electronics diff: only compute if both states have electronics registered
            if sim_info.component_info.has_electronics:
                # Pass single component_info; comparability is validated inside diff
                electronics_diff, electronics_diff_count = self.electronics_state[
                    i
                ].diff(other.electronics_state[i], sim_info.component_info)
            else:
                # valid case when electronics is really not registered in either
                electronics_diff = Data(
                    x=torch.empty((0, 4), dtype=torch.bfloat16),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    node_mask=torch.empty((0,), dtype=torch.bool),
                    edge_mask=torch.empty((0,), dtype=torch.bool),
                    num_nodes=0,
                )
                electronics_diff_count = 0
            electronics_diffs.append(electronics_diff)
            electronics_diff_counts.append(electronics_diff_count)
            # For TensorClass, we need to slice to get individual batch elements
            self_physical_state_i = self.physical_state[
                i : i + 1
            ]  # Get slice for batch element i
            other_physical_state_i = other.physical_state[i : i + 1]
            physical_diff, physical_diff_count = self_physical_state_i.diff(
                other_physical_state_i, sim_info.physical_info
            )
            physical_diffs.append(physical_diff)
            physical_diff_counts.append(physical_diff_count)

        electronics_diff_counts = torch.tensor(electronics_diff_counts)
        physical_diff_counts = torch.tensor(physical_diff_counts)
        total_diff_counts = electronics_diff_counts + physical_diff_counts

        return {
            "physical_diff": physical_diffs,
            "electronics_diff": electronics_diffs,
            # "fluid_diff": fluid_diffs,
            # "physical_diff_count": physical_diff_counts,  # probably unnecessary.
            # "electronics_diff_count": electronics_diff_counts,
            # "fluid_diff_count": fluid_diff_counts,
        }, total_diff_counts

    def save(
        self,
        path: Path,
        scene_id,
        init: bool,
        env_idx: torch.Tensor | None = None,
    ):
        """Save the full RepairsSimState TensorClass for simple persistence.

        Args:
            path: Base directory to save under.
            scene_id: ID of the scene.
            init: Whether saving initial or desired state variant.
            env_idx: Index of the environment to save. If None, saves all environments.
        """
        assert self.ndim == 1, "Expected that batch dim is int or size of 1."
        # Normalize batch size to an int for comparisons and indexing
        bs: int = (
            int(self.batch_size)
            if isinstance(self.batch_size, int)
            else int(self.batch_size[0])
        )
        assert bs >= 1, "Expected that batch dim is at least 1."
        # Ensure target directory exists
        state_path, _info_path = get_state_and_info_save_paths(
            path, scene_id, init=init
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)

        if env_idx is None:  # save all envs
            env_idx = torch.arange(bs)

        torch.save(self[env_idx], state_path)
        # sim_info is saved separately once per scene.


def get_state_and_info_save_paths(base_dir: Path, scene_id: int, init: bool):
    # note: env_id is not used because graphs are batched per scene.
    postfix = "init" if init else "des"
    # mech_graph_path = (
    #     base_dir / "graphs" / f"mechanical_graphs_{int(scene_id)}_{postfix}.pt"
    # )
    # elec_graph_path = (
    #     base_dir / "graphs" / f"electronics_graphs_{int(scene_id)}_{postfix}.pt"
    # )
    state_save_path = base_dir / "state" / f"state_{int(scene_id)}_{postfix}.pt"
    # info is invariant across init/des; save once without postfix
    info_save_path = base_dir / "state" / f"info_{int(scene_id)}.pt"

    return state_save_path, info_save_path


# deprecated: do directly.
def merge_global_states(state_list: list[RepairsSimState]):
    assert len(state_list) > 0, "State list can not be zero."
    # FIXME: this should be simple torch.cat now.
    assert all(state.ndim == 1 for state in state_list)
    return torch.cat(state_list)


def merge_global_states_at_idx(  # note: this is not idx anymore, this is mask.
    old_state: RepairsSimState, new_state: RepairsSimState, reset_mask: torch.Tensor
):  # not fixing that this is a bool tensor only because maybe, gods of JIT will reward me later.
    "Insert new states at indicated by bool tensor `reset_mask`."
    # assert len(old_state.electronics_state) == len(new_state.electronics_state), (
    #     "States lists must have the same length."
    # )
    assert len(old_state.electronics_state) > 0, "States lists can not be empty."
    assert len(reset_mask) == len(old_state.electronics_state), (
        "Reset mask must have the same length as states."
    )
    assert reset_mask.dtype == torch.bool, "Reset mask must be a bool tensor."
    assert reset_mask.ndim == 1, (
        "reset mask has ndim of one because it's only for this scene."
    )
    assert len(new_state.electronics_state) == reset_mask.int().sum(), (
        "Count of reset states must be equal to the batch dimension of the incoming states."
    )
    old_idx = reset_mask.nonzero().squeeze(1)

    for new_id, old_id in enumerate(old_idx):
        old_state.electronics_state[old_id] = new_state.electronics_state[new_id]
        # For TensorClass, update the specific batch indices
        old_state.tool_state[old_id] = new_state.tool_state[new_id]
    # batch update
    old_state.physical_state[old_idx] = new_state.physical_state

    return old_state


def reconstruct_sim_state(
    electronics_graphs: list[Data],
    mechanical_state: PhysicalState,  # Now PhysicalState objects directly
    electronics_indices: dict[str, int],
    mechanical_indices: dict[str, int],
    tool_data: torch.Tensor,  # int tensor of tool ids
    starting_hole_positions: torch.Tensor,
    starting_hole_quats: torch.Tensor,
    part_hole_batch: torch.Tensor,
    fluid_data_placeholder: list[dict[str, int]] | None = None,
) -> RepairsSimState:
    """Rebuild a batched RepairsSimState from saved PhysicalState and electronics graphs.

    Note: ElectronicsState reconstruction from graphs is not yet implemented; this function
    restores the PhysicalState and ToolState and updates hole locations.
    """
    from repairs_components.training_utils.sim_state_global import RepairsSimState
    from repairs_components.processing.translation import update_hole_locs

    assert fluid_data_placeholder is None, NotImplementedError(
        "Fluid data reconstruction is not implemented."
    )
    assert len(electronics_graphs) == len(mechanical_state), (
        "Electronics graphs and mechanical states must have the same length."
    )

    B = len(electronics_graphs)
    device = mechanical_state.device
    repairs_sim_state: RepairsSimState = torch.stack(
        [RepairsSimState(device=device)] * B
    )

    # Physical state is already a batched TensorClass
    repairs_sim_state.physical_state = mechanical_state

    # Restore tool ids into ToolState
    repairs_sim_state.tool_state = ToolState.rebuild_from_saved(tool_data)

    # Update hole locations based on saved metadata
    repairs_sim_state = update_hole_locs(
        repairs_sim_state, starting_hole_positions, starting_hole_quats, part_hole_batch
    )

    return repairs_sim_state
