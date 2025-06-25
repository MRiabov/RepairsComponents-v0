from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.fluid_state import FluidState
from repairs_components.logic.tools.tools_state import ToolState
from repairs_components.training_utils.sim_state import SimState
from torch_geometric.data import Data


@dataclass
class RepairsSimState(SimState):
    "A convenience sim state class to put diff logic out of a step function"

    scene_batch_dim: int
    """The batch dim of this scene. This is the number of scenes genesis sim batch.
    Primarily it for sanity checks."""

    # the main states.
    electronics_state: list[ElectronicsState] = field(default_factory=list)
    physical_state: list[PhysicalState] = field(default_factory=list)
    fluid_state: list[FluidState] = field(default_factory=list)
    tool_state: list[ToolState] = field(default_factory=list)

    # to prevent computation of certain objects if they are not present.
    has_electronics: bool = False
    has_fluid: bool = False

    def __init__(self, batch_dim: int):
        super().__init__()
        self.scene_batch_dim = batch_dim
        self.electronics_state = [ElectronicsState() for _ in range(batch_dim)]
        self.physical_state = [PhysicalState() for _ in range(batch_dim)]
        self.fluid_state = [FluidState() for _ in range(batch_dim)]
        self.tool_state = [ToolState() for _ in range(batch_dim)]

    def diff(self, other: "RepairsSimState"):  # batched diff.
        assert len(self.electronics_state) == len(other.electronics_state), (
            "Batch dim mismatch!"
        )
        electronics_diffs = []
        electronics_diff_counts = []
        physical_diffs = []
        physical_diff_counts = []
        fluid_diffs = []
        fluid_diff_counts = []
        total_diff_counts = []
        for i in range(len(self.electronics_state)):
            electronics_diff, electronics_diff_count = self.electronics_state[i].diff(
                other.electronics_state[i]
            )
            electronics_diffs.append(electronics_diff)
            electronics_diff_counts.append(electronics_diff_count)
            physical_diff, physical_diff_count = self.physical_state[i].diff(
                other.physical_state[i]
            )
            physical_diffs.append(physical_diff)
            physical_diff_counts.append(physical_diff_count)
            fluid_diff, fluid_diff_count = self.fluid_state[i].diff(
                other.fluid_state[i]
            )
            fluid_diffs.append(fluid_diff)
            fluid_diff_counts.append(fluid_diff_count)

        electronics_diff_counts = torch.tensor(electronics_diff_counts)
        physical_diff_counts = torch.tensor(physical_diff_counts)
        fluid_diff_counts = torch.tensor(fluid_diff_counts)
        total_diff_counts = (
            electronics_diff_counts + physical_diff_counts + fluid_diff_counts
        )

        return {
            "physical_diff": physical_diffs,
            "electronics_diff": electronics_diffs,
            "fluid_diff": fluid_diffs,
            # "physical_diff_count": physical_diff_counts,  # probably unnecessary.
            # "electronics_diff_count": electronics_diff_counts,
            # "fluid_diff_count": fluid_diff_counts,
        }, total_diff_counts

    def save(self, path: Path, scene_id, env_idx: torch.Tensor | None = None):
        """Save the state to a JSON file with a unique identifier.

        Returns:
            - Names of bodies located under indices
            - Names of electronics bodies located under indices.
        """
        # Create output directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        if env_idx is None:  # save all envs
            env_idx = torch.arange(self.scene_batch_dim)

        # # Generate a unique filename
        # uid = str(uuid.uuid4())
        # filename = f"step_state_{uid}.json"
        # filepath = path / filename

        #explicitly patch some fields as expected to be missing:
        self.physical_state[0].fastener = None

        # Create a dictionary with all states
        state_dict = asdict(self)

        # save graphs, everything else can be reconstructed from the build123d scene.

        for env_id in env_idx:
            mech_graph_path, elec_graph_path = get_graph_save_paths(
                path, scene_id, int(env_id)
            )
            torch.save(self.physical_state[env_id].graph, mech_graph_path)
            torch.save(self.electronics_state[env_id].graph, elec_graph_path)

        state_dict.pop("physical_state")
        state_dict.pop("electronics_state")

        electronics_indices = self.electronics_state[0].indices
        physical_indices = self.physical_state[0].body_indices
        return electronics_indices, physical_indices


def get_graph_save_paths(base_dir: Path, scene_id: int, env_id: int):
    mech_graph_path = (
        base_dir / "graphs" / f"mechanical_graphs_{int(scene_id)}_{int(env_id)}.pt"
    )
    elec_graph_path = (
        base_dir / "graphs" / f"electronic_graphs_{int(scene_id)}_{int(env_id)}.pt"
    )
    return mech_graph_path, elec_graph_path


def merge_global_states(state_list: list[RepairsSimState]):
    assert len(state_list) > 0, "State list can not be zero."
    repairs_sim_state = RepairsSimState(len(state_list))
    repairs_sim_state.electronics_state = [
        elec for state in state_list for elec in state.electronics_state
    ]
    repairs_sim_state.physical_state = [
        phys for state in state_list for phys in state.physical_state
    ]
    repairs_sim_state.fluid_state = [
        fluid for state in state_list for fluid in state.fluid_state
    ]
    repairs_sim_state.tool_state = [
        tool for state in state_list for tool in state.tool_state
    ]
    return repairs_sim_state


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
        old_state.physical_state[old_id] = new_state.physical_state[new_id]
        old_state.fluid_state[old_id] = new_state.fluid_state[new_id]
        old_state.tool_state[old_id] = new_state.tool_state[new_id]

    return old_state


def reconstruct_sim_state(
    electronics_graphs: list[Data],
    mechanical_graphs: list[Data],
    electronics_indices: list[dict[str, int]],
    mechanical_indices: list[dict[str, int]],
    tool_data: torch.Tensor,  # int tensor of tool ids
    fluid_data_placeholder: list[dict[str, int]] | None = None,
) -> RepairsSimState:
    """Load a single simulation state from graphs and indices (i.e. from the offline dataset)"""
    from repairs_components.training_utils.sim_state_global import RepairsSimState

    assert fluid_data_placeholder is None, NotImplementedError(
        "Fluid data reconstruction is not implemented."
    )
    assert len(electronics_graphs) == len(mechanical_graphs), (
        "Electronics and mechanical graphs must have the same length."
    )

    batch_dim = len(electronics_graphs)
    repairs_sim_state = RepairsSimState(batch_dim)
    repairs_sim_state.electronics_state = [
        ElectronicsState.rebuild_from_graph(graph, indices)
        for graph, indices in zip(electronics_graphs, electronics_indices)
    ]
    repairs_sim_state.physical_state = [
        PhysicalState.rebuild_from_graph(graph, indices)
        for graph, indices in zip(mechanical_graphs, mechanical_indices)
    ]
    repairs_sim_state.tool_state = [
        ToolState.rebuild_from_saved(indices) for indices in tool_data
    ]

    return repairs_sim_state
