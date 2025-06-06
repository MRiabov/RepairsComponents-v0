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


@dataclass
class RepairsSimState(SimState):
    "A convenience sim state class to put diff logic out of a step function"

    electronics_state: list[ElectronicsState] = field(default_factory=list)
    physical_state: list[PhysicalState] = field(default_factory=list)
    fluid_state: list[FluidState] = field(default_factory=list)
    tool_state: list[ToolState] = field(default_factory=list)

    def __init__(self, batch_dim: int):
        super().__init__()
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

        electronics_diff_counts = torch.stack(electronics_diff_counts)
        physical_diff_counts = torch.stack(physical_diff_counts)
        fluid_diff_counts = torch.stack(fluid_diff_counts)
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

    def save(self, output_dir: str = "./step_states") -> str:
        """Save the current state to a JSON file with a unique identifier.

        Args:
            electronics_state: Current electronics state
            physical_state: Current physical state
            fluid_state: Current fluid state
            output_dir: Directory to save the state file

        Returns:
            str: Path to the saved state file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        uid = str(uuid.uuid4())
        filename = f"step_state_{uid}.json"
        filepath = Path(output_dir) / filename

        # Create a dictionary with all states
        state_dict = asdict(self)

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(state_dict, f)

        print(f"Step state saved to: {output_dir}")

        return str(filepath)


def merge_global_states(state_list: list[RepairsSimState]):
    assert len(state_list) > 0, "State list can not be zero."
    repairs_sim_state = RepairsSimState(len(state_list))
    repairs_sim_state.electronics_state = [
        state.electronics_state for state in state_list
    ]
    repairs_sim_state.physical_state = [state.physical_state for state in state_list]
    repairs_sim_state.fluid_state = [state.fluid_state for state in state_list]
    repairs_sim_state.tool_state = [state.tool_state for state in state_list]
    return repairs_sim_state


def merge_global_states_at_idx(
    old_state: RepairsSimState, new_state: RepairsSimState, reset_states: torch.Tensor
):
    "Insert new states at indicated by bool tensor `reset_states`."
    assert len(old_state.electronics_state) == len(new_state.electronics_state), (
        "States lists must have the same length."
    )
    assert len(old_state.electronics_state) > 0, "States lists can not be empty."
    assert len(reset_states) == len(old_state.electronics_state), (
        "Reset states must have the same length as states."
    )
    assert reset_states.dtype == torch.bool, "Reset states must be a bool tensor."
    assert reset_states.ndim == 1, (
        "reset states has ndim of one because it's only for this scene."
    )

    repaired_sim_state = RepairsSimState(len(old_state.electronics_state))

    for i in range(len(old_state.electronics_state)):
        if reset_states[i]:
            repaired_sim_state.electronics_state[i] = new_state.electronics_state[i]
            repaired_sim_state.physical_state[i] = new_state.physical_state[i]
            repaired_sim_state.fluid_state[i] = new_state.fluid_state[i]
            repaired_sim_state.tool_state[i] = new_state.tool_state[i]
        else:
            repaired_sim_state.electronics_state[i] = old_state.electronics_state[i]
            repaired_sim_state.physical_state[i] = old_state.physical_state[i]
            repaired_sim_state.fluid_state[i] = old_state.fluid_state[i]
            repaired_sim_state.tool_state[i] = old_state.tool_state[i]
    return repaired_sim_state
