from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import uuid
import numpy as np
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
        for i in range(len(self.electronics_state)):
            electronics_diff, electronics_diff_count = self.electronics_state[i].diff(
                other.electronics_state[i]
            )
            physical_diff, physical_diff_count = self.physical_state[i].diff(
                other.physical_state[i]
            )
            fluid_diff, fluid_diff_count = self.fluid_state[i].diff(
                other.fluid_state[i]
            )
            total_diff_left = (
                electronics_diff_count + physical_diff_count + fluid_diff_count
            )
        return {
            "physical_diff": physical_diff,
            "electronics_diff": electronics_diff,
            "fluid_diff": fluid_diff,
        }, total_diff_left

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
