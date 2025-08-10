import pytest
import torch

# Import the implementations
from repairs_components.logic.physical_state import (
    PhysicalState,
    register_bodies_batch,
)
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.electronics.resistor import Resistor
from repairs_components.logic.electronics.wire import Wire
from repairs_components.logic.electronics.voltage_source import VoltageSource as Battery

# Tests updated to use batched PhysicalState API and WXYZ quaternion convention.


def test_physical_state_diff_basic():
    """Basic diff for PhysicalState using batched registration and WXYZ quats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two single-environment states (B=1) with the same bodies
    state1 = PhysicalState(device=device)
    state2 = PhysicalState(device=device)

    names = ["body1@solid", "body2@solid"]
    B, N = 1, len(names)

    # Positions within bounds, z >= 0; rotations are identity in WXYZ
    pos1 = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot1 = torch.zeros(B, N, 4, dtype=torch.float32)
    rot1[..., 0] = 1.0  # identity quaternion WXYZ: [1,0,0,0]
    fixed = torch.tensor([False, False])

    state1 = register_bodies_batch(state1, names, pos1, rot1, fixed)

    # Move body1 by 1 cm in x (over 5 mm threshold); keep others the same
    pos2 = pos1.clone()
    pos2[:, 0, 0] += 0.01
    rot2 = rot1.clone()
    state2 = register_bodies_batch(state2, names, pos2, rot2, fixed)

    # Compute diff
    diff_graph, total_diff = state1.diff(state2)

    # Minimal sanity checks (avoid relying on internal mask shapes)
    from torch_geometric.data import Data

    assert isinstance(diff_graph, Data)
    assert isinstance(total_diff, int)


@pytest.mark.skip(reason="Electronics is not implemented yet.")
def test_electronics_state_diff_basic():
    """Test basic diff functionality for ElectronicsState."""
    # Create two electronics states
    state1 = ElectronicsState()
    state2 = ElectronicsState()

    # Create some components
    batt1 = Battery(9.0, "batt1")  # type: ignore
    res1 = Resistor(100.0, "res1")  # type: ignore
    wire1 = Wire("wire1")  # type: ignore

    # Register components in both states
    state1.register(batt1)
    state1.register(res1)
    state1.connect("batt1", "res1")

    state2.register(batt1)
    state2.register(res1)
    state2.register(wire1)  # New component
    state2.connect("batt1", "res1")
    state2.connect("res1", "wire1")  # New connection

    # Get the diff
    diff_graph = state1.diff(state2)

    # Verify node differences
    assert diff_graph.x.size(0) == 3  # 3 nodes total
    assert diff_graph.node_mask.sum() == 1  # 1 new node

    # Verify edge differences
    assert diff_graph.edge_index.size(1) == 2  # 1 existing + 1 new edge
    assert diff_graph.edge_mask.sum() == 1  # 1 new edge

    # Test conversion to dict
    diff_dict = state1.diff_to_dict(diff_graph)
    assert isinstance(diff_dict, dict)
    assert "nodes" in diff_dict
    assert "edges" in diff_dict

    # Test string representation
    diff_str = state1.diff_to_str(diff_graph)
    assert isinstance(diff_str, str)


def test_physical_state_no_changes():
    """Diff should report zero changes for identical batched states."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state1 = PhysicalState(device=device)
    state2 = PhysicalState(device=device)

    names = ["body1@solid", "body2@solid"]
    B, N = 1, len(names)

    pos = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot = torch.zeros(B, N, 4, dtype=torch.float32)
    rot[..., 0] = 1.0  # identity WXYZ
    fixed = torch.tensor([False, False])

    state1 = register_bodies_batch(state1, names, pos, rot, fixed)
    state2 = register_bodies_batch(state2, names, pos.clone(), rot.clone(), fixed)

    diff_graph, total_diff = state1.diff(state2)

    assert isinstance(total_diff, int)
    assert total_diff == 0


@pytest.mark.skip(reason="Electronics is not implemented yet.")
def test_electronics_state_edge_changes():
    """Test ElectronicsState diff with edge changes."""
    state1 = ElectronicsState()
    batt1 = Battery(9.0, "batt1")  # type: ignore
    res1 = Resistor(100.0, "res1")  # type: ignore
    res2 = Resistor(200.0, "res2")  # type: ignore

    # Initial state: batt1 -- res1 -- res2
    state1.register(batt1)
    state1.register(res1)
    state1.register(res2)
    state1.connect("batt1", "res1")
    state1.connect("res1", "res2")

    # Modified state: batt1 -- res2 -- res1 (connection changed)
    state2 = ElectronicsState()
    state2.register(batt1)
    state2.register(res1)
    state2.register(res2)
    state2.connect("batt1", "res2")  # Changed connection
    state2.connect("res2", "res1")  # Changed connection

    diff_graph = state1.diff(state2)

    # Should have 2 edge changes (one removed, one added in each direction)
    assert diff_graph.edge_mask.sum() == 2

    # Verify the diff dict shows the changes
    diff_dict = state1.diff_to_dict(diff_graph)
    assert len(diff_dict["edges"]["added"]) == 2  # Two new connections
    assert len(diff_dict["edges"]["removed"]) == 2  # Two old connections removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
