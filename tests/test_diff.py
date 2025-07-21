import pytest
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

# Import the original implementations
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.electronics.components import (
    Component,
    Battery,
    Resistor,
    Wire,
)

# note: AI-generated. Was not actually run.


def test_physical_state_diff_basic():
    """Test basic diff functionality for PhysicalState."""
    # Create two physical states
    state1 = PhysicalState()
    state2 = PhysicalState()

    # Register some bodies in both states
    state1.register_body(
        "body1", position=torch.tensor([0, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state1.register_body(
        "body2", position=torch.tensor([1, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state1.connect_fastener_to_one_body("conn1", "body1", "body2")

    state2.register_body(
        "body1", position=torch.tensor([0.1, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )  # Slightly moved
    state2.register_body(
        "body2", position=torch.tensor([1, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state2.register_body(
        "body3", position=torch.tensor([0, 1, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )  # New body
    state2.connect_fastener_to_one_body("conn1", "body1", "body2")
    state2.connect_fastener_to_one_body("conn2", "body2", "body3")  # New connection

    # Get the diff
    diff_graph = state1.diff(state2)

    # Verify node differences
    assert diff_graph.x.size(0) == 3  # 3 nodes total
    assert (
        diff_graph.node_mask.sum() == 3
    )  # All nodes have changes (1 moved, 1 unchanged, 1 new)

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


def test_electronics_state_diff_basic():
    """Test basic diff functionality for ElectronicsState."""
    # Create two electronics states
    state1 = ElectronicsState()
    state2 = ElectronicsState()

    # Create some components
    batt1 = Battery("batt1", 9.0)
    res1 = Resistor("res1", 100.0)
    wire1 = Wire("wire1")

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
    """Test PhysicalState diff with identical states."""
    state1 = PhysicalState()
    state1.register_body(
        "body1", position=torch.tensor([0, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state1.register_body(
        "body2", position=torch.tensor([1, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state1.connect_fastener_to_one_body("conn1", "body1", "body2")

    state2 = PhysicalState()
    state2.register_body(
        "body1", position=torch.tensor([0, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state2.register_body(
        "body2", position=torch.tensor([1, 0, 0]), rotation=torch.tensor([0, 0, 0, 1])
    )
    state2.connect_fastener_to_one_body("conn1", "body1", "body2")

    diff_graph = state1.diff(state2)

    # No nodes or edges should be marked as changed
    assert diff_graph.node_mask.sum() == 0
    assert diff_graph.edge_mask.sum() == 0


def test_electronics_state_edge_changes():
    """Test ElectronicsState diff with edge changes."""
    state1 = ElectronicsState()
    batt1 = Battery("batt1", 9.0)
    res1 = Resistor("res1", 100.0)
    res2 = Resistor("res2", 200.0)

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
