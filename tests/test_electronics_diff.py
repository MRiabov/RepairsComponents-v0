import pytest
import torch

from torch_geometric.data import Data

from repairs_components.logic.electronics.electronics_state import (
    ElectronicsState,
    register_components_batch,
    connect_terminals_batch,
)
from repairs_components.logic.electronics.component import ElectricalComponentsEnum as ECE


class TestElectronicsDiff:
    def make_two_states_same_components(self):
        names = ["batt", "res"]
        component_type = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        voltage_max_A = torch.tensor([9.0, 0.0], dtype=torch.float32)
        voltage_max_B = torch.tensor([9.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)

        A = ElectronicsState()
        B = ElectronicsState()
        A = register_components_batch(A, names, component_type, voltage_max_A, current_max, terminals_per_component_batch)
        B = register_components_batch(B, names, component_type.clone(), voltage_max_B, current_max.clone(), terminals_per_component_batch.clone())
        return A, B

    def test_diff_edges_added(self):
        A, B = self.make_two_states_same_components()
        # A: no edges; B: batt(0).t0(0) connected to res(1).t0(2)
        B = connect_terminals_batch(B, torch.tensor([[0, 2]], dtype=torch.long))

        diff_graph, n = A.diff(B)
        assert isinstance(diff_graph, Data)
        assert isinstance(n, int)

        # One changed edge in symmetric difference
        assert diff_graph.edge_index.shape[1] == 1
        assert diff_graph.edge_mask.shape[0] == 1
        assert int(diff_graph.edge_mask.sum().item()) == 1
        assert int(diff_graph.num_nodes) == 2
        # No node feature changes
        assert int(diff_graph.node_mask.sum().item()) == 0

        # Check dict representation
        d = A.diff_to_dict(diff_graph)
        assert d["num_nodes"] == 2
        assert d["num_edges"] == 1
        assert len(d["edges_changed"]) == 1
        # Names should resolve via inverse_component_indices
        pair = d["edges_changed"][0]
        assert set(pair) == {"batt", "res"}

    def test_diff_edges_removed(self):
        A, B = self.make_two_states_same_components()
        # A has the connection; B has no connections
        A = connect_terminals_batch(A, torch.tensor([[0, 2]], dtype=torch.long))

        diff_graph, n = A.diff(B)
        assert diff_graph.edge_index.shape[1] == 1
        assert int(diff_graph.edge_mask.sum().item()) == 1
        assert int(diff_graph.node_mask.sum().item()) == 0
        assert n == 1

    def test_diff_nodes_changed(self):
        # Change a scalar node feature (max_voltage) on one component
        names = ["batt", "res"]
        component_type = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        voltage_max_A = torch.tensor([9.0, 0.0], dtype=torch.float32)
        voltage_max_B = torch.tensor([12.0, 0.0], dtype=torch.float32)  # changed batt voltage
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)

        A = ElectronicsState()
        B = ElectronicsState()
        A = register_components_batch(A, names, component_type, voltage_max_A, current_max, terminals_per_component_batch)
        B = register_components_batch(B, names, component_type.clone(), voltage_max_B, current_max.clone(), terminals_per_component_batch.clone())

        diff_graph, n = A.diff(B)
        assert isinstance(n, int)
        # One node changed (index 0)
        assert int(diff_graph.node_mask.sum().item()) == 1
        assert diff_graph.edge_index.shape[1] == 0
        assert int(diff_graph.edge_mask.sum().item()) == 0

        d = A.diff_to_dict(diff_graph)
        assert len(d["nodes_changed"]) == 1
        assert d["nodes_changed"][0]["name"] == "batt"

    def test_diff_to_str(self):
        A, B = self.make_two_states_same_components()
        B = connect_terminals_batch(B, torch.tensor([[0, 2]], dtype=torch.long))
        diff_graph, _ = A.diff(B)
        s = A.diff_to_str(diff_graph)
        assert isinstance(s, str)
        assert "Edges changed:" in s

    def test_diff_with_batched_states_via_slice(self):
        # Ensure we can stack states (B>1) and still diff by slicing a single env
        A, B = self.make_two_states_same_components()
        B = connect_terminals_batch(B, torch.tensor([[0, 2]], dtype=torch.long))

        Ab: ElectronicsState = torch.stack([A, A])  # type: ignore
        Bb: ElectronicsState = torch.stack([B, B])  # type: ignore

        # Slice to single env and compute diff as usual
        diff_graph, n = Ab[0].diff(Bb[0])  # type: ignore[index]
        assert isinstance(n, int)
        assert diff_graph.edge_index.shape[1] == 1
        assert int(diff_graph.edge_mask.sum().item()) == 1
