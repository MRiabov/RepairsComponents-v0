import pytest
import torch

from repairs_components.logic.electronics.electronics_state import (
    ElectronicsState,
    register_components_batch,
    connect_terminals_batch,
)
from repairs_components.logic.electronics.component import ElectricalComponentsEnum as ECE


class TestElectronicsStateTCConnections:
    def test_register_components_single_env_basic(self):
        state = ElectronicsState()

        names = ["batt", "res", "led"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR), int(ECE.LED)], dtype=torch.long)
        vmax = torch.tensor([9.0, 0.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([1.0, 0.1, 0.05], dtype=torch.float32)
        tpc = torch.tensor([2, 2, 2], dtype=torch.long)

        state = register_components_batch(state, names, ctype, vmax, imax, tpc)

        # Name maps should be dicts
        assert isinstance(state.component_indices_from_name, dict)
        assert isinstance(state.inverse_component_indices, dict)
        assert state.component_indices_from_name["batt"] == 0
        assert state.inverse_component_indices[2] == "led"

        # terminal_to_component mapping shape and content
        assert state.terminal_to_component.ndim == 1
        assert state.terminal_to_component.shape[0] == int(tpc.sum().item())
        # First two terminals belong to component 0, next two to 1, etc.
        assert torch.equal(
            state.terminal_to_component,
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        )

        # net_id should be 1D with all -1
        assert state.net_id.ndim == 1
        assert torch.all(state.net_id == -1)

    def test_register_components_batched_sets_net_shape(self):
        # Register on a single state, then stack to create a batched state (B=2)
        single = ElectronicsState()
        names = ["batt", "res"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        vmax = torch.tensor([9.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([1.0, 0.1], dtype=torch.float32)
        tpc = torch.tensor([2, 2], dtype=torch.long)

        single = register_components_batch(single, names, ctype, vmax, imax, tpc)
        batched: ElectronicsState = torch.stack([single, single])  # type: ignore

        # Name maps may be kept as dict or become list[dict] after stacking; accept both
        if isinstance(batched.component_indices_from_name, dict):
            assert batched.component_indices_from_name["res"] == 1
        else:
            assert isinstance(batched.component_indices_from_name, list)
            assert isinstance(batched.component_indices_from_name[0], dict)
            assert batched.component_indices_from_name[0]["res"] == 1

        # net_id should be [B, T]
        T = int(tpc.sum().item())
        assert batched.net_id.shape == (2, T)
        assert torch.all(batched.net_id == -1)

    def test_connect_terminals_single_env_two_nets(self):
        state = ElectronicsState()
        names = ["a", "b"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        vmax = torch.tensor([5.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([1.0, 0.1], dtype=torch.float32)
        tpc = torch.tensor([2, 2], dtype=torch.long)  # terminals: [0,1] -> a, [2,3] -> b
        state = register_components_batch(state, names, ctype, vmax, imax, tpc)

        # Connect a.t0 (0) with b.t0 (2) and a.t1 (1) with b.t1 (3)
        pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        state = connect_terminals_batch(state, pairs)

        # Expect two nets formed with minimal gid chosen per pair (0 and 1)
        # Order of gids is deterministic due to min(candidates)
        assert state.net_id.ndim == 1
        assert state.net_id.shape[0] == 4
        # First pair -> gid 0 on terminals 0 and 2
        # Second pair -> gid 1 on terminals 1 and 3
        assert state.net_id[0].item() == state.net_id[2].item() == 0
        assert state.net_id[1].item() == state.net_id[3].item() == 1

        # export_graph should yield a single component-level edge (a-b)
        g = state.export_graph()
        assert g.edge_index.shape[1] == 1
        u, v = g.edge_index[:, 0].tolist()
        assert {u, v} == {0, 1}

    def test_connect_terminals_batched_broadcast_pairs(self):
        # Prepare B=2 by stacking after single registration
        single = ElectronicsState()
        names = ["a", "b"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        vmax = torch.tensor([5.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([1.0, 0.1], dtype=torch.float32)
        tpc = torch.tensor([2, 2], dtype=torch.long)
        single = register_components_batch(single, names, ctype, vmax, imax, tpc)
        batched: ElectronicsState = torch.stack([single, single])  # type: ignore

        # Provide pairs as [K,2]; should broadcast to both batch elements
        pairs = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        batched = connect_terminals_batch(batched, pairs)

        assert batched.net_id.shape == (2, 4)
        for b in range(2):
            nb = batched.net_id[b]
            assert nb[0].item() == nb[2].item() == 0
            assert nb[1].item() == nb[3].item() == 1

    def test_connect_terminals_merges_existing_nets(self):
        state = ElectronicsState()
        names = ["x", "y"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        vmax = torch.tensor([3.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([0.5, 0.1], dtype=torch.float32)
        tpc = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(state, names, ctype, vmax, imax, tpc)

        # Form two separate nets, then merge them
        state = connect_terminals_batch(state, torch.tensor([[0, 2]], dtype=torch.long))
        state = connect_terminals_batch(state, torch.tensor([[1, 3]], dtype=torch.long))
        # Now connect terminals across the two nets to merge them
        state = connect_terminals_batch(state, torch.tensor([[1, 2]], dtype=torch.long))

        # All terminals should now share the minimal gid (0)
        assert torch.all(state.net_id == state.net_id[0])
        assert int(state.net_id[0].item()) == 0

    def test_clear_all_connections(self):
        state = ElectronicsState()
        names = ["a", "b"]
        ctype = torch.tensor([int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long)
        vmax = torch.tensor([5.0, 0.0], dtype=torch.float32)
        imax = torch.tensor([1.0, 0.1], dtype=torch.float32)
        tpc = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(state, names, ctype, vmax, imax, tpc)

        state = connect_terminals_batch(state, torch.tensor([[0, 2]], dtype=torch.long))
        assert torch.any(state.net_id >= 0)

        state = state.clear_all_connections()
        assert torch.all(state.net_id == -1)
