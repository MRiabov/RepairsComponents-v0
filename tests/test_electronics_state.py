import torch

from repairs_components.logic.electronics.electronics_state import (
    ElectronicsState,
    register_components_batch,
    register_connectors_batch,
    connect_connector_to_one_terminal,
)
from repairs_components.logic.electronics.component import (
    ElectricalComponentsEnum as ECE,
)


class TestElectronicsStateConnections:
    def test_register_components_single_env_basic(self):
        # Create a batched empty state (B=2)
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        state = state.__post_init__()

        names = ["batt", "res", "led"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR), int(ECE.LED)], dtype=torch.long
        )
        max_voltage = torch.tensor([9.0, 0.0, 0.0], dtype=torch.float32)
        max_current = torch.tensor([1.0, 0.1, 0.05], dtype=torch.float32)
        terminals_per_component = torch.tensor([2, 2, 2], dtype=torch.long)

        state = register_components_batch(
            state,
            names,
            component_type,
            max_voltage,
            max_current,
            terminals_per_component,
        )

        # Name maps should be dicts
        assert isinstance(state.component_indices_from_name, dict)
        assert isinstance(state.inverse_component_indices, dict)
        assert state.component_indices_from_name["batt"] == 0
        assert all(n in state.component_indices_from_name for n in names)
        assert all(
            i == state.component_indices_from_name[n] for i, n in enumerate(names)
        )
        # terminal_to_component mapping shape and content
        assert state.terminal_to_component_batch.shape == (
            2,
            int(terminals_per_component.sum().item()),
        )

        # First two terminals belong to component 0, next two to 1, etc.
        assert torch.equal(
            state.terminal_to_component_batch[0],
            torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        )

        # net_id should be [B, T] with all -1
        assert state.net_ids.ndim == 2
        assert state.net_ids.shape[0] == 2
        assert state.net_ids.shape[1] == int(state.terminals_per_component.sum().item())
        assert torch.all(state.net_ids == -1)

    def test_register_components_batched_sets_net_shape(self):
        # Register directly on a batched state (B=2)
        # Removing unused variable to satisfy linter
        batched_state: ElectronicsState = torch.stack(
            [ElectronicsState(), ElectronicsState()]
        )  # type: ignore
        batched_state = batched_state.__post_init__()
        names = ["batt", "res"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([9.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)

        batched_state = register_components_batch(
            batched_state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        assert all(name in batched_state.component_indices_from_name for name in names)
        assert all(
            i == batched_state.component_indices_from_name[n]
            for i, n in enumerate(names)
        )

        # net_id should be [B, T]
        T = int(terminals_per_component_batch.sum().item())
        assert batched_state.net_ids.shape == (2, T)
        assert torch.all(batched_state.net_ids == -1)

    def test_connectors_connect_two_nets_batched(self):
        # Build batched state (B=2)
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        names = ["a", "b"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([5.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor(
            [2, 2], dtype=torch.long
        )  # terminals: [0,1] -> a, [2,3] -> b
        state = register_components_batch(
            state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        # Register two connectors (each has two terminals)
        state = register_connectors_batch(state, ["conn0", "conn1"])

        # Terminals: components -> 0,1 (a); 2,3 (b); connectors -> 4,5 (conn0), 6,7 (conn1)
        # Connect conn0: 4->0, 5->2 across both batch elements
        B = state.batch_size[0]
        pair = torch.tensor([[b, 4, 0] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, pair)
        pair = torch.tensor([[b, 5, 2] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, pair)
        # Connect conn1: 6->1, 7->3
        pair = torch.tensor([[b, 6, 1] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, pair)
        pair = torch.tensor([[b, 7, 3] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, pair)

        # Expect two nets formed with minimal gid chosen per pair (0 and 1)
        # Order of gids is deterministic due to min(candidates)
        assert state.net_ids.ndim == 2
        assert state.net_ids.shape[1] == 8
        # First pair -> gid 0 on terminals 0 and 2
        # Second pair -> gid 1 on terminals 1 and 3
        # For each batch env, check first component terminals form separate nets
        for b in range(state.batch_size[0]):
            nb = state.net_ids[b]
            assert nb[0].item() == nb[2].item()
            assert nb[1].item() == nb[3].item()

        # export_graph should yield a single component-level edge (a-b)
        g = state[0].export_graph()  # slice to single env for export
        assert g.edge_index.shape[1] == 1
        u, v = g.edge_index[:, 0].tolist()
        assert {u, v} == {0, 1}

    def test_connectors_batched_broadcast_pairs(self):
        # Prepare B=2 batched state and broadcast same connector pairs
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        names = ["a", "b"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([5.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(
            state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        state = register_connectors_batch(state, ["conn0", "conn1"])
        B = state.batch_size[0]
        # Connect same pairs across the batch
        for pair in [(4, 0), (5, 2), (6, 1), (7, 3)]:
            p = torch.tensor(
                [[b, pair[0], pair[1]] for b in range(B)], dtype=torch.long
            )
            state = connect_connector_to_one_terminal(state, p)

        assert state.net_ids.shape == (2, 8)
        for b in range(2):
            nb = state.net_ids[b]
            assert nb[0].item() == nb[2].item()
            assert nb[1].item() == nb[3].item()

    def test_connectors_merge_existing_nets(self):
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        names = ["x", "y"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([3.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([0.5, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(
            state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        state = register_connectors_batch(state, ["conn0", "conn1", "conn2"])
        B = state.batch_size[0]
        # Form two separate nets using two connectors
        for pair in [(4, 0), (5, 2), (6, 1), (7, 3)]:
            p = torch.tensor(
                [[b, pair[0], pair[1]] for b in range(B)], dtype=torch.long
            )
            state = connect_connector_to_one_terminal(state, p)
        # Now use third connector to bridge across the two nets: connect terminal 8->0 and 9->1
        for pair in [(8, 0), (9, 1)]:
            p = torch.tensor(
                [[b, pair[0], pair[1]] for b in range(B)], dtype=torch.long
            )
            state = connect_connector_to_one_terminal(state, p)

        # All terminals should now share the minimal gid (0)
        for b in range(state.batch_size[0]):
            nb = state.net_ids[b]
            # All original component terminals (0..3) should share the same gid
            assert nb[0].item() == nb[1].item() == nb[2].item() == nb[3].item()

    def test_clear_all_connections(self):
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        names = ["a", "b"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([5.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(
            state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        state = register_connectors_batch(state, ["conn0"])
        B = state.batch_size[0]
        p = torch.tensor([[b, 4, 0] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, p)
        p = torch.tensor([[b, 5, 2] for b in range(B)], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, p)
        assert torch.any(state.net_ids >= 0)

        state = state.clear_all_connections()
        assert torch.all(state.net_ids == -1)

    def test_connectors_batched_per_batch_pairs(self):
        # Prepare B=2 by registering on a batched state
        state: ElectronicsState = torch.stack([ElectronicsState(), ElectronicsState()])  # type: ignore
        names = ["a", "b"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max = torch.tensor([5.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)
        state = register_components_batch(
            state,
            names,
            component_type,
            voltage_max,
            current_max,
            terminals_per_component_batch,
        )
        state = register_connectors_batch(state, ["conn0"])
        # Different pairs per batch element: env0 -> connect 4->0, env1 -> connect 4->1
        pairs = torch.tensor([[0, 4, 0], [1, 4, 1]], dtype=torch.long)
        state = connect_connector_to_one_terminal(state, pairs)

        assert state.net_ids.shape == (2, 6)
        nb0 = state.net_ids[0]
        nb1 = state.net_ids[1]
        # env0: 0 connected (to connector), 1 remains unconnected (-1)
        assert nb0[0].item() >= 0 and nb0[1].item() == -1
        # env1: 1 connected (to connector), 0 remains unconnected (-1)
        assert nb1[1].item() >= 0 and nb1[0].item() == -1
