import pytest
import torch
from torch_geometric.data import Data

from repairs_components.logic.electronics.electronics_state import (
    register_components_batch,
    connect_terminal_to_net_or_create_new,
)
from repairs_components.logic.electronics.component import (
    ElectricalComponentsEnum as ECE,
)
from repairs_components.logic.electronics.electronics_state import ElectronicsState


class TestRegisterComponentsBatch:
    def test_basic_registration_shapes_and_meta(self):
        names = ["batt", "res"]
        component_types = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        max_voltages = torch.tensor([9.0, 0.0], dtype=torch.float32)
        max_currents = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component = torch.tensor([2, 2], dtype=torch.long)
        device = torch.device("cpu")

        comp_info, state = register_components_batch(
            names,
            component_types,
            max_voltages,
            max_currents,
            terminals_per_component,
            device=device,
            batch_size=2,
        )
        # terminal_to_component length equals total terminals
        T = terminals_per_component.sum().item()
        assert state.net_ids.shape == (2, T)
        assert comp_info.terminal_to_component_batch.shape[0] == T
        # net_ids initialized to -1
        assert state.net_ids.max().item() == -1
        # name maps present
        assert comp_info.component_indices_from_name["batt"] == 0
        assert comp_info.inverse_component_indices[0] == "batt"

    def test_connectors_auto_connect_and_map_to_terminals(self):
        # Components: male connector (1t), female connector (1t), wire (2t)
        names = [
            "plug_male@connector",
            "plug_female@connector",
            "wire1",
        ]
        component_types = torch.tensor(
            [int(ECE.CONNECTOR), int(ECE.CONNECTOR), int(ECE.WIRE)], dtype=torch.long
        )
        max_voltages = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        max_currents = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        terminals_per_component = torch.tensor([1, 1, 2], dtype=torch.long)
        device = torch.device("cpu")

        # Terminal indices: male:0, female:1, wire:2,3
        conn_a = torch.tensor([2], dtype=torch.long)  # male -> wire.t0
        conn_b = torch.tensor([3], dtype=torch.long)  # female -> wire.t1

        comp_info, state = register_components_batch(
            names,
            component_types,
            max_voltages,
            max_currents,
            terminals_per_component,
            device=device,
            batch_size=1,
            connector_terminal_connectivity_a=conn_a,
            connector_terminal_connectivity_b=conn_b,
        )

        # Expect three connections all on the same net: male<->female, male<->wire.t0, female<->wire.t1
        # Which implies all four terminals share the same net id
        net = state.net_ids[0]
        gids = net[torch.tensor([0, 1, 2, 3])]
        assert torch.unique(gids).numel() == 1

    def test_connector_bounds_and_validation(self):
        names = [
            "plug_male@connector",
            "plug_female@connector",
            "wire1",
        ]
        component_types = torch.tensor(
            [int(ECE.CONNECTOR), int(ECE.CONNECTOR), int(ECE.WIRE)], dtype=torch.long
        )
        max_voltages = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        max_currents = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        terminals_per_component = torch.tensor([1, 1, 2], dtype=torch.long)
        device = torch.device("cpu")
        T = terminals_per_component.sum().item()

        # Out of bounds should assert
        with pytest.raises(AssertionError):
            register_components_batch(
                names,
                component_types,
                max_voltages,
                max_currents,
                terminals_per_component,
                device=device,
                batch_size=1,
                connector_terminal_connectivity_a=torch.tensor([T], dtype=torch.long),
                connector_terminal_connectivity_b=torch.tensor([3], dtype=torch.long),
            )

        # Mismatched lengths should assert
        with pytest.raises(AssertionError):
            register_components_batch(
                names,
                component_types,
                max_voltages,
                max_currents,
                terminals_per_component,
                device=device,
                batch_size=1,
                connector_terminal_connectivity_a=torch.tensor(
                    [2, 2], dtype=torch.long
                ),
                connector_terminal_connectivity_b=torch.tensor([3], dtype=torch.long),
            )


def make_simple_state(batch_size: int = 1):
    # Two non-connector components, each with 2 terminals: total T=4
    names = ["batt", "res"]
    component_types = torch.tensor(
        [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
    )
    max_voltages = torch.tensor([9.0, 0.0], dtype=torch.float32)
    max_currents = torch.tensor([1.0, 0.1], dtype=torch.float32)
    terminals_per_component = torch.tensor([2, 2], dtype=torch.long)

    device = torch.device("cpu")
    comp_info, state = register_components_batch(
        names,
        component_types,
        max_voltages,
        max_currents,
        terminals_per_component,
        device=device,
        batch_size=batch_size,
    )
    return comp_info, state


class TestConnectTerminalToNetOrCreateNew:
    def test_create_new_net_both_unassigned(self):
        ci, st = make_simple_state(batch_size=1)
        # connect terminal 0 to 2 in batch 0 => new net created
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0], dtype=torch.long),
            terminal_ids=torch.tensor([0], dtype=torch.long),
            other_terminal_ids=torch.tensor([2], dtype=torch.long),
        )
        gid = st.net_ids[0, 0].item()
        assert gid >= 0
        assert st.net_ids[0, 2].item() == gid

    def test_attach_to_existing_net(self):
        ci, st = make_simple_state(batch_size=1)
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([0]),
            other_terminal_ids=torch.tensor([2]),
        )
        # attach terminal 3 to the same net via terminal 0
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([0]),
            other_terminal_ids=torch.tensor([3]),
        )
        gid = st.net_ids[0, 0].item()
        assert st.net_ids[0, 3].item() == gid

    def test_merge_two_nets(self):
        ci, st = make_simple_state(batch_size=1)
        # Create two nets: (0,1) and (2,3)
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0, 0]),
            terminal_ids=torch.tensor([0, 2]),
            other_terminal_ids=torch.tensor([1, 3]),
        )
        gid01 = st.net_ids[0, 0].item()
        gid23 = st.net_ids[0, 2].item()
        assert gid01 != gid23
        # Merge by connecting 1 and 2
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([1]),
            other_terminal_ids=torch.tensor([2]),
        )
        gids = st.net_ids[0, torch.tensor([0, 1, 2, 3])]
        assert torch.unique(gids).numel() == 1
        assert torch.unique(gids)[0].item() == min(gid01, gid23)

    def test_target_net_attach_and_merge(self):
        ci, st = make_simple_state(batch_size=1)
        # Assign explicit net to terminal 0
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([0]),
            target_net_ids=torch.tensor([5]),
        )
        assert st.net_ids[0, 0].item() == 5
        # Merge into net 3
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([0]),
            target_net_ids=torch.tensor([3]),
        )
        assert st.net_ids[0, 0].item() == 3
        # Ignore -1 target
        st2 = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([1]),
            target_net_ids=torch.tensor([-1]),
        )
        assert st2.net_ids[0, 1].item() == -1

    def test_multiple_pairs_and_next_id(self):
        ci, st = make_simple_state(batch_size=1)
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0, 0]),
            terminal_ids=torch.tensor([0, 2]),
            other_terminal_ids=torch.tensor([1, 3]),
        )
        # Expect two nets: {0,1} and {2,3}
        gid01 = st.net_ids[0, 0].item()
        assert st.net_ids[0, 1].item() == gid01
        gid23 = st.net_ids[0, 2].item()
        assert st.net_ids[0, 3].item() == gid23
        assert gid01 != gid23

    def test_ignores_negative_other(self):
        ci, st = make_simple_state(batch_size=1)
        st2 = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0]),
            terminal_ids=torch.tensor([0]),
            other_terminal_ids=torch.tensor([-1]),
        )
        # unchanged
        assert torch.equal(st2.net_ids, st.net_ids)

    def test_per_batch_isolated(self):
        ci, st = make_simple_state(batch_size=2)
        # batch 0: connect (0,1); batch 1: connect (2,3)
        st = connect_terminal_to_net_or_create_new(
            st,
            ci,
            batch_ids=torch.tensor([0, 1]),
            terminal_ids=torch.tensor([0, 2]),
            other_terminal_ids=torch.tensor([1, 3]),
        )
        gid_b0 = st.net_ids[0, 0].item()
        assert st.net_ids[0, 1].item() == gid_b0
        gid_b1 = st.net_ids[1, 2].item()
        assert st.net_ids[1, 3].item() == gid_b1
        # Ensure different batches can reuse same net ids independently
        assert gid_b0 == 0 and gid_b1 == 0


class TestElectronicsDiff:
    def make_two_states_same_components(self):
        names = ["batt", "res"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max_A = torch.tensor([9.0, 0.0], dtype=torch.float32)
        voltage_max_B = torch.tensor([9.0, 0.0], dtype=torch.float32)
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)
        device = torch.device("cpu")
        comp_info, A = register_components_batch(
            names,
            component_type,
            voltage_max_A,
            current_max,
            terminals_per_component_batch,
            device=device,
            batch_size=1,
        )
        _, B = register_components_batch(
            names,
            component_type.clone(),
            voltage_max_B,
            current_max.clone(),
            terminals_per_component_batch.clone(),
            device=device,
            batch_size=1,
        )
        return comp_info, A, B

    def test_diff_edges_added(self):
        comp_info, A, B = self.make_two_states_same_components()
        # A: no edges; B: batt(0).t0(0) connected to res(1).t0(2)
        B = connect_terminal_to_net_or_create_new(
            B,
            comp_info,
            batch_ids=torch.tensor([0], dtype=torch.long),
            terminal_ids=torch.tensor([0], dtype=torch.long),
            other_terminal_ids=torch.tensor([2], dtype=torch.long),
        )

        diff_graph, n = A.diff(B, comp_info)
        assert isinstance(diff_graph, Data)
        assert isinstance(n, int)

        # One changed edge in symmetric difference
        assert isinstance(diff_graph.edge_index, torch.Tensor)
        assert diff_graph.edge_index.shape[1] == 1
        assert torch.is_tensor(diff_graph.edge_mask)
        assert diff_graph.edge_mask.numel() == 1
        assert diff_graph.edge_mask.sum().item() == 1
        assert isinstance(diff_graph.num_nodes, int)
        assert diff_graph.num_nodes == 2
        # No node feature changes
        assert diff_graph.node_mask.sum().item() == 0

        # Check dict representation
        d = A.diff_to_dict(diff_graph, comp_info)
        assert d["num_nodes"] == 2
        assert d["num_edges"] == 1
        assert len(d["edges_changed"]) == 1
        # Names should resolve via inverse_component_indices
        pair = d["edges_changed"][0]
        assert set(pair) == {"batt", "res"}

    def test_diff_edges_removed(self):
        comp_info, A, B = self.make_two_states_same_components()
        # A has the connection; B has no connections
        A = connect_terminal_to_net_or_create_new(
            A,
            comp_info,
            batch_ids=torch.tensor([0], dtype=torch.long),
            terminal_ids=torch.tensor([0], dtype=torch.long),
            other_terminal_ids=torch.tensor([2], dtype=torch.long),
        )

        diff_graph, n = A.diff(B, comp_info)
        assert isinstance(diff_graph.edge_index, torch.Tensor)
        assert diff_graph.edge_index.shape[1] == 1
        assert diff_graph.edge_mask.sum().item() == 1
        assert diff_graph.node_mask.sum().item() == 0
        assert n == 1

    def test_diff_nodes_changed(self):
        # Change a scalar node feature (max_voltage) on one component
        names = ["batt", "res"]
        component_type = torch.tensor(
            [int(ECE.VOLTAGE_SOURCE), int(ECE.RESISTOR)], dtype=torch.long
        )
        voltage_max_A = torch.tensor([9.0, 0.0], dtype=torch.float32)
        voltage_max_B = torch.tensor(
            [12.0, 0.0], dtype=torch.float32
        )  # changed batt voltage
        current_max = torch.tensor([1.0, 0.1], dtype=torch.float32)
        terminals_per_component_batch = torch.tensor([2, 2], dtype=torch.long)

        device = torch.device("cpu")
        comp_info, A = register_components_batch(
            names,
            component_type,
            voltage_max_A,
            current_max,
            terminals_per_component_batch,
            device=device,
            batch_size=1,
        )
        _comp_info_b, B = register_components_batch(
            names,
            component_type.clone(),
            voltage_max_B,
            current_max.clone(),
            terminals_per_component_batch.clone(),
            device=device,
            batch_size=1,
        )

        # Node features are defined by component_info and not compared across states.
        diff_graph, n = A.diff(B, comp_info)
        assert isinstance(n, int)
        # No node changes should be reported when comparing states with same component_info
        assert diff_graph.node_mask.sum().item() == 0
        assert isinstance(diff_graph.edge_index, torch.Tensor)
        assert diff_graph.edge_index.shape[1] == 0
        assert diff_graph.edge_mask.sum().item() == 0

        d = A.diff_to_dict(diff_graph, comp_info)
        assert len(d["nodes_changed"]) == 0

    def test_diff_to_str(self):
        comp_info, A, B = self.make_two_states_same_components()
        B = connect_terminal_to_net_or_create_new(
            B,
            comp_info,
            batch_ids=torch.tensor([0], dtype=torch.long),
            terminal_ids=torch.tensor([0], dtype=torch.long),
            other_terminal_ids=torch.tensor([2], dtype=torch.long),
        )
        diff_graph, _ = A.diff(B, comp_info)
        s = A.diff_to_str(diff_graph, comp_info)
        assert isinstance(s, str)
        assert "Edges changed:" in s

    def test_diff_with_batched_states_via_slice(self):
        # Ensure we can stack states (B>1) and still diff by slicing a single env
        comp_info, A, B = self.make_two_states_same_components()
        B = connect_terminal_to_net_or_create_new(
            B,
            comp_info,
            batch_ids=torch.tensor([0], dtype=torch.long),
            terminal_ids=torch.tensor([0], dtype=torch.long),
            other_terminal_ids=torch.tensor([2], dtype=torch.long),
        )

        Ab: ElectronicsState = torch.stack([A, A])  # type: ignore
        Bb: ElectronicsState = torch.stack([B, B])  # type: ignore

        # Slice to single env and compute diff as usual
        diff_graph, n = Ab[0].diff(Bb[0], comp_info)  # type: ignore[index]
        assert isinstance(n, int)
        assert isinstance(diff_graph.edge_index, torch.Tensor)
        assert diff_graph.edge_index.shape[1] == 1
        assert diff_graph.edge_mask.sum().item() == 1

    def test_diff_raises_on_terminal_count_mismatch(self):
        comp_info, A, _ = self.make_two_states_same_components()
        T = A.net_ids.shape[1]
        # Create a new state with mismatched terminal dimension
        B = ElectronicsState(device=A.device)
        B.net_ids = torch.full((A.batch_size[0], T + 1), -1, dtype=torch.long)
        with pytest.raises(AssertionError):
            A.diff(B, comp_info)
