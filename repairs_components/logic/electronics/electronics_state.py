from dataclasses import dataclass, field
from typing import Dict, List, overload

import torch
from torch_geometric.data import Data
from repairs_components.geometry.connectors.connectors import ConnectorsEnum
from repairs_components.logic.electronics.component import (
    ElectricalComponent,
    ElectricalComponentsEnum,
)
from repairs_components.training_utils.sim_state import SimState
from repairs_components.logic.physical_state import _diff_edge_features


@dataclass  # note: possibly doable with @tensorclass. But it'll be not easy.
class ElectronicsState(SimState):
    components: dict[str, ElectricalComponent] = field(default_factory=dict)
    graph: Data = field(default_factory=Data)
    "Note: don't use this graph for learning. Use export_graph() instead."  # TODO: merge all node features into one.
    indices: dict[str, int] = field(default_factory=dict)
    inverse_indices: dict[int, str] = field(default_factory=dict)
    # TODO add reverse indices updates where they need to be.
    _graph_built: bool = field(default=False, init=False)
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def __post_init__(self):
        # Initialize sparse graph representation
        self.indices = {}
        self.inverse_indices = {}
        self.graph = Data()
        self.graph.edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=self.device
        )  # NOTE: I think there is sense in keeping edge index in electronics because of many-to-many connections, unlike with fasteners.
        # Initialize individual feature tensors
        self.graph.max_voltage = torch.tensor(
            [], dtype=torch.float32, device=self.device
        )
        self.graph.max_current = torch.tensor(
            [], dtype=torch.float32, device=self.device
        )
        self.graph.component_type = torch.tensor(
            [], dtype=torch.long, device=self.device
        )
        self.graph.component_id = torch.tensor([], dtype=torch.long, device=self.device)
        self.graph.num_nodes = 0
        # TODO loose components? or are they unnecessary?
        self._graph_built = False

    def _build_graph(self):
        if self._graph_built:
            return

        # Build index mapping
        self.indices = {name: idx for idx, name in enumerate(self.components.keys())}
        self.inverse_indices = {idx: name for name, idx in self.indices.items()}
        num_nodes = len(self.indices)

        # Initialize feature tensors
        max_voltages = []
        max_currents = []
        component_types = []
        component_ids = []

        # Populate feature tensors from components
        for name, comp in self.components.items():
            if comp.max_load is not None:
                max_voltages.append(comp.max_load[0])
                max_currents.append(comp.max_load[1])
            else:
                max_voltages.append(0.0)
                max_currents.append(0.0)

            # Store component type and ID
            component_types.append(comp.component_type)
            component_ids.append(comp.component_id)

        # Gather edges
        edge_list: list[tuple[int, int]] = []
        for name, comp in self.components.items():
            src = self.indices[name]
            for other in comp.connected_to:
                dst = self.indices[other.name]
                edge_list.append((src, dst))

        # Create graph data
        data = Data()
        if edge_list:
            data.edge_index = (
                torch.tensor(edge_list, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        # Assign individual feature tensors
        data.max_voltage = torch.tensor(
            max_voltages, dtype=torch.float32, device=self.device
        )
        data.max_current = torch.tensor(
            max_currents, dtype=torch.float32, device=self.device
        )
        data.component_type = torch.tensor(
            component_types, dtype=torch.long, device=self.device
        )
        data.component_id = torch.tensor(
            component_ids, dtype=torch.long, device=self.device
        )
        data.num_nodes = num_nodes

        self.graph = data
        self._graph_built = True

    def export_graph(self):
        # sanity check:
        assert len(self.components) == len(self.indices), (
            "Graph indices do not match number of components."
        )
        assert len(self.components) == self.graph.num_nodes, (
            "Graph num_nodes does not match number of components."
        )
        # print(f"Exporting graph with {len(self.components)} components.")
        new_graph = Data(
            x=torch.cat(
                [
                    self.graph.max_voltage,
                    self.graph.max_current,
                    self.graph.component_type,
                    self.graph.component_id,
                ],
                dim=-1,
            ),
            num_nodes=len(self.components),
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,  # e.g. fastener size.
        )
        return new_graph

    def diff(self, other: "ElectronicsState") -> tuple[Data, int]:
        """Compute a graph diff between two electronic states.

        Returns:
            tuple[Data, int]: A tuple containing:
                - A PyG Data object representing the diff with:
                    - x: Node features [num_nodes, 1] (1 if node exists in both states)
                    - edge_index: Edge connections [2, num_edges]
                    - edge_attr: Edge features [num_edges, 3] (is_added, is_removed, is_changed)
                    - node_mask: Boolean mask of nodes that exist in either state [num_nodes]
                    - edge_mask: Boolean mask of changed edges [num_edges]
                    - num_nodes: Total number of nodes
                - An integer count of the total number of differences
        """
        # if no electronics is present...
        if len(self.components) == 0:
            assert len(other.components) == 0, "Other graph must be empty."
            return Data(), 0
        assert self.graph.num_nodes is not None and other.graph.num_nodes is not None, (
            f"Graphs must have num_nodes assigned. This: {self.graph.num_nodes}."
        )
        assert other.graph.num_nodes > 0, "Graph must not be empty."
        assert other.graph.num_nodes > 0, "Compared graph must not be empty."
        assert self.graph.num_nodes == other.graph.num_nodes, (
            "Graphs must have the same number of nodes."
        )
        # Ensure graphs are built
        self._build_graph()
        other._build_graph()

        # Get edge differences
        edge_diff, edge_diff_count = _diff_edge_features(self.graph, other.graph)

        # Calculate total differences (only edge differences for now, as nodes are just existence checks)
        total_diff_count = edge_diff_count

        # Create diff graph with same nodes as original
        diff_graph = Data()
        num_nodes = max(
            self.graph.num_nodes,
            other.graph.num_nodes,
        )

        # Node features: 1 if node exists in both states, 0 otherwise
        node_exists = torch.zeros((num_nodes, 1), device=self.device)
        common_nodes = min(self.graph.num_nodes, other.graph.num_nodes)
        if common_nodes > 0:
            # For common nodes, we can compute load differences
            has_diff = False

            # Check max_voltage differences
            if len(self.graph.max_voltage) > 0 and len(other.graph.max_voltage) > 0:
                voltage_diff = torch.abs(
                    self.graph.max_voltage[:common_nodes]
                    - other.graph.max_voltage[:common_nodes]
                )
                has_diff = has_diff | (voltage_diff > 1e-6)

            # Check max_current differences
            if len(self.graph.max_current) > 0 and len(other.graph.max_current) > 0:
                current_diff = torch.abs(
                    self.graph.max_current[:common_nodes]
                    - other.graph.max_current[:common_nodes]
                )
                has_diff = has_diff | (current_diff > 1e-6)

            # Check component_type differences
            if (
                len(self.graph.component_type) > 0
                and len(other.graph.component_type) > 0
            ):
                type_diff = (
                    self.graph.component_type[:common_nodes]
                    != other.graph.component_type[:common_nodes]
                )
                has_diff = has_diff | type_diff

            # Check component_id differences
            if len(self.graph.component_id) > 0 and len(other.graph.component_id) > 0:
                id_diff = (
                    self.graph.component_id[:common_nodes]
                    != other.graph.component_id[:common_nodes]
                )
                has_diff = has_diff | id_diff

            node_exists[:common_nodes] = has_diff.float().view(-1, 1)

        # Node mask: True if node exists in either state
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        if self.graph.num_nodes > 0:
            node_mask[: self.graph.num_nodes] = True
        if other.graph.num_nodes > 0:
            node_mask[: other.graph.num_nodes] = True

        diff_graph.x = node_exists
        diff_graph.node_mask = node_mask

        # Edge features and mask
        edge_index = torch.tensor([], dtype=torch.long, device=self.device).reshape(
            2, 0
        )
        edge_attr = torch.tensor([], device=self.device).reshape(
            0, 3
        )  # [is_added, is_removed, is_changed]
        edge_mask = torch.tensor([], dtype=torch.bool, device=self.device)

        if edge_diff["added"] or edge_diff["removed"]:
            added_edges = (
                torch.tensor(edge_diff["added"], device=self.device).t()
                if edge_diff["added"]
                else torch.tensor([], device=self.device).long().reshape(2, 0)
            )
            removed_edges = (
                torch.tensor(edge_diff["removed"], device=self.device).t()
                if edge_diff["removed"]
                else torch.tensor([], device=self.device).long().reshape(2, 0)
            )

            # Combine all edges
            edge_index = torch.cat([added_edges, removed_edges], dim=1)

            # Create edge features
            added_attrs = torch.zeros((added_edges.size(1), 3), device=self.device)
            added_attrs[:, 0] = 1  # is_added
            added_attrs[:, 2] = 1  # is_changed

            removed_attrs = torch.zeros((removed_edges.size(1), 3), device=self.device)
            removed_attrs[:, 1] = 1  # is_removed
            removed_attrs[:, 2] = 1  # is_changed

            edge_attr = torch.cat([added_attrs, removed_attrs], dim=0)
            edge_mask = torch.ones(
                edge_index.size(1), dtype=torch.bool, device=self.device
            )

        diff_graph.edge_index = edge_index
        diff_graph.edge_attr = edge_attr
        diff_graph.edge_mask = edge_mask
        diff_graph.num_nodes = num_nodes

        return diff_graph, total_diff_count

    def diff_to_dict(self, diff_graph: Data) -> dict:
        """Convert the graph diff to a human-readable dictionary format.

        Args:
            diff_graph: The graph diff returned by diff()

        Returns:
            dict: A dictionary with 'nodes' and 'edges' keys containing
                  human-readable diff information
        """
        # Build index to name mapping
        index_to_name = {idx: name for name, idx in self.indices.items()}

        # Get added and removed edges with component names
        edge_attr = diff_graph.edge_attr
        edge_index = diff_graph.edge_index

        added_edges = []
        removed_edges = []

        if edge_attr.size(0) > 0:
            added_mask = edge_attr[:, 0].bool()
            removed_mask = edge_attr[:, 1].bool()

            added_edges = [
                (
                    index_to_name.get(src.item(), str(src.item())),
                    index_to_name.get(dst.item(), str(dst.item())),
                )
                for src, dst in edge_index[:, added_mask].t()
            ]

            removed_edges = [
                (
                    index_to_name.get(src.item(), str(src.item())),
                    index_to_name.get(dst.item(), str(dst.item())),
                )
                for src, dst in edge_index[:, removed_mask].t()
            ]

        # Get node information
        nodes = []
        for i in range(diff_graph.node_mask.size(0)):
            if diff_graph.node_mask[i]:
                node_info = {"id": index_to_name.get(i, str(i))}
                # Add feature information
                node_features = {}

                if i < len(diff_graph.max_voltage):
                    node_features["max_voltage"] = diff_graph.max_voltage[i].item()

                if i < len(diff_graph.max_current):
                    node_features["max_current"] = diff_graph.max_current[i].item()

                if i < len(diff_graph.component_type):
                    node_features["component_type"] = diff_graph.component_type[
                        i
                    ].item()

                if i < len(diff_graph.component_id):
                    node_features["component_id"] = diff_graph.component_id[i].item()

                if node_features:
                    node_info["features"] = node_features
                nodes.append(node_info)

        return {
            "nodes": nodes,
            "edges": {"added": added_edges, "removed": removed_edges},
        }

    def diff_to_str(self, diff_graph: Data) -> str:
        """Convert the graph diff to a human-readable string.

        Args:
            diff_graph: The graph diff returned by diff()

        Returns:
            str: A formatted string describing the diff
        """
        diff_dict = self.diff_to_dict(diff_graph)
        lines = ["Electronics State Diff:", "=" * 50]

        # Edge changes
        added_edges = diff_dict["edges"]["added"]
        removed_edges = diff_dict["edges"]["removed"]

        if added_edges:
            lines.append("\nAdded Connections:")
            for src, dst in added_edges:
                lines.append(f"  {src} -> {dst}")

        if removed_edges:
            lines.append("\nRemoved Connections:")
            for src, dst in removed_edges:
                lines.append(f"  {src} -> {dst}")

        if not (added_edges or removed_edges):
            lines.append("No changes detected in connections.")

        return "\n".join(lines)

    def register(self, component: ElectricalComponent):
        """Register a new electrical component."""
        assert component.name not in self.components, (
            f"Component {component.name} already registered"
        )
        self.components[component.name] = component
        self._graph_built = False  # Invalidate graph cache

        # Update node features if graph is already built
        if hasattr(self.graph, "max_voltage"):  # good use of hasattr.
            # Add new node features
            max_voltage = (
                component.max_load[0] if component.max_load is not None else 0.0
            )
            max_current = (
                component.max_load[1] if component.max_load is not None else 0.0
            )
            component_type = (
                component.component_type if hasattr(component, "component_type") else 0
            )
            component_id = (
                component.component_id if hasattr(component, "component_id") else 0
            )

            # Append new features to each tensor
            self.graph.max_voltage = torch.cat(
                [
                    self.graph.max_voltage,
                    torch.tensor(
                        [max_voltage], dtype=torch.float32, device=self.device
                    ),
                ]
            )
            self.graph.max_current = torch.cat(
                [
                    self.graph.max_current,
                    torch.tensor(
                        [max_current], dtype=torch.float32, device=self.device
                    ),
                ]
            )
            self.graph.component_type = torch.cat(
                [
                    self.graph.component_type,
                    torch.tensor(
                        [component_type], dtype=torch.long, device=self.device
                    ),
                ]
            )
            self.graph.component_id = torch.cat(
                [
                    self.graph.component_id,
                    torch.tensor([component_id], dtype=torch.long, device=self.device),
                ]
            )

            self.graph.num_nodes = len(self.components)

    # def register_contacts(self, contacts: dict[str, tuple[str, str]]):
    #     "Register components of body A to bodies B"
    def connect(self, name: str, other_name: str):
        "Connect two components"
        self.components[name].connect(self.components[other_name])
        self.components[other_name].connect(self.components[name])
        self._graph_built = False

    def clear_connections(self):
        "Disconnect all connections between a component"
        for component in self.components.values():
            component.connected_to = []
        self._graph_built = False

    def check_if_electronics_connected(self, id_1: int, id_2: int):
        "Check if two components are connected"
        connection = self.graph.edge_index == torch.tensor(
            [id_1, id_2], device=self.device
        )
        inverse_connection = self.graph.edge_index == torch.tensor(
            [id_2, id_1], device=self.device
        )
        return connection.any() or inverse_connection.any()

    @classmethod
    def rebuild_from_graph(
        cls, graph: Data, indices: dict[str, int]
    ) -> "ElectronicsState":
        "Rebuild an ElectronicsState from a graph"
        assert graph.num_nodes is not None
        assert graph.num_nodes == len(indices), (
            f"Graph and indices do not match: {graph.num_nodes} != {len(indices)}"
        )

        # Create a new ElectronicsState instance
        state = cls()
        state.indices = indices
        state.inverse_indices = {v: k for k, v in indices.items()}
        state.graph = graph

        # Rebuild components from individual feature tensors
        for component_id in range(graph.num_nodes):
            component_name = state.inverse_indices[component_id]

            # Get component features from individual tensors
            max_voltage = graph.max_voltage[component_id].item()
            max_current = graph.max_current[component_id].item()
            component_type_val = graph.component_type[component_id].item()
            component_id_val = graph.component_id[component_id].item()

            # Create component based on type
            try:
                component_enum = ElectricalComponentsEnum(component_type_val)

                match component_enum:
                    case ElectricalComponentsEnum.CONNECTOR:
                        component = ConnectorsEnum(component_name)
                    # Add other component types as needed
                    case _:
                        raise NotImplementedError(
                            f"Rebuilding component type {component_enum} not implemented"
                        )

                # Set component properties
                if hasattr(component, "max_load"):
                    component.max_load = (max_voltage, max_current)
                if hasattr(component, "component_id"):
                    component.component_id = component_id_val

                state.components[component_name] = component

            except ValueError as e:
                raise ValueError(
                    f"Invalid component type value: {component_type_val}"
                ) from e

        # Rebuild connections from edge_index
        if hasattr(graph, "edge_index") and graph.edge_index is not None:
            for src_idx, dst_idx in graph.edge_index.t().tolist():
                src_name = state.inverse_indices[src_idx]
                dst_name = state.inverse_indices[dst_idx]
                state.components[src_name].connect(state.components[dst_name])

        state._graph_built = True
        return state
