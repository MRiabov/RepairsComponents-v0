from dataclasses import dataclass, field
from typing import Dict, List, overload

import torch
from torch_geometric.data import Data
from repairs_components.logic.electronics.component import ElectricalComponent
from repairs_components.training_utils.sim_state import SimState
from repairs_components.logic.physical_state import _diff_edge_features


@dataclass # note: possibly doable with @tensorclass. But it'll be not easy.
class ElectronicsState(SimState):
    components: dict[str, ElectricalComponent] = field(default_factory=dict)
    graph: Data = field(default_factory=Data)
    indices: dict[str, int] = field(default_factory=dict)
    reverse_indices: dict[int, str] = field(default_factory=dict)
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
        self.graph = Data()
        self.graph.edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=self.device
        )
        # fixme: better go from graph.x to individual features... I think.
        self.graph.x = torch.empty(
            (0, 2), dtype=torch.float32, device=self.device
        )  # [max_voltage, max_current]
        self.graph.num_nodes = 0
        self._graph_built = False

    def _build_graph(self):
        if self._graph_built:
            return
        # Build index mapping
        self.indices = {name: idx for idx, name in enumerate(self.components.keys())}
        num_nodes = len(self.indices)

        # Initialize node features with max_load or [0,0] if not specified
        node_features = []
        for comp in self.components.values():
            if comp.max_load is not None:
                node_features.append([comp.max_load[0], comp.max_load[1]])
            else:
                node_features.append([0.0, 0.0])  # Default to no load limit

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

        data.x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        data.num_nodes = num_nodes
        self.graph = data
        self._graph_built = True

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
        assert self.graph.num_nodes > 0, "Graph must not be empty."
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
            self.graph.num_nodes if hasattr(self.graph, "num_nodes") else 0,
            other.graph.num_nodes if hasattr(other.graph, "num_nodes") else 0,
        )

        # Node features: 1 if node exists in both states, 0 otherwise
        node_exists = torch.zeros((num_nodes, 1), device=self.device)
        if hasattr(self.graph, "num_nodes") and hasattr(other.graph, "num_nodes"):
            common_nodes = min(self.graph.num_nodes, other.graph.num_nodes)
            if common_nodes > 0:
                # For common nodes, we can compute load differences
                if hasattr(self.graph, "x") and hasattr(other.graph, "x"):
                    load_diff = torch.abs(
                        self.graph.x[:common_nodes] - other.graph.x[:common_nodes]
                    )
                    node_exists[:common_nodes] = (
                        (load_diff > 1e-6).any(dim=1, keepdim=True).float()
                    )
                else:
                    node_exists[:common_nodes] = 1.0

        # Node mask: True if node exists in either state
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        if hasattr(self.graph, "num_nodes") and self.graph.num_nodes > 0:
            node_mask[: self.graph.num_nodes] = True
        if hasattr(other.graph, "num_nodes") and other.graph.num_nodes > 0:
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
        edge_attr = diff_graph.edge_attr if hasattr(diff_graph, "edge_attr") else None
        edge_index = (
            diff_graph.edge_index if hasattr(diff_graph, "edge_index") else None
        )

        added_edges = []
        removed_edges = []

        if edge_attr is not None and edge_index is not None and edge_attr.size(0) > 0:
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
        if hasattr(diff_graph, "node_mask"):
            for i in range(diff_graph.node_mask.size(0)):
                if diff_graph.node_mask[i]:
                    node_info = {"id": index_to_name.get(i, str(i))}
                    # Add load information if available
                    if (
                        hasattr(diff_graph, "x")
                        and diff_graph.x is not None
                        and i < diff_graph.x.size(0)
                    ):
                        if diff_graph.x[i].dim() > 0:  # If it's not a scalar
                            node_info["load_diff"] = diff_graph.x[i].tolist()
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
        if hasattr(self.graph, "x") and self.graph.x is not None:
            # Add new node feature row with max_load or [0,0] if not specified
            if component.max_load is not None:
                new_feature = torch.tensor(
                    [[component.max_load[0], component.max_load[1]]], device=self.device
                )
            else:
                new_feature = torch.zeros((1, 2), device=self.device)
            self.graph.x = torch.cat([self.graph.x, new_feature], dim=0)
            self.graph.num_nodes = len(self.components)

    # def register_contacts(self, contacts: dict[str, tuple[str, str]]):
    #     "Register components of body A to bodies B"
    def connect(self, name: str, other_name: str):
        "Connect two components"
        self.components[name].connect(self.components[other_name])
        self.components[other_name].connect(self.components[name])
        self._graph_built = False

    @overload
    def connect(self, id_1: int, id_2: int):
        "Connect two components"
        # not good but will do
        self.connect(self.reverse_indices[id_1], self.reverse_indices[id_2])

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
