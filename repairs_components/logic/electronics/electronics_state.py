from dataclasses import dataclass, field

import torch
from tensordict import TensorClass
from torch_geometric.data import Data
from repairs_components.geometry.connectors.connectors import ConnectorsEnum
from repairs_components.logic.electronics.component import (
    ElectricalComponent,
    ElectricalComponentsEnum,
)
from repairs_components.training_utils.sim_state import SimState
from typing_extensions import deprecated
from repairs_components.logic.physical_state import _diff_fastener_features


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
            x=torch.cat(  # was torch.cat... but it resulted in a wrong shape?
                [
                    self.graph.max_voltage.unsqueeze(-1),
                    self.graph.max_current.unsqueeze(-1),
                    self.graph.component_type.unsqueeze(-1),
                    self.graph.component_id.unsqueeze(-1),
                ],
                dim=-1,
            ),
            num_nodes=len(self.components),
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
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
        edge_diff, edge_diff_count = _diff_fastener_features(self.graph, other.graph)

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
        assert not component.name.endswith(("@connector")), (
            f"Connectors should not be registered in electrical state (yet). Failed at {component.name}."
        )

        self.components[component.name] = component
        assert not self._graph_built, (
            "Graph is already built, cannot register component"
        )
        # fixme: it better is to add to graph directly rather than to components. but that's later.
        # self._graph_built = False  # Invalidate graph cache
        # note: probably better to assert that graph is not built at all? why would it?

        # # Update node features if graph is already built
        # if hasattr(self.graph, "max_voltage"):  # good use of hasattr.
        #     # Add new node features
        #     max_voltage = (
        #         component.max_load[0] if component.max_load is not None else 0.0
        #     )
        #     max_current = (
        #         component.max_load[1] if component.max_load is not None else 0.0
        #     )
        #     component_type = (
        #         component.component_type if hasattr(component, "component_type") else 0
        #     )
        #     component_id = (
        #         component.component_id if hasattr(component, "component_id") else 0
        #     )

        #     # Append new features to each tensor
        #     self.graph.max_voltage = torch.cat(
        #         [
        #             self.graph.max_voltage,
        #             torch.tensor(
        #                 [max_voltage], dtype=torch.float32, device=self.device
        #             ),
        #         ]
        #     )
        #     self.graph.max_current = torch.cat(
        #         [
        #             self.graph.max_current,
        #             torch.tensor(
        #                 [max_current], dtype=torch.float32, device=self.device
        #             ),
        #         ]
        #     )
        #     self.graph.component_type = torch.cat(
        #         [
        #             self.graph.component_type,
        #             torch.tensor(
        #                 [component_type], dtype=torch.long, device=self.device
        #             ),
        #         ]
        #     )
        #     self.graph.component_id = torch.cat(
        #         [
        #             self.graph.component_id,
        #             torch.tensor([component_id], dtype=torch.long, device=self.device),
        #         ]
        #     )

        #     self.graph.num_nodes = len(self.components)

    # def register_contacts(self, contacts: dict[str, tuple[str, str]]):
    #     "Register components of body A to bodies B"
    def connect(self, name: str, other_name: str):
        "Connect two components"
        assert not name.endswith(("@connector")), (
            f"Connectors should not be registered or connected in electrical state (yet). Failed at {name}."
        )
        assert not other_name.endswith(("@connector")), (
            f"Connectors should not be registered or connected in electrical state (yet). Failed at {other_name}."
        )
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


# -----------------------------------------------------------------------------
# New TensorClass-based Electronics State (non-imperative, batchable)
# -----------------------------------------------------------------------------
class ElectronicsStateTC(TensorClass):
    """TensorClass-based electronics state.

    Design goals:
    - Represent component features as tensors for accelerator-friendly ops.
    - Represent electrical connectivity at terminal level via net labels per terminal.
    - Provide functional, batchable APIs (no in-place Python mutation of components).

    Conventions:
    - Use "terminal" terminology for electronic connection points.
    - net_id labels group terminals that are electrically the same node.
      Terminals that share the same non-negative net_id belong to the same net.
      A value of -1 means the terminal is not connected to any net yet.
    - Batch dimension (if present) is the first dimension for batchable tensors.
    """

    # Per-component tensors (shape: [N] or [B, N])
    component_type: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    component_id: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    max_voltage: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    max_current: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    terminals_per_component: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )

    # Terminal mapping (no batch dimension, constant mapping for all envs)
    # terminal_to_component maps each terminal id in [0, T) to owning component index in [0, N)
    terminal_to_component: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )

    # Connectivity: per-env terminal net assignments
    # Shape: [T] (no batch) or [B, T] (batched). Values: -1 for unassigned else non-negative net id.
    net_id: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )

    # Optional Python-side name-index mappings (like PhysicalState)
    # Can be a single dict for single env, or a list[dict] for batched envs
    component_indices_from_name: dict | list[dict] = field(default_factory=dict)
    inverse_component_indices: dict | list[dict] = field(default_factory=dict)

    def __post_init__(self):
        # Establish name maps; debatch Python dict fields like PhysicalState does.
        if isinstance(self.component_indices_from_name, list):
            assert self.batch_size and self.batch_size[0] >= 1, (
                "Batch size must be set for batched state"
            )
            assert len(self.component_indices_from_name) == self.batch_size[0], (
                "Expected one name-index dict per batch element"
            )
            first = self.component_indices_from_name[0]
            assert isinstance(first, dict), (
                "component_indices_from_name[0] must be a dict"
            )
            # Debatch to a single dict reference (assumed identical across envs)
            self.component_indices_from_name = first
            self.inverse_component_indices = {v: k for k, v in first.items()}
        elif isinstance(self.component_indices_from_name, dict):
            self.inverse_component_indices = {
                v: k for k, v in self.component_indices_from_name.items()
            }
        else:
            # leave empty if not set yet (to be filled in register function)
            self.inverse_component_indices = {}
        return self

    # ------------------------------------------------------------------
    # Registration API (functional)
    # ------------------------------------------------------------------
    @deprecated("Use register_components_batch(state, ...) instead.")
    def register_components_batch(
        self,
        names: list[str],
        component_types: torch.Tensor,  # [N]
        max_voltages: torch.Tensor,  # [N]
        max_currents: torch.Tensor,  # [N]
        terminals_per_component: torch.Tensor,  # [N]
        component_ids: torch.Tensor | None = None,  # [N], optional
    ) -> "ElectronicsStateTC":
        return register_components_batch(
            self,
            names,
            component_types,
            max_voltages,
            max_currents,
            terminals_per_component,
            component_ids,
        )

    # ------------------------------------------------------------------
    # Connectivity API (functional, batchable)
    # ------------------------------------------------------------------
    @deprecated("Use connect_terminals_batch(state, ...) instead.")
    def connect_terminals_batch(
        self, terminal_pairs: torch.Tensor
    ) -> "ElectronicsStateTC":
        return connect_terminals_batch(self, terminal_pairs)

    def clear_all_connections(self) -> "ElectronicsStateTC":
        """Reset all terminal nets to -1 (no connectivity)."""
        device = self.device
        if self.net_id.ndim == 2:
            B, T = self.net_id.shape
            self.net_id = torch.full((B, T), -1, dtype=torch.long, device=device)
        elif self.net_id.ndim == 1:
            T = self.net_id.shape[0]
            self.net_id = torch.full((T,), -1, dtype=torch.long, device=device)
        else:
            self.net_id = torch.empty((0,), dtype=torch.long, device=device)
        return self

    # ------------------------------------------------------------------
    # Export & Diff APIs
    # ------------------------------------------------------------------
    def _build_component_edges_from_nets(self) -> torch.Tensor:
        """Derive undirected component-level edges from terminal net assignments.

        Returns:
            edge_index [2, E] for single-env.
        Note: If batched, this currently supports only single-env and will raise.
        """
        assert self.terminal_to_component.numel() > 0, "Register components first"
        if self.net_id.ndim == 2:
            # For now, keep export single-env to avoid mixing batches
            raise NotImplementedError(
                "export for batched ElectronicsStateTC not yet supported"
            )

        net = self.net_id
        T = int(self.terminal_to_component.numel())
        assert net.shape[0] == T, "net_id length must equal number of terminals"

        # Build nets -> terminals mapping
        valid_mask = net >= 0
        if not valid_mask.any():
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        unique_nets = torch.unique(net[valid_mask])
        edges_set: set[tuple[int, int]] = set()
        for gid in unique_nets.tolist():
            term_idx = torch.nonzero(net == gid, as_tuple=False).flatten()
            comp_idx = self.terminal_to_component[term_idx]
            # Unique components participating in this net
            comps = torch.unique(comp_idx).tolist()
            # Generate all i<j pairs among components in this net
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    a, b = comps[i], comps[j]
                    if a == b:
                        continue
                    u, v = (a, b) if a < b else (b, a)
                    edges_set.add((int(u), int(v)))

        if not edges_set:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_list = torch.tensor(
            sorted(list(edges_set)), dtype=torch.long, device=self.device
        )
        return edge_list.t().contiguous()

    def export_graph(self) -> Data:
        """Export a torch_geometric.Data graph for ML consumers.

        Node features: [max_voltage, max_current, component_type, component_id]
        Edge index: derived from terminal net connectivity at component level.
        """
        edge_index = self._build_component_edges_from_nets()

        # Normalize shapes of node features to [N, F]
        def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                # Validate equal across batch then use first env
                assert torch.allclose(x, x[:1].expand_as(x)), (
                    "Per-component features must match across batch for export"
                )
                return x[0]
            return x

        mv = _ensure_1d(self.max_voltage).to(torch.float32)
        mc = _ensure_1d(self.max_current).to(torch.float32)
        ctype = _ensure_1d(self.component_type).to(torch.float32)
        cid = _ensure_1d(self.component_id).to(torch.float32)
        x = torch.stack([mv, mc, ctype, cid], dim=1).bfloat16()

        graph = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
        return graph

    def diff(self, other: "ElectronicsStateTC") -> tuple[Data, int]:
        """Compute a graph diff between two electronics states (component-level edges).

        Returns a PyG Data with:
            - x: node features (from self)
            - edge_index: combined edges from both states (union)
            - node_mask: bool mask of nodes whose scalar features differ
            - edge_mask: bool mask of edges that were added/removed/changed
        and an integer count of total differences (nodes + edges).
        """
        # Build edge sets
        e_self = self._build_component_edges_from_nets()
        e_other = other._build_component_edges_from_nets()

        def _edges_to_set(ei: torch.Tensor) -> set[tuple[int, int]]:
            return set(map(tuple, ei.t().tolist())) if ei.numel() > 0 else set()

        set_a = _edges_to_set(e_self)
        set_b = _edges_to_set(e_other)
        all_edges_sorted = sorted(list(set_a.union(set_b)))
        edge_index = (
            torch.tensor(all_edges_sorted, dtype=torch.long, device=self.device).t()
            if all_edges_sorted
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )

        # Node feature deltas
        # Use tolerances of zero for now (different max_voltage/current or type/id => changed)
        def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                assert torch.allclose(x, x[:1].expand_as(x)), (
                    "Per-component features must match across batch for diff"
                )
                return x[0]
            return x

        mv_a, mv_b = _ensure_1d(self.max_voltage), _ensure_1d(other.max_voltage)
        mc_a, mc_b = _ensure_1d(self.max_current), _ensure_1d(other.max_current)
        ct_a, ct_b = _ensure_1d(self.component_type), _ensure_1d(other.component_type)
        id_a, id_b = _ensure_1d(self.component_id), _ensure_1d(other.component_id)

        # Align N
        assert mv_a.shape[0] == mv_b.shape[0], "Node count mismatch between states"
        node_mask = (mv_a != mv_b) | (mc_a != mc_b) | (ct_a != ct_b) | (id_a != id_b)

        # Edge mask: edges in symmetric difference
        set_sym = set_a.symmetric_difference(set_b)
        edge_mask = (
            torch.tensor(
                [tuple(e) in set_sym for e in all_edges_sorted],
                dtype=torch.bool,
                device=self.device,
            )
            if all_edges_sorted
            else torch.empty((0,), dtype=torch.bool, device=self.device)
        )

        diff_graph = Data(
            x=torch.stack(
                [mv_a, mc_a, ct_a.to(torch.float32), id_a.to(torch.float32)], dim=1
            ).bfloat16(),
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            num_nodes=mv_a.shape[0],
        )

        n_diffs = int(node_mask.sum().item()) + int(edge_mask.sum().item())
        return diff_graph, n_diffs

    def diff_to_dict(self, diff_graph: Data) -> dict:
        """Convert diff graph to human-readable dict."""
        nodes_changed = (
            torch.nonzero(diff_graph.node_mask, as_tuple=False).flatten().tolist()
        )
        edges_changed = (
            torch.nonzero(diff_graph.edge_mask, as_tuple=False).flatten().tolist()
        )

        # Map indices to names if available
        def _idx_to_name(i: int) -> str:
            if isinstance(self.inverse_component_indices, dict):
                return self.inverse_component_indices.get(i, str(i))
            if (
                isinstance(self.inverse_component_indices, list)
                and len(self.inverse_component_indices) > 0
                and isinstance(self.inverse_component_indices[0], dict)
            ):
                return self.inverse_component_indices[0].get(i, str(i))
            return str(i)

        # Build edge list tuples with names if possible
        edge_pairs: list[tuple[str, str]] = []
        edge_index = getattr(diff_graph, "edge_index", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            for u, v in edge_index.t().tolist():
                edge_pairs.append((_idx_to_name(int(u)), _idx_to_name(int(v))))

        # Only include changed edges in listing
        changed_edge_pairs = (
            [edge_pairs[i] for i in edges_changed] if edge_pairs else []
        )

        return {
            "nodes_changed": [
                {"index": i, "name": _idx_to_name(i)} for i in nodes_changed
            ],
            "edges_changed": changed_edge_pairs,
            "num_nodes": int(diff_graph.num_nodes),
            "num_edges": (
                int(edge_index.shape[1])
                if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2
                else 0
            ),
        }

    def diff_to_str(self, diff_graph: Data) -> str:
        d = self.diff_to_dict(diff_graph)
        parts = [
            f"Nodes changed: {len(d['nodes_changed'])}",
            f"Edges changed: {len(d['edges_changed'])}",
        ]
        if d["nodes_changed"]:
            parts.append(
                "Changed nodes: "
                + ", ".join([f"{n['name']}({n['index']})" for n in d["nodes_changed"]])
            )
        if d["edges_changed"]:
            parts.append(
                "Changed edges: "
                + ", ".join([f"{u}-{v}" for u, v in d["edges_changed"]])
            )
        return "\n".join(parts)


# -----------------------------------------------------------------------------
# Standalone functions for batch processing ElectronicsStateTC
# -----------------------------------------------------------------------------
def register_components_batch(
    electronics_states: "ElectronicsStateTC",
    names: list[str],
    component_types: torch.Tensor,
    max_voltages: torch.Tensor,
    max_currents: torch.Tensor,
    terminals_per_component: torch.Tensor,
    component_ids: torch.Tensor | None = None,
) -> "ElectronicsStateTC":
    state = electronics_states
    device = state.device
    N = len(names)
    assert component_types.shape[-1] == N
    assert max_voltages.shape[-1] == N
    assert max_currents.shape[-1] == N
    assert terminals_per_component.shape[-1] == N

    def to_dev(x: torch.Tensor) -> torch.Tensor:
        return x.to(device)
    state.component_type = to_dev(component_types)
    state.max_voltage = to_dev(max_voltages)
    state.max_current = to_dev(max_currents)
    state.terminals_per_component = to_dev(terminals_per_component)

    if component_ids is None:
        state.component_id = torch.arange(N, dtype=torch.long, device=device)
    else:
        assert component_ids.shape[-1] == N
        state.component_id = to_dev(component_ids)

    assert state.terminals_per_component.ndim in (1, 2)
    tpc = state.terminals_per_component
    if tpc.ndim == 2:
        assert torch.allclose(tpc, tpc[:1].expand_as(tpc))
        tpc = tpc[0]

    comp_idx = torch.arange(N, device=device)
    state.terminal_to_component = torch.repeat_interleave(comp_idx, tpc)
    T = int(state.terminal_to_component.numel())

    # Set name-index maps once (kept identical across batch)
    state.component_indices_from_name = {name: i for i, name in enumerate(names)}
    state.inverse_component_indices = {
        i: n for n, i in state.component_indices_from_name.items()
    }

    if state.batch_size and state.batch_size[0] >= 1:
        B = state.batch_size[0]
        state.net_id = torch.full((B, T), -1, dtype=torch.long, device=device)
    else:
        state.net_id = torch.full((T,), -1, dtype=torch.long, device=device)
        

    return state


def connect_terminals_batch(
    electronics_states: "ElectronicsStateTC",
    terminal_pairs: torch.Tensor,
) -> "ElectronicsStateTC":
    state = electronics_states
    assert state.terminal_to_component.numel() > 0
    device = state.device
    T = int(state.terminal_to_component.numel())

    net = state.net_id
    # Canonicalize shapes: net -> [B, T], terminal_pairs -> [B, K, 2]
    orig_net_was_1d = net.ndim == 1
    if orig_net_was_1d:
        net = net.unsqueeze(0)
    B = net.shape[0]

    if terminal_pairs.ndim == 2:
        terminal_pairs = terminal_pairs.unsqueeze(0).expand(B, -1, -1)
    elif terminal_pairs.ndim != 3:
        raise AssertionError("terminal_pairs must be [K,2] or [B,K,2]")

    # Process batch uniformly
    for b in range(B):
        nb = net[b]
        for p in terminal_pairs[b]:
            t1, t2 = int(p[0].item()), int(p[1].item())
            assert 0 <= t1 < T and 0 <= t2 < T
            g1 = int(nb[t1].item())
            g2 = int(nb[t2].item())
            candidates = [c for c in (g1, g2, t1, t2) if c >= 0] or [min(t1, t2)]
            gid = int(min(candidates))
            if g1 >= 0:
                nb[nb == g1] = gid
            if g2 >= 0 and g2 != g1:
                nb[nb == g2] = gid
            nb[t1] = gid
            nb[t2] = gid

    # Restore original dimensionality for single-env states
    state.net_id = (net[0] if orig_net_was_1d else net).to(device)
    return state
