from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch_geometric.data import Data
from repairs_components.logic.electronics.component import ElectricalComponent
from repairs_components.training_utils.sim_state import SimState
from repairs_components.logic.physical_state import _diff_edge_features


@dataclass
class ElectronicsState(SimState):
    components: dict[str, ElectricalComponent] = field(default_factory=dict)
    graph: Data = field(default_factory=Data)
    indices: dict[str, int] = field(default_factory=dict)
    _graph_built: bool = field(default=False, init=False)

    def __post_init__(self):
        # Initialize sparse graph representation
        self.indices = {}
        self.graph = Data()
        self.graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.graph.num_nodes = 0
        self._graph_built = False

    def _build_graph(self):
        if self._graph_built:
            return
        # Build index mapping
        self.indices = {name: idx for idx, name in enumerate(self.components.keys())}
        num_nodes = len(self.indices)
        # Gather edges
        edge_list: list[tuple[int, int]] = []
        for name, comp in self.components.items():
            src = self.indices[name]
            for other in comp.connected_to:
                dst = self.indices[other.name]
                edge_list.append((src, dst))
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data()
        data.edge_index = edge_index
        data.num_nodes = num_nodes
        self.graph = data
        self._graph_built = True

    def diff(
        self, other: "ElectronicsState"
    ) -> tuple[Dict[str, Dict[str, List[str]]], int]:
        """Compute per-component connection differences and total change count.

        Args:
            other (ElectronicsState): state to compare against.

        Returns:
            - connection_differences: mapping component names to {'added', 'removed'} lists
            - total_changes: int count of all added+removed connections
        """
        # Ensure graphs are built
        self._build_graph()
        other._build_graph()

        edge_diff, total_changes = _diff_edge_features(self.graph, other.graph)
        diff_map: Dict[str, Dict[str, List[str]]] = {}
        # Reverse index mapping
        index_to_name = {idx: name for name, idx in self.indices.items()}
        for src, dst in edge_diff["added"]:
            comp_name = index_to_name[src]
            diff_map.setdefault(comp_name, {"added": [], "removed": []})["added"].append(index_to_name[dst])
        for src, dst in edge_diff["removed"]:
            comp_name = index_to_name[src]
            diff_map.setdefault(comp_name, {"added": [], "removed": []})["removed"].append(index_to_name[dst])
        return diff_map, total_changes

    def register(self, component: ElectricalComponent):
        """Register a new electrical component."""
        assert component.name not in self.components, f"Component {component.name} already registered"
        self.components[component.name] = component
        self._graph_built = False

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

