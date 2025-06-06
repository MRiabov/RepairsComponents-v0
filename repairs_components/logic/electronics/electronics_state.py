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
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

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

    def diff(self, other: "ElectronicsState") -> Data:
        """Compute a graph diff between two electronic states.
        
        Returns:
            Data: A PyG Data object representing the diff with:
                - x: Node features [num_nodes, 1] (1 if node exists in both states)
                - edge_index: Edge connections [2, num_edges]
                - edge_attr: Edge features [num_edges, 3] (is_added, is_removed, is_changed)
                - node_mask: Boolean mask of nodes that exist in either state [num_nodes]
                - edge_mask: Boolean mask of changed edges [num_edges]
                - num_nodes: Total number of nodes
        """
        # Ensure graphs are built
        self._build_graph()
        other._build_graph()
        
        # Get edge differences
        edge_diff, _ = _diff_edge_features(self.graph, other.graph)
        
        # Create diff graph with same nodes as original
        diff_graph = Data()
        num_nodes = max(
            self.graph.num_nodes if hasattr(self.graph, 'num_nodes') else 0,
            other.graph.num_nodes if hasattr(other.graph, 'num_nodes') else 0
        )
        
        # Node features: 1 if node exists in both states, 0 otherwise
        node_exists = torch.zeros((num_nodes, 1), device=self.device)
        if hasattr(self.graph, 'num_nodes') and hasattr(other.graph, 'num_nodes'):
            common_nodes = min(self.graph.num_nodes, other.graph.num_nodes)
            if common_nodes > 0:
                node_exists[:common_nodes] = 1
        
        # Node mask: True if node exists in either state
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        if hasattr(self.graph, 'num_nodes') and self.graph.num_nodes > 0:
            node_mask[:self.graph.num_nodes] = True
        if hasattr(other.graph, 'num_nodes') and other.graph.num_nodes > 0:
            node_mask[:other.graph.num_nodes] = True
        
        diff_graph.x = node_exists
        diff_graph.node_mask = node_mask
        
        # Edge features and mask
        edge_index = torch.tensor([], dtype=torch.long, device=self.device).reshape(2, 0)
        edge_attr = torch.tensor([], device=self.device).reshape(0, 3)  # [is_added, is_removed, is_changed]
        edge_mask = torch.tensor([], dtype=torch.bool, device=self.device)
        
        if edge_diff['added'] or edge_diff['removed']:
            added_edges = torch.tensor(edge_diff['added'], device=self.device).t() if edge_diff['added'] else torch.tensor([], device=self.device).long().reshape(2, 0)
            removed_edges = torch.tensor(edge_diff['removed'], device=self.device).t() if edge_diff['removed'] else torch.tensor([], device=self.device).long().reshape(2, 0)
            
            # Combine all edges
            edge_index = torch.cat([added_edges, removed_edges], dim=1)
            
            # Create edge features
            added_attrs = torch.zeros((added_edges.size(1), 3), device=self.device)
            added_attrs[:, 0] = 1  # is_added
            added_attrs[:, 2] = 1   # is_changed
            
            removed_attrs = torch.zeros((removed_edges.size(1), 3), device=self.device)
            removed_attrs[:, 1] = 1  # is_removed
            removed_attrs[:, 2] = 1   # is_changed
            
            edge_attr = torch.cat([added_attrs, removed_attrs], dim=0)
            edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=self.device)
        
        diff_graph.edge_index = edge_index
        diff_graph.edge_attr = edge_attr
        diff_graph.edge_mask = edge_mask
        diff_graph.num_nodes = num_nodes
        
        return diff_graph
        
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
                (index_to_name.get(src.item(), str(src.item())), 
                 index_to_name.get(dst.item(), str(dst.item())))
                for src, dst in edge_index[:, added_mask].t()
            ]
            
            removed_edges = [
                (index_to_name.get(src.item(), str(src.item())), 
                 index_to_name.get(dst.item(), str(dst.item())))
                for src, dst in edge_index[:, removed_mask].t()
            ]
        
        return {
            'nodes': {
                'existing': [index_to_name.get(i, str(i)) 
                           for i in range(diff_graph.x.size(0)) 
                           if diff_graph.x[i].item() > 0]
            },
            'edges': {
                'added': added_edges,
                'removed': removed_edges
            }
        }
        
    def diff_to_str(self, diff_graph: Data) -> str:
        """Convert the graph diff to a human-readable string.
        
        Args:
            diff_graph: The graph diff returned by diff()
            
        Returns:
            str: A formatted string describing the diff
        """
        diff_dict = self.diff_to_dict(diff_graph)
        lines = ["Electronics State Diff:", "="*50]
        
        # Edge changes
        added_edges = diff_dict['edges']['added']
        removed_edges = diff_dict['edges']['removed']
        
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

    # def batch_diff(self, other: "ElectronicsState")
