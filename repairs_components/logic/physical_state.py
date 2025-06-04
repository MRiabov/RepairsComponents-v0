"""
Holds state of assembly components:
- Fastener connections: bodies attached per fastener
- Rigid bodies: absolute positions & rotations

Provides diff methods:
- _fastener_diff: connection changes per fastener
- _body_diff: transform changes per body

diff(): combines both into {'fasteners', 'bodies'} with total change count
"""

from dataclasses import dataclass, field

import torch
from repairs_components.geometry.fasteners import Fastener
import math


from dataclasses import dataclass, field
import torch
from torch_geometric.data import Data, Batch


@dataclass
class PhysicalState:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Graph storing part nodes and fastener edges
    graph: Data = field(default_factory=Data)
    indices: dict[str, int] = field(default_factory=dict)

    # Fastener metadata (shared across batch)
    fastener_prototype: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        self.graph.x = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )  # pos(3) + quat(4)
        self.graph.edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=self.device
        )
        self.graph.edge_attr = torch.empty(
            (0, 12), dtype=torch.float32, device=self.device
        )  # placeholder size

    def register_body(self, name: str, position: torch.Tensor, rotation: torch.Tensor):
        assert name not in self.indices, f"Body {name} already registered"
        assert position.shape == (3,) and rotation.shape == (4,)

        idx = len(self.indices)
        self.indices[name] = idx

        node_feature = torch.cat([position, rotation]).unsqueeze(0).to(self.device)
        self.graph.x = torch.cat([self.graph.x, node_feature], dim=0)

    def register_fastener(self, name: str, attr: dict[str, torch.Tensor]):
        self.fastener_prototype[name] = {k: v.to(self.device) for k, v in attr.items()}

    def connect(self, fastener_name: str, body_a: str, body_b: str):
        src, dst = self.indices[body_a], self.indices[body_b]

        # Undirected: add both directions
        new_edges = torch.tensor(
            [[src, dst], [dst, src]], dtype=torch.long, device=self.device
        )

        # Construct edge features from prototype fastener
        base_attr = self.fastener_prototype[fastener_name]
        attr_vec = torch.cat([v.view(-1) for v in base_attr.values()])
        edge_attr = attr_vec.repeat(2, 1).to(self.device)

        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)
        self.graph.edge_attr = torch.cat([self.graph.edge_attr, edge_attr], dim=0)

    def disconnect(self, fastener_name: str, disconnected_body: str):
        fastener_id = self.indices[fastener_name]
        body_id = self.indices[disconnected_body]

        # Find edge indices to remove (brute-force based on node match only)
        edge_index = self.graph.edge_index
        src_mask = edge_index[0] == body_id
        dst_mask = edge_index[1] == body_id
        match_mask = src_mask | dst_mask

        keep_mask = ~match_mask
        self.graph.edge_index = edge_index[:, keep_mask]
        self.graph.edge_attr = self.graph.edge_attr[keep_mask]

    def diff(self, other: "PhysicalState") -> tuple[dict, int]:
        edge_diff, edge_features_diff_count = _diff_edge_features(
            self.graph, other.graph
        )
        node_diff, node_features_diff_count = _diff_node_features(
            self.graph, other.graph
        )

        return {
            "nodes": node_diff,
            "edges": edge_diff,
        }, int(edge_features_diff_count + node_features_diff_count)


def _quaternion_angle_diff(q1, q2):
    # q1, q2: [N, 4]
    dot = torch.sum(q1 * q2, dim=1).clamp(-1.0, 1.0).abs()
    return 2 * torch.acos(dot).rad2deg()


def _diff_node_features(data_a, data_b, pos_threshold=3.0, deg_threshold=5.0):
    pos_diff = torch.norm(data_a.pos - data_b.pos, dim=1)
    rot_diff = _quaternion_angle_diff(data_a.quat, data_b.quat)

    pos_mask = pos_diff > pos_threshold
    rot_mask = rot_diff > deg_threshold

    # Collect indices where there are differences
    changes = torch.nonzero(pos_mask | rot_mask, as_tuple=False).squeeze()
    return {
        "changed_indices": changes,
        "pos_diff": pos_diff[pos_mask],
        "rot_diff": rot_diff[rot_mask],
    }, pos_mask.sum().item() + rot_mask.sum().item()


def _diff_edge_features(data_a: Data, data_b: Data) -> tuple[dict, int]:
    # Stack edge pairs for easy comparison
    def to_sorted_tuple_tensor(edge_index):
        sorted_idx = edge_index.sort(dim=0)[0]
        return sorted_idx.t()

    edges_a = to_sorted_tuple_tensor(data_a.edge_index)
    edges_b = to_sorted_tuple_tensor(data_b.edge_index)

    a_set = set(map(tuple, edges_a.tolist()))
    b_set = set(map(tuple, edges_b.tolist()))

    added = list(b_set - a_set)  # well, this is sloppy, but let it be.
    removed = list(a_set - b_set)

    return {
        "added": added,
        "removed": removed,
    }, len(added) + len(removed)



def batch_graph_diff(batch_a: Batch, batch_b: Batch):
    assert batch_a.batch_size == batch_b.batch_size
    results = torch.zeros(batch_a.batch_size)
    for i in range(batch_a.num_graphs):
        data_a = batch_a.get_example(i)
        data_b = batch_b.get_example(i)
        results[i] = graph_diff(data_a, data_b)[1]
    return results