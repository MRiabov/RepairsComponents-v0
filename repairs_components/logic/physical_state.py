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
    reverse_indices: dict[int, str] = field(default_factory=dict)

    # Fastener metadata (shared across batch)
    fastener_prototype: dict[str, Fastener] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize graph attributes
        self.graph.edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=self.device
        )
        self.graph.edge_attr = torch.empty(
            (0, 12), dtype=torch.float32, device=self.device
        )  # placeholder size

        # Node attributes
        self.graph.position = torch.empty(
            (0, 3), dtype=torch.float32, device=self.device
        )  # note: torch_geomeric conventionally uses `pos` for 2d or 3d positions. You could too.
        self.graph.quat = torch.empty((0, 4), dtype=torch.float32, device=self.device)

    def register_body(self, name: str, position: tuple, rotation: tuple):
        assert name not in self.indices, f"Body {name} already registered"
        # assert position.shape == (3,) and rotation.shape == (4,) # was a tensor.
        assert len(position) == 3 and len(rotation) == 3, (
            f"Position must be 3D vector, got {position}"
            f"Rotation must be 4D vector, got {rotation}"
        )
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_euler("xyz", rotation).as_quat()

        idx = len(self.indices)
        self.indices[name] = idx
        self.reverse_indices[idx] = name

        self.graph.position = torch.cat(
            [
                self.graph.position,
                torch.tensor(position, device=self.device).unsqueeze(0),
            ],
            dim=0,
        )
        self.graph.quat = torch.cat(
            [self.graph.quat, torch.tensor(rotation, device=self.device).unsqueeze(0)],
            dim=0,
        )
        # set num_nodes manually because otherwise there is no way for PyG to know the number of nodes.
        self.graph.num_nodes = len(self.indices)

    def register_fastener(self, name: str, fastener: Fastener):
        self.fastener_prototype[name] = fastener
        # FIXME should automatically connect if there are part labels

    def connect(self, fastener_name: str, body_a: str, body_b: str):
        src = self.indices[body_a]
        dst = self.indices[body_b]

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

    def diff(self, other: "PhysicalState") -> tuple[Data, int]:
        """Compute a graph diff between two physical states.

        Returns:
            tuple[Data, int]: A tuple containing:
                - A PyG Data object representing the diff with:
                    - x: Node features [num_nodes, node_feature_dim]
                    - edge_index: Edge connections [2, num_edges]
                    - edge_attr: Edge features [num_edges, edge_feature_dim]
                    - node_mask: Boolean mask of changed nodes [num_nodes]
                    - edge_mask: Boolean mask of changed edges [num_edges]
                    - num_nodes: Total number of nodes
                - An integer count of the total number of differences
        """
        assert self.graph.num_nodes > 0, "Graph must not be empty."
        assert other.graph.num_nodes > 0, "Compared graph must not be empty."
        assert self.graph.num_nodes == other.graph.num_nodes, (
            "Graphs must have the same number of nodes."
        )
        # Get node differences
        node_diff, node_diff_count = _diff_node_features(self.graph, other.graph)

        # Get edge differences
        edge_diff, edge_diff_count = _diff_edge_features(self.graph, other.graph)

        # Calculate total differences
        total_diff_count = node_diff_count + edge_diff_count

        # Create diff graph with same nodes as original
        diff_graph = Data()
        num_nodes = max(
            self.graph.num_nodes if hasattr(self.graph, "num_nodes") else 0,
            other.graph.num_nodes if hasattr(other.graph, "num_nodes") else 0,
        )

        # Node features
        diff_graph.position = torch.zeros((num_nodes, 3), device=self.device)
        diff_graph.quat = torch.zeros((num_nodes, 4), device=self.device)
        diff_graph.node_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device
        )

        if "changed_indices" in node_diff and len(node_diff["changed_indices"]) > 0:
            changed_indices = node_diff["changed_indices"].to(self.device)
            diff_graph.position[changed_indices] = (
                node_diff["pos_diff"].to(self.device).unsqueeze(-1)
            )

            # Create identity quaternions for rotation differences
            rot_diffs = node_diff["rot_diff"].to(self.device)
            identity_quats = torch.zeros((len(changed_indices), 4), device=self.device)
            identity_quats[:, 0] = 1.0  # w=1, x=y=z=0 for identity quaternion
            diff_graph.quat[changed_indices] = (
                identity_quats  # Store rotation differences as quaternions
            )

            diff_graph.node_mask[changed_indices] = True

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

        return diff_graph, int(total_diff_count)

    def diff_to_dict(self, diff_graph: Data) -> dict:
        """Convert the graph diff to a human-readable dictionary format.

        Args:
            diff_graph: The graph diff returned by diff()

        Returns:
            dict: A dictionary with 'nodes' and 'edges' keys containing
                  human-readable diff information
        """
        result = {
            "nodes": {
                "changed_indices": diff_graph.node_mask.nonzero().squeeze(-1).tolist(),
                "position_diffs": diff_graph.position.tolist(),
                "quaternion_diffs": diff_graph.quat.tolist(),
            },
            "edges": {
                "added": diff_graph.edge_index[:, diff_graph.edge_attr[:, 0].bool()]
                .t()
                .tolist(),
                "removed": diff_graph.edge_index[:, diff_graph.edge_attr[:, 1].bool()]
                .t()
                .tolist(),
            },
        }
        return result

    def diff_to_str(self, diff_graph: Data) -> str:
        """Convert the graph diff to a human-readable string.

        Args:
            diff_graph: The graph diff returned by diff()

        Returns:
            str: A formatted string describing the diff
        """
        diff_dict = self.diff_to_dict(diff_graph)
        lines = ["Physical State Diff:", "=" * 50]

        # Node changes
        changed_nodes = diff_dict["nodes"]["changed_indices"]
        if changed_nodes:
            lines.append("\nNode Changes:")
            for i, idx in enumerate(changed_nodes):
                pos_diff = diff_dict["nodes"]["position_diffs"][i]
                quat_diff = diff_dict["nodes"]["quaternion_diffs"][i]
                lines.append(f"  Node {idx}:")
                lines.append(f"    Position diff: {pos_diff}")
                lines.append(f"    Quaternion diff: {quat_diff}")

        # Edge changes
        added_edges = diff_dict["edges"]["added"]
        removed_edges = diff_dict["edges"]["removed"]

        if added_edges:
            lines.append("\nAdded Edges:")
            for src, dst in added_edges:
                lines.append(f"  {src} -> {dst}")

        if removed_edges:
            lines.append("\nRemoved Edges:")
            for src, dst in removed_edges:
                lines.append(f"  {src} -> {dst}")

        if not (changed_nodes or added_edges or removed_edges):
            lines.append("No changes detected.")

        return "\n".join(lines)

    def check_if_fastener_inserted(self, body_id: int, fastener_id: int) -> bool:
        """Check if a body (body_id) is still connected to a fastener (fastener_id)."""
        edge_index = self.graph.edge_index
        # No edges means no connection
        if edge_index.numel() == 0:
            return False
        src, dst = edge_index
        # Check for edge in either direction
        connected_direct = ((src == body_id) & (dst == fastener_id)).any()
        connected_reverse = ((src == fastener_id) & (dst == body_id)).any()
        return bool(connected_direct or connected_reverse)

    def check_if_part_placed(self, part_id: int, data_a: Data, data_b: Data) -> bool:
        """Check if a part (part_id) is in its desired position (within thresholds)."""
        # Compute node feature diffs (position & orientation) between data_a and data_b
        diff_dict, _ = _diff_node_features(data_a, data_b)
        changed = diff_dict["changed_indices"]
        # Return True if this part_id did not exceed thresholds

        #FIXME: this does not account for a) batch of environments, b) batch of parts.
        # ^batch of environments would be cool, but batch of parts would barely ever happen.
        return (changed == part_id).any() 


def _quaternion_angle_diff(q1, q2):
    # q1, q2: [N, 4]
    dot = torch.sum(q1 * q2, dim=1).clamp(-1.0, 1.0).abs()
    return 2 * torch.acos(dot).rad2deg()


def _diff_node_features(
    data_a: Data, data_b: Data, pos_threshold=3.0, deg_threshold=5.0
):
    assert data_a.position is not None and data_b.position is not None, (
        "Both graphs must have positions."
    )
    pos_diff = torch.norm(data_a.position - data_b.position, dim=1)
    rot_diff = _quaternion_angle_diff(data_a.quat, data_b.quat)

    pos_mask = pos_diff > pos_threshold
    rot_mask = rot_diff > deg_threshold

    # Collect indices where there are differences
    changes = torch.nonzero(pos_mask | rot_mask, as_tuple=False).squeeze(dim=0)
    # Note^ I did not check whether dim=0 or dim=1, maybe debug here. (it was flat squeeze.)
    return {
        "changed_indices": changes,
        "pos_diff": pos_diff[pos_mask],
        "rot_diff": rot_diff[rot_mask],
    }, pos_mask.sum().item() + rot_mask.sum().item()


def _diff_edge_features(data_a: Data, data_b: Data) -> tuple[dict, int]:
    # FIXME: does this not check for feature value differences?
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

    return {"added": added, "removed": removed}, len(added) + len(removed)


# def batch_graph_diff(batch_a: Batch, batch_b: Batch):
#     assert batch_a.batch_size == batch_b.batch_size
#     results = torch.zeros(batch_a.batch_size)
#     for i in range(batch_a.num_graphs):
#         data_a = batch_a.get_example(i)
#         data_b = batch_b.get_example(i)
#         results[i] = diff(data_a, data_b)[1]
#     return results


