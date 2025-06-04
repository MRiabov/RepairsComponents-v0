"""
Holds state of assembly components with batched PyTorch tensor support:
- Fastener connections: bodies attached per fastener
- Rigid bodies: batched absolute positions & rotations

Provides diff methods:
- _fastener_diff: connection changes per fastener
- _body_diff: transform changes per body

diff(): combines both into {'fasteners', 'bodies'} with total change count
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch_geometric.data import Batch, Data
from repairs_components.geometry.fasteners import Fastener
import math


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

    def to_batch(self):
        return Batch.from_data_list([self.graph])

    def _fastener_diff(
        self, other: "PhysicalState"
    ) -> tuple[dict[str, dict[str, list[str]]], int]:
        """Compute per-fastener connection differences.

        Returns:
            Tuple of:
            - conn_diff: mapping fastener name -> {'added', 'removed'} lists
            - total_changes: count of all connection changes
        """
        conn_diff: dict[str, dict[str, list[str]]] = {}
        total_changes = 0

        # If either graph is not initialized, return empty diff
        if (
            self.body_and_fastener_graph is None
            or other.body_and_fastener_graph is None
        ):
            return conn_diff, total_changes

        # Get all fastener IDs
        all_fastener_ids = set()
        if self.body_and_fastener_graph.edge_attr.numel() > 0:
            all_fastener_ids.update(
                self.body_and_fastener_graph.edge_attr.squeeze().tolist()
            )
        if other.body_and_fastener_graph.edge_attr.numel() > 0:
            all_fastener_ids.update(
                other.body_and_fastener_graph.edge_attr.squeeze().tolist()
            )

        # Get reverse mapping from index to name
        idx_to_name = {idx: name for name, idx in self.indices.items()}

        for fastener_id in all_fastener_ids:
            if fastener_id not in idx_to_name:
                continue  # Skip invalid fastener IDs

            fastener_name = idx_to_name[fastener_id]

            # Get connected bodies in self and other
            self_edges = self.body_and_fastener_graph.edge_index[
                :,
                (
                    self.body_and_fastener_graph.edge_attr.squeeze() == fastener_id
                ).nonzero(as_tuple=True)[0],
            ]
            other_edges = other.body_and_fastener_graph.edge_index[
                :,
                (
                    other.body_and_fastener_graph.edge_attr.squeeze() == fastener_id
                ).nonzero(as_tuple=True)[0],
            ]

            # Get unique connected nodes
            self_connected = (
                set(self_edges[0].tolist() + self_edges[1].tolist())
                if self_edges.numel() > 0
                else set()
            )
            other_connected = (
                set(other_edges[0].tolist() + other_edges[1].tolist())
                if other_edges.numel() > 0
                else set()
            )

            # Convert node indices to names
            self_connected_names = [
                idx_to_name[idx] for idx in self_connected if idx in idx_to_name
            ]
            other_connected_names = [
                idx_to_name[idx] for idx in other_connected if idx in idx_to_name
            ]

            # Calculate differences
            added = list(set(other_connected_names) - set(self_connected_names))
            removed = list(set(self_connected_names) - set(other_connected_names))

            if added or removed:
                conn_diff[fastener_name] = {"added": added, "removed": removed}
                total_changes += len(added) + len(removed)

        return conn_diff, total_changes

    def _body_diff(
        self,
        other: "PhysicalState",
        deg_threshold: float = 5.0,
        pos_threshold: float = 3.0,
    ) -> tuple[dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]], int]:
        """Compute differences in rigid body positions and rotations.

        Ignores position changes below pos_threshold units and rotation changes below deg_threshold degrees.

        Args:
            other: state to compare against.
            deg_threshold: angle threshold in degrees.
            pos_threshold: positional threshold.

        Returns:
            Tuple of (body_diff, total_changes).
            body_diff[name] = {'position': (pos, other_pos), 'rotation': (rot, other_rot)}
        """
        body_diff: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        total_changes = 0
        for name, pos in self.positions.items():
            other_pos = other.positions.get(name)
            other_rot = other.rotations.get(name)
            changes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            # position diff
            if other_pos is not None:
                pos_tensor = (
                    pos
                    if isinstance(pos, torch.Tensor)
                    else torch.tensor(pos, device=self.device)
                )
                other_pos_tensor = (
                    other_pos
                    if isinstance(other_pos, torch.Tensor)
                    else torch.tensor(other_pos, device=self.device)
                )

                if torch.dist(pos_tensor, other_pos_tensor) > float(pos_threshold):
                    changes["position"] = (pos_tensor, other_pos_tensor)
                    total_changes += 1

            # rotation diff
            rot = self.rotations.get(name)
            if rot is not None and other_rot is not None:
                rot_tensor = (
                    rot
                    if isinstance(rot, torch.Tensor)
                    else torch.tensor(rot, device=self.device)
                )
                other_rot_tensor = (
                    other_rot
                    if isinstance(other_rot, torch.Tensor)
                    else torch.tensor(other_rot, device=self.device)
                )

                # quaternion angle diff
                dot = torch.abs(torch.sum(rot_tensor * other_rot_tensor))
                angle = 2 * math.degrees(math.acos(min(1.0, max(-1.0, float(dot)))))
                if angle > deg_threshold:
                    changes["rotation"] = (rot_tensor, other_rot_tensor)
                    total_changes += 1

            if changes:
                body_diff[name] = changes
        return body_diff, total_changes

    def diff(self, other: "PhysicalState") -> tuple[dict[str, dict], int]:
        """Compute combined fastener and rigid body differences and total changes."""
        fast_diff, fast_changes = self._fastener_diff(other)
        body_diff, body_changes = self._body_diff(other)
        return {
            "fasteners": fast_diff,
            "bodies": body_diff,
        }, fast_changes + body_changes
