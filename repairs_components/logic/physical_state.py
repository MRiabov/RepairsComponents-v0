"""
Holds state of assembly components:
- Fastener connections: bodies attached per fastener
- Rigid bodies: absolute positions & rotations

Provides diff methods:
- _fastener_diff: connection changes per fastener
- _body_diff: transform changes per body

diff(): combines both into {'fasteners', 'bodies'} with total change count
"""

import torch
from repairs_components.geometry.fasteners import (
    Fastener,
    get_fastener_params_from_name,
)
from torch_geometric.data import Data
from dataclasses import dataclass, field
from tensordict import TensorClass
from repairs_components.processing.geom_utils import (
    are_quats_within_angle,
    euler_deg_to_quat_wxyz,
    get_connector_pos,
    quaternion_delta,
    sanitize_quaternion,
)
from typing_extensions import deprecated


@dataclass
class PhysicalStateInfo:
    """A dataclass holding all singleton values for PhysicalState, e.g. body indices, fixed,
    starting holes, fasteners diam/length, hole metadata (positions, quats, depth, through, diameter), etc.
    """

    # --- bodies ---

    fixed: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.bool)
    )
    # body_indices and inverse_body_indices
    body_indices: dict[str, int] = field(default_factory=dict)
    inverse_body_indices: dict[int, str] = field(default_factory=dict)
    permanently_constrained_parts: list[list[str]] = field(default_factory=list)
    """List of lists of permanently constrained parts (linked_groups from EnvSetup)"""

    # --- assets ---  # TODO: move to a separate persistence state info.
    mesh_file_names: dict[str, str] = field(default_factory=dict)
    """Per-scene mapping from body/fastener names to asset file paths (mesh/MJCF)."""

    # --- terminals --- # maybe this should be connectors.
    terminal_indices_from_name: dict[str, int] = field(default_factory=dict)
    """Terminal indices per connector name."""

    male_terminal_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Male terminal indices per part in the batch (physical part, not electronics component!)"""
    female_terminal_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Female terminal indices per part in the batch (physical part, not electronics component!)"""

    # --- fasteners ---
    """Fixed body flags (not fasteners)"""
    fasteners_diam: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Fastener diameters in meters [num_fasteners]"""
    fasteners_length: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Fastener lengths in meters [num_fasteners]"""

    # --- holes ---
    starting_hole_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Starting hole positions for every part, batched with part_hole_batch. 
    Equal over the batch. Shape: (H, 3)"""
    starting_hole_quats: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 4), dtype=torch.float32)
    )
    """Starting hole quats for every part, batched with part_hole_batch. 
    Equal over the batch. Shape: (H, 4)"""
    hole_depth: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Hole depths for every part.
    Equal over the batch. Shape: (H)"""
    hole_is_through: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.bool)
    )
    """Boolean mask indicating whether each hole is through (True) or blind (False).
    Equal over the batch. Shape: (H)"""
    hole_diameter: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Hole diameters in meters.
    Equal over the batch. Shape: (H)"""
    part_hole_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    "Part index for every hole. Equal over the batch. Shape: (H)"
    # hole_indices_from_name: dict[str, int] = field(default_factory=dict)
    # """Hole indices per part name.""" # unused.

    env_size: torch.Tensor = torch.tensor([640, 640, 640], dtype=torch.float)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        # move all to device
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(self.device))
        # NOTE: don't know if I want to continue spamming myself with more assertions, but this is from ConcurrentSceneDataclass.
        # holes
        self.hole_count = self.starting_hole_positions.shape[0]
        assert self.starting_hole_positions.shape == (self.hole_count, 3), (
            f"Starting hole positions must have shape ({self.hole_count}, 3), but got {self.starting_hole_positions.shape}"
        )  # note the no batch dim.
        assert self.starting_hole_quats.shape == (self.hole_count, 4), (
            f"Starting hole quats must have shape ({self.hole_count}, 4), but got {self.starting_hole_quats.shape}"
        )
        assert self.part_hole_batch.shape == (self.hole_count,), (
            "Part hole batch must have shape (H,)"
        )
        assert self.hole_depth.shape == (self.hole_count,), (
            "Hole depths must have shape (H,)"
        )
        assert self.hole_is_through.shape == (self.hole_count,), (
            "hole_is_through must have shape (H,)"
        )
        assert self.hole_diameter.shape == (self.hole_count,), (
            "hole_diameter must have shape (H,)"
        )
        assert (self.hole_depth > 0).all(), "Hole depths must be positive."
        assert self.hole_is_through.shape == (self.hole_count,), (
            "Hole is through must have shape (H,)"
        )


class PhysicalState(TensorClass):
    """A dataclass holding mechanical state. Holds all values across the batch. Interdependent with PhysicalStateInfo for singleton values.

    NOTE: PhysicalState will always have to be instantiated with either PhysicalState(device=device).unsqueeze(0)
    or torch.stack([PhysicalState(device=device)]*B). This is because they are expected to be batched with a leading dimension."""

    # TODO: batch size field?
    position: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Solids positions (not fasteners)"""
    quat: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 4), dtype=torch.float32)
    )
    """Solids quaternions (not fasteners)"""
    count_fasteners_held: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.int8)
    )
    """Count of fasteners held by each body (not fasteners)"""
    male_terminal_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Male terminal positions per part, batched with male_terminal_batch (physical part, not electronics component!)"""
    female_terminal_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Female terminal positions per part in current env, batched with female_terminal_batch (physical part, not electronics component!)"""

    # # Edge attributes (previously in graph)
    # edge_index: torch.Tensor = field(
    #     default_factory=lambda: torch.empty((2, 0), dtype=torch.long)
    # )
    # """Edge connections (fastener connections) [2, num_edges]"""
    # edge_attr: torch.Tensor = field(
    #     default_factory=lambda: torch.empty((0, 12), dtype=torch.float32)
    # ) # 10.8 just commented out - they are obsolete, use fasteners_attached_to_body instead.
    # """Edge attributes (fastener connections) [num_edges, edge_feature_dim]
    # Includes:
    # - fastener diameter (1)
    # - fastener length (1)
    # - fastener position (3)
    # - fastener quaternion (4)
    # - is_connected_a/b (2)

    # """
    # FIXME: where does 12 come from? Probably valid, because I copied it from previous init, but I don't remember now.
    # possibly: xyz(3)+quat(4)+connected_to_1(1)+connected_to_2(1)... what else?
    # note: edge_attr is not used for learning. Use export_graph() instead.

    # Fastener attributes
    fasteners_pos: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Fastener positions [num_fasteners, 3]"""
    fasteners_quat: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 4), dtype=torch.float32)
    )
    """Fastener quaternions [num_fasteners, 4]"""
    fasteners_attached_to_body: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.int8)
    )
    """Which bodies fasteners are attached to [num_fasteners, 2] (-1 for unattached)"""
    fasteners_attached_to_hole: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.int8)
    )
    """Which holes fasteners are attached to [num_fasteners, 2] (-1 for unattached)"""

    # TODO encode mass, and possibly velocity.
    # note: fasteners_inserted_into_holes is not meant to be exported. for internal ref in screw in logic only.

    """Hole indices per part in the batch."""
    # FIXME: part_hole_batch is a duplication from ConcurrentSimState
    hole_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3))
    )  # [H, 3]
    """Hole positions per part, batched with hole_indices_batch."""
    hole_quats: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 4))
    )  # [H, 4]
    """Hole quats per part, batched with hole_indices_batch."""

    # next: there is something that needs to be figured out with data storage and reconstruction.
    # So 1. I do save STL/gltf files,
    # but during offline I don't want to scan meshes because that's costly.
    # I only want to scan physical and electrical states, and load meshes to genesis only once. build123d should not be used at all.
    # Consequently, I need to store meshes, and their positions for genesis to read.
    # this is stored in the graph, however fasteners are stored in the graph too.
    # so I need an extra container for free fasteners.
    # however what's critical is to ensure no disparity between offline and online loads.
    # It must be fully deterministic, which it currently is not due to fastener connections not specifying A or B.
    # wait, do I not register free fasteners in the graph at all?

    def export_graph(self, physical_info: PhysicalStateInfo):
        """Export the graph to a torch_geometric Data object usable by ML."""

        # only export fasteners which aren not attached to nothing.
        global_feat_mask = (self.fasteners_attached_to_body == -1).any(dim=-1)
        global_feat_export = torch.cat(
            [
                self.fasteners_pos[global_feat_mask],
                self.fasteners_quat[global_feat_mask],
                (self.fasteners_attached_to_body[global_feat_mask] == -1).float(),
                # export which are attached and which not, but not their ids.
            ],
            dim=-1,
        )
        edge_attr, edge_index = self._build_fastener_edge_attr_and_index(physical_info)
        # how would I handle batchsizes in all of this? e.g. here logic would need to cat differently based on batch dim present or not.

        graph = Data(  # expected len of x - 8.
            x=torch.cat(
                [
                    self.position,
                    self.quat,
                    self.count_fasteners_held.float().unsqueeze(-1),
                    # TODO: construct count_fasteners_held on export.
                ],
                dim=1,
            ).bfloat16(),
            edge_index=edge_index,
            edge_attr=edge_attr,  # e.g. fastener size.
            num_nodes=len(physical_info.body_indices),
            global_feat=global_feat_export,
        )
        # print("debug: graph global feat shape", graph.global_feat.shape)
        return graph

    def diff(
        self, other: "PhysicalState", physical_info: PhysicalStateInfo
    ) -> tuple[Data, int]:
        """Compute a graph diff between two physical states. As always, physical_info belongs to both states.

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
        # note: passing in PhysicalInfo is deliberate: it will diff hole-fastener pairs and not ids. Now, it could diff ids indirectly, but this does not account for equality in fasteners.
        assert self.position.shape[1] > 0, "Physical state must not be empty."
        assert other.position.shape[1] > 0, "Compared physical state must not be empty."
        assert self.position.shape[1] == other.position.shape[1], (
            "Compared physical states must have equal number of bodies."
        )
        # assert physical_info belongs to both states
        assert len(physical_info.body_indices) == self.position.shape[1]
        assert len(physical_info.body_indices) == other.position.shape[1]

        # Get node and edge differences
        body_diff, body_diff_count = _diff_body_features(self, other)
        fastener_diff, fastener_diff_count = _diff_fastener_features(
            self, other, physical_info
        )
        total_diff_count = body_diff_count + fastener_diff_count

        # Helper to normalize edge containers to [2, K] long tensors on device
        def to_edge_tensor(edges) -> torch.Tensor:
            if isinstance(edges, torch.Tensor):
                if edges.numel() == 0:
                    return torch.empty((2, 0), dtype=torch.long, device=self.device)
                return edges.to(self.device)
            if not edges:  # empty list
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            return torch.tensor(edges, dtype=torch.long, device=self.device).t()

        # Prepare diff graph
        diff_graph = Data()
        num_nodes = self.position.shape[1]

        # Per-node diffs
        diff_graph.position = body_diff["pos_diff"].to(self.device)
        diff_graph.quat = body_diff["quat_diff"].to(self.device)

        # count_fasteners_held diffs (supports batched [B, N])
        count_a = self.count_fasteners_held
        count_b = other.count_fasteners_held
        if count_a.ndim == 2:
            count_a = count_a[0]
        if count_b.ndim == 2:
            count_b = count_b[0]
        count_diff = (count_b.to(torch.int32) - count_a.to(torch.int32)).to(torch.int32)
        diff_graph.count_fasteners_held_diff = count_diff.to(self.device)

        # Node mask from changed indices + count diffs
        node_mask = torch.zeros((num_nodes,), dtype=torch.bool, device=self.device)
        changed_indices = body_diff["changed_indices"].to(self.device)
        if (
            changed_indices.ndim == 2 and changed_indices.size(1) == 2
        ):  # [K, 2] from nonzero
            changed_indices = changed_indices[:, 1]
        node_mask[changed_indices] = True
        node_mask |= count_diff != 0
        diff_graph.node_mask = node_mask

        # Edge tensors in order: added, removed, changed
        added_edges = to_edge_tensor(fastener_diff.get("added", []))
        removed_edges = to_edge_tensor(fastener_diff.get("removed", []))
        changed_edges = to_edge_tensor(
            fastener_diff.get("changed_edges", torch.empty((2, 0), dtype=torch.long))
        )
        edge_index = torch.cat([added_edges, removed_edges, changed_edges], dim=1)

        # Edge attributes: [is_added, is_removed, diam_changed, length_changed, pos_changed, quat_changed]
        n_added, n_removed, n_changed = (
            added_edges.size(1),
            removed_edges.size(1),
            changed_edges.size(1),
        )
        added_attrs = torch.zeros((n_added, 6), device=self.device)
        removed_attrs = torch.zeros((n_removed, 6), device=self.device)
        changed_attrs = torch.zeros((n_changed, 6), device=self.device)
        if n_added:
            added_attrs[:, 0] = 1
        if n_removed:
            removed_attrs[:, 1] = 1
        if n_changed:
            diam_changed = fastener_diff["diam_changed"].to(self.device)
            length_changed = fastener_diff["length_changed"].to(self.device)
            pos_diff = fastener_diff["pos_diff"].to(self.device)
            quat_delta = fastener_diff["quat_delta"].to(self.device)

            changed_attrs[:, 2] = diam_changed.to(torch.float32)
            changed_attrs[:, 3] = length_changed.to(torch.float32)
            changed_attrs[:, 4] = (torch.linalg.norm(pos_diff, dim=1) > 0).to(
                torch.float32
            )
            changed_attrs[:, 5] = (torch.linalg.norm(quat_delta, dim=1) > 0).to(
                torch.float32
            )

            # Attach detailed deltas aligned with the combined edge order
            diff_graph.fastener_pos_diff = (
                torch.cat(
                    [
                        torch.zeros((n_added + n_removed, 3), device=self.device),
                        pos_diff,
                    ],
                    dim=0,
                )
                if (n_added + n_removed + n_changed) > 0
                else torch.empty((0, 3), device=self.device)
            )
            diff_graph.fastener_quat_delta = (
                torch.cat(
                    [
                        torch.zeros((n_added + n_removed, 4), device=self.device),
                        quat_delta,
                    ],
                    dim=0,
                )
                if (n_added + n_removed + n_changed) > 0
                else torch.empty((0, 4), device=self.device)
            )
        else:
            # Keep alignment even if there are no changed edges
            if (n_added + n_removed) > 0:
                diff_graph.fastener_pos_diff = torch.zeros(
                    (n_added + n_removed, 3), device=self.device
                )
                diff_graph.fastener_quat_delta = torch.zeros(
                    (n_added + n_removed, 4), device=self.device
                )

        edge_attr = (
            torch.cat([added_attrs, removed_attrs, changed_attrs], dim=0)
            if (n_added + n_removed + n_changed) > 0
            else torch.empty((0, 6), device=self.device)
        )
        edge_mask = (
            edge_attr.any(dim=1)
            if edge_attr.numel() > 0
            else torch.empty((0,), dtype=torch.bool, device=self.device)
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
        # Defensive extraction of tensors (edge_attr and edge_index may be missing)
        edge_attr = getattr(diff_graph, "edge_attr", None)
        edge_index = getattr(diff_graph, "edge_index", None)
        if edge_attr is None:
            edge_attr = torch.empty((0, 6), device=self.device)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        # Masks for added/removed/changed
        is_added = (
            edge_attr[:, 0].bool()
            if edge_attr.numel()
            else torch.zeros((0,), dtype=torch.bool, device=self.device)
        )
        is_removed = (
            edge_attr[:, 1].bool()
            if edge_attr.numel()
            else torch.zeros((0,), dtype=torch.bool, device=self.device)
        )
        has_attr_change = (
            edge_attr[:, 2:].any(dim=1)
            if edge_attr.numel()
            else torch.zeros((0,), dtype=torch.bool, device=self.device)
        )
        changed_mask = (
            (~is_added) & (~is_removed) & has_attr_change
            if edge_attr.numel()
            else torch.zeros((0,), dtype=torch.bool, device=self.device)
        )

        result = {
            "nodes": {
                "changed_indices": diff_graph.node_mask.nonzero().squeeze(-1).tolist(),
                "position_diffs": diff_graph.position.tolist(),
                "quaternion_diffs": diff_graph.quat.tolist(),
                "count_fasteners_held_diff": diff_graph.count_fasteners_held_diff.tolist()
                if hasattr(diff_graph, "count_fasteners_held_diff")
                else [],
            },
            "edges": {
                "added": edge_index[:, is_added].t().tolist(),
                "removed": edge_index[:, is_removed].t().tolist(),
                "changed": edge_index[:, changed_mask].t().tolist(),
                "fastener_attr_flags": edge_attr.tolist(),
                "changed_flags": edge_attr[changed_mask].tolist()
                if edge_attr.numel()
                else [],
                "fastener_pos_diff": getattr(
                    diff_graph, "fastener_pos_diff", torch.empty((0, 3))
                ).tolist()
                if hasattr(diff_graph, "fastener_pos_diff")
                else [],
                "fastener_quat_delta": getattr(
                    diff_graph, "fastener_quat_delta", torch.empty((0, 4))
                ).tolist()
                if hasattr(diff_graph, "fastener_quat_delta")
                else [],
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
        changed_edges = diff_dict["edges"].get("changed", [])

        if added_edges:
            lines.append("\nAdded Edges:")
            for src, dst in added_edges:
                lines.append(f"  {src} -> {dst}")

        if removed_edges:
            lines.append("\nRemoved Edges:")
            for src, dst in removed_edges:
                lines.append(f"  {src} -> {dst}")

        if changed_edges:
            lines.append("\nChanged Edges (fastener attributes):")
            # Use changed_flags aligned with changed_edges
            changed_flags = diff_dict["edges"].get("changed_flags", [])
            for i, (src, dst) in enumerate(changed_edges):
                flags = (
                    changed_flags[i] if i < len(changed_flags) else [0, 0, 0, 0, 0, 0]
                )
                lines.append(
                    f"  {src} -> {dst} | diam:{bool(flags[2])} len:{bool(flags[3])} pos:{bool(flags[4])} quat:{bool(flags[5])}"
                )

        if not (changed_nodes or added_edges or removed_edges):
            lines.append("No changes detected.")

        return "\n".join(lines)

    # @deprecated("Warning: this is an incorrect implementation. Should not be used.")
    # def check_if_fastener_inserted(self, body_id: int, fastener_id: int) -> bool:
    #     """Check if a body (body_id) is still connected to a fastener (fastener_id)."""
    #     edge_index = self.edge_index
    #     # No edges means no connection
    #     if edge_index.numel() == 0:
    #         return False
    #     src, dst = edge_index
    #     # Check for edge in either direction
    #     connected_direct = ((src == body_id) & (dst == fastener_id)).any()
    #     connected_reverse = ((src == fastener_id) & (dst == body_id)).any()
    #     return bool(connected_direct or connected_reverse)
    # TODO: this should be refactored to "are two parts connected" which is what it does.

    def check_if_part_in_desired_pos(
        self, part_id: int, data_a: "PhysicalState", data_b: "PhysicalState"
    ) -> bool:
        """Check if a part (part_id) is in its desired position (within thresholds)."""
        # Compute node feature diffs (position & orientation) between data_a and data_b
        diff_dict, _ = _diff_body_features(data_a, data_b)
        changed = diff_dict["changed_indices"]
        # Return True if this part_id did not exceed thresholds

        # FIXME: this does not account for a) batch of environments, b) batch of parts.
        # ^batch of environments would be cool, but batch of parts would barely ever happen.
        return (changed == part_id).any()

    def _build_fastener_edge_attr_and_index(self, physical_info: PhysicalStateInfo):
        # cat shapes: 1,1,3,4,2 = 11
        # expected shape: [num_fasteners, 11]
        edge_attr = torch.cat(
            [
                physical_info.fasteners_diam.unsqueeze(-1),  #
                physical_info.fasteners_length.unsqueeze(-1),
                self.fasteners_pos,
                self.fasteners_quat,
                (
                    self.fasteners_attached_to_body != -1
                ).float(),  # if attached at a or b.
            ],
            dim=1,
        )
        # -- index --
        # Build body-body edge_index from fastener attachments (undirected, deduplicated)
        assert (
            self.fasteners_attached_to_body.ndim == 2
            and self.fasteners_attached_to_body.shape[1] == 2
        )
        assert self.position.ndim == 2 and self.quat.ndim == 2
        N_bodies = int(self.position.shape[0])
        atb = self.fasteners_attached_to_body
        valid = (atb[:, 0] >= 0) & (atb[:, 1] >= 0)
        if valid.any():
            pairs = atb[valid].to(torch.long)
            u = torch.minimum(pairs[:, 0], pairs[:, 1])
            v = torch.maximum(pairs[:, 0], pairs[:, 1])
            keys = u * N_bodies + v
            order = torch.argsort(keys)
            keys_sorted = keys[order]
            keys_u, counts = torch.unique(keys_sorted, return_counts=True)
            csum = torch.cumsum(counts, dim=0)
            first_sorted_idx = csum - counts
            idx_u = order[first_sorted_idx]
            edge_index = torch.stack([u[idx_u], v[idx_u]], dim=0)
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.position.device
            )

        return edge_attr, edge_index


def _diff_body_features(
    data_a: "PhysicalState",
    data_b: "PhysicalState",
    pos_threshold: float = 5.0 / 1000,
    deg_threshold: float = 5.0,
) -> tuple[dict, int]:
    """Return per-node diffs (static shape) and changed indices/count."""
    # Position diff
    pos_raw = data_b.position - data_a.position  # [B, N, 3]
    pos_dist = torch.linalg.norm(pos_raw, dim=2)
    pos_mask = pos_dist > pos_threshold  # [B, N]
    pos_diff = torch.zeros_like(pos_raw)
    pos_diff[pos_mask] = pos_raw[pos_mask]

    # Quaternion diff
    quat_delta = quaternion_delta(data_a.quat, data_b.quat)  # [N,4]
    rot_mask = ~are_quats_within_angle(
        data_a.quat, data_b.quat, torch.tensor(deg_threshold, device=data_a.quat.device)
    )  # [N]
    quat_diff = torch.zeros_like(quat_delta)
    quat_diff[rot_mask] = quat_delta[rot_mask]

    changed_mask = pos_mask | rot_mask
    changed_indices = torch.nonzero(changed_mask, as_tuple=False).squeeze(1)

    return (
        {
            "changed_indices": changed_indices,
            "pos_diff": pos_diff,
            "quat_diff": quat_diff,
        },
        int(changed_mask.sum().item()),
    )


def _diff_fastener_features(
    data_a: PhysicalState, data_b: PhysicalState, physical_info: PhysicalStateInfo
) -> tuple[dict, int]:
    """Compare fastener features between two PhysicalState instances. PhysicalStateInfo is supposed to be equal for both states.

    Expected logic:
        For each hole id, get which fastener is in it (by a tensor lookup), works since fasteners are unique (however there may be two indices of them as hole A and B). From there, get all fasteners' parameters (doable via a single tensor operation), and assert isclose of both fasteners' parameters. You have an equal set of fasteners in each parameter, but they may be in the wrong holes.
        (The whole diff shouldn't take more than 10 lines, it seems.)

    Additionally enforces fastener-hole compatibility: if a fastener is attached to a pair of holes, its diameter must equal the diameters of both holes it occupies. This is validated for both input states and uses strict equality within 1e-6. A mismatch raises an AssertionError.

    Returns a dict with:
        - added: list[(u, v)] added edges (by body indices, undirected)
        - removed: list[(u, v)] removed edges (by body indices, undirected)
        - changed_edges: list[(u, v)] edges present in both where fastener attrs changed
        - diam_changed: List[bool] same length as changed_edges
        - length_changed: List[bool] same length as changed_edges
        - pos_diff: Tensor [K, 3] diffs for changed edges (zeros if not changed)
        - quat_delta: Tensor [K, 4] quat deltas for changed edges (zeros if not changed)
    And an integer total count: len(added)+len(removed)+len(changed_edges)
    """
    assert (
        data_a.fasteners_pos.shape[1]
        == data_b.fasteners_pos.shape[1]
        == physical_info.fasteners_length.shape[0]
    ), (
        f"data_a, data_b and physical_info must have equal number of fasteners. Got {data_a.fasteners_pos.shape}, {data_b.fasteners_pos.shape} and {physical_info.fasteners_length.shape}"
    )
    assert data_a.fasteners_pos.shape[0] == data_b.fasteners_pos.shape[0], (
        "Batch dim mismatch across fastener features."
    )

    # Flatten batch for fastener features
    a_atb = (
        data_a.fasteners_attached_to_body.reshape(-1, 2)
        if data_a.fasteners_attached_to_body.ndim == 3
        else data_a.fasteners_attached_to_body
    )
    b_atb = (
        data_b.fasteners_attached_to_body.reshape(-1, 2)
        if data_b.fasteners_attached_to_body.ndim == 3
        else data_b.fasteners_attached_to_body
    )
    a_ath = (
        data_a.fasteners_attached_to_hole.reshape(-1, 2)
        if data_a.fasteners_attached_to_hole.ndim == 3
        else data_a.fasteners_attached_to_hole
    )
    b_ath = (
        data_b.fasteners_attached_to_hole.reshape(-1, 2)
        if data_b.fasteners_attached_to_hole.ndim == 3
        else data_b.fasteners_attached_to_hole
    )
    a_pos = (
        data_a.fasteners_pos.reshape(-1, 3)
        if data_a.fasteners_pos.ndim == 3
        else data_a.fasteners_pos
    )
    b_pos = (
        data_b.fasteners_pos.reshape(-1, 3)
        if data_b.fasteners_pos.ndim == 3
        else data_b.fasteners_pos
    )
    a_quat = (
        data_a.fasteners_quat.reshape(-1, 4)
        if data_a.fasteners_quat.ndim == 3
        else data_a.fasteners_quat
    )
    b_quat = (
        data_b.fasteners_quat.reshape(-1, 4)
        if data_b.fasteners_quat.ndim == 3
        else data_b.fasteners_quat
    )

    N_fast = int(physical_info.fasteners_length.shape[0])
    N_bodies = int(data_a.position.shape[1])
    H = (
        int(physical_info.hole_diameter.shape[0])
        if physical_info.hole_diameter.numel()
        else 0
    )

    # Vectorized hole/fastener diameter compatibility for both states
    def _check_diam(attached_to_hole: torch.Tensor):
        if attached_to_hole.numel() == 0:
            return
        valid = (attached_to_hole[:, 0] >= 0) & (attached_to_hole[:, 1] >= 0)
        if not bool(valid.any()):
            return
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        f_idx = idx % N_fast
        ha = attached_to_hole[idx, 0].to(torch.long)
        hb = attached_to_hole[idx, 1].to(torch.long)
        fd = physical_info.fasteners_diam[f_idx]
        da = physical_info.hole_diameter[ha]
        db = physical_info.hole_diameter[hb]
        close = (fd - da).abs() <= 1e-6
        close &= (fd - db).abs() <= 1e-6
        assert bool(close.all()), "Fastener-hole diameter mismatch detected"

    _check_diam(a_ath)
    _check_diam(b_ath)

    # Added/removed edges from body attachments (undirected, dedup across batch)
    def _edges_from_atb(atb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if atb.numel() == 0:
            return torch.empty((0,), dtype=torch.long), torch.empty(
                (0, 2), dtype=torch.long
            )
        valid = (atb[:, 0] >= 0) & (atb[:, 1] >= 0)
        pairs = atb[valid].to(torch.long)
        if pairs.numel() == 0:
            return torch.empty((0,), dtype=torch.long), torch.empty(
                (0, 2), dtype=torch.long
            )
        u = torch.minimum(pairs[:, 0], pairs[:, 1])
        v = torch.maximum(pairs[:, 0], pairs[:, 1])
        edges = torch.stack([u, v], dim=1)
        keys = u * N_bodies + v
        # Unique keys with first occurrence indices (PyTorch-compatible)
        order = torch.argsort(keys)
        keys_sorted = keys[order]
        keys_u, counts = torch.unique(keys_sorted, return_counts=True)
        # first index in sorted order per unique key
        csum = torch.cumsum(counts, dim=0)
        first_sorted_idx = csum - counts
        idx_u = order[first_sorted_idx]
        return keys_u, edges[idx_u]

    a_keys, a_edges = _edges_from_atb(a_atb)
    b_keys, b_edges = _edges_from_atb(b_atb)
    if a_keys.numel() == 0 and b_keys.numel() == 0:
        added_edges = torch.empty((2, 0), dtype=torch.long)
        removed_edges = torch.empty((2, 0), dtype=torch.long)
    else:
        add_mask = (
            (~torch.isin(b_keys, a_keys))
            if a_keys.numel()
            else torch.ones_like(b_keys, dtype=torch.bool)
        )
        rem_mask = (
            (~torch.isin(a_keys, b_keys))
            if b_keys.numel()
            else torch.ones_like(a_keys, dtype=torch.bool)
        )
        added_edges = (
            b_edges[add_mask].t()
            if add_mask.any()
            else torch.empty((2, 0), dtype=torch.long)
        )
        removed_edges = (
            a_edges[rem_mask].t()
            if rem_mask.any()
            else torch.empty((2, 0), dtype=torch.long)
        )

    # Changed edges by matching unique hole pairs across states
    def _unique_hole_keys(ath: torch.Tensor):
        if ath.numel() == 0 or H == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 2), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        valid = (ath[:, 0] >= 0) & (ath[:, 1] >= 0)
        hp = ath[valid].to(torch.long)
        if hp.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 2), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        h0 = torch.minimum(hp[:, 0], hp[:, 1])
        h1 = torch.maximum(hp[:, 0], hp[:, 1])
        keys = h0 * H + h1
        order = torch.argsort(keys)
        keys_sorted = keys[order]
        keys_u, counts = torch.unique(keys_sorted, return_counts=True)
        csum = torch.cumsum(counts, dim=0)
        first_sorted_idx = csum - counts
        idx_u = order[first_sorted_idx]
        # map back to original indices among valid rows
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        return keys_u, torch.stack([h0, h1], dim=1)[idx_u], valid_idx[idx_u]

    a_hkeys, _a_hpairs, a_sel = _unique_hole_keys(a_ath)
    b_hkeys, _b_hpairs, b_sel = _unique_hole_keys(b_ath)

    if a_hkeys.numel() == 0 or b_hkeys.numel() == 0:
        changed_edges = torch.empty((2, 0), dtype=torch.long)
        pos_diff = torch.empty((0, 3))
        quat_delta = torch.empty((0, 4))
        diam_changed = torch.empty((0,), dtype=torch.bool)
        length_changed = torch.empty((0,), dtype=torch.bool)
    else:
        b_sorted, b_order = torch.sort(b_hkeys)
        pos_in_b = torch.searchsorted(b_sorted, a_hkeys)
        in_bounds = (pos_in_b >= 0) & (pos_in_b < b_sorted.numel())
        match_mask = in_bounds & (
            b_sorted[pos_in_b.clamp(min=0, max=max(b_sorted.numel() - 1, 0))] == a_hkeys
        )
        if not bool(match_mask.any()):
            changed_edges = torch.empty((2, 0), dtype=torch.long)
            pos_diff = torch.empty((0, 3))
            quat_delta = torch.empty((0, 4))
            diam_changed = torch.empty((0,), dtype=torch.bool)
            length_changed = torch.empty((0,), dtype=torch.bool)
        else:
            a_idx = a_sel[match_mask]
            b_idx = b_sel[b_order[pos_in_b[match_mask]]]
            # body pairs from B for edge identity (undirected)
            bp = b_atb[b_idx].to(torch.long)
            u = torch.minimum(bp[:, 0], bp[:, 1])
            v = torch.maximum(bp[:, 0], bp[:, 1])
            changed_edges = torch.stack([u, v], dim=0)

            # feature deltas with thresholds
            pos_raw = b_pos[b_idx] - a_pos[a_idx]
            pos_dist = torch.linalg.norm(pos_raw, dim=1)
            pos_mask = pos_dist > (5.0 / 1000)
            pos_diff = torch.zeros_like(pos_raw)
            pos_diff[pos_mask] = pos_raw[pos_mask]

            q_delta = quaternion_delta(a_quat[a_idx], b_quat[b_idx])
            rot_mask = ~are_quats_within_angle(
                a_quat[a_idx], b_quat[b_idx], torch.tensor(5.0, device=q_delta.device)
            )
            quat_delta = torch.zeros_like(q_delta)
            quat_delta[rot_mask] = q_delta[rot_mask]

            # physical_info singleton (not diffed): flags remain False
            K = changed_edges.size(1)
            diam_changed = torch.zeros((K,), dtype=torch.bool)
            length_changed = torch.zeros((K,), dtype=torch.bool)

    total = int(
        added_edges.size(1)
        + removed_edges.size(1)
        + (0 if changed_edges.numel() == 0 else changed_edges.size(1))
    )
    out = {
        "added": added_edges,
        "removed": removed_edges,
        "changed_edges": changed_edges,
        "diam_changed": diam_changed,
        "length_changed": length_changed,
        "pos_diff": pos_diff,
        "quat_delta": quat_delta,
    }
    return out, total


def compound_pos_to_sim_pos(
    compound_pos: torch.Tensor, env_size_mm=(640, 640, 640)
) -> torch.Tensor:
    """Convert position in compound to position in sim/sim state."""
    assert compound_pos.ndim >= 2 and compound_pos.shape[-1] == 3, (
        f"compound_pos must be [N, 3] or [B, N, 3], got {compound_pos.shape}"
    )
    assert (compound_pos > 0).all(), (
        f"Compound_pos must be non-negative, including 0, got {compound_pos}"
    )
    env_size_mm = torch.tensor(env_size_mm, device=compound_pos.device)
    compound_pos_xy = (compound_pos[..., :2] - env_size_mm[:2] / 2) / 1000
    compound_pos_z = compound_pos[..., 2] / 1000  # z needs not go lower.
    return torch.cat([compound_pos_xy, compound_pos_z.unsqueeze(1)], dim=1)


# Standalone functions for batch processing PhysicalState
def register_bodies_batch(
    names: list[str],  # List of body names to register across all environments
    positions: torch.Tensor,  # [B, num_bodies, 3] positions for bodies across environments
    rotations: torch.Tensor,  # [B, num_bodies, 4] quaternions for bodies across environments
    fixed: torch.Tensor,  # [num_bodies]
    terminal_position_relative_to_center: torch.Tensor
    | None = None,  # [num_bodies, 3] for connectors, can contain NaN for non-connectors
    min_bounds: torch.Tensor = torch.tensor([-0.32, -0.32, 0.0]),
    max_bounds: torch.Tensor = torch.tensor([0.32, 0.32, 0.64]),
) -> tuple["PhysicalState", "PhysicalStateInfo"]:
    """Register multiple bodies across multiple environments using tensor operations.

    Args:
        names: List of body names to register across all environments
        positions: Tensor of positions [B, num_bodies, 3] for bodies across environments
        rotations: Tensor of quaternions [B, num_bodies, 4] for bodies across environments
        fixed: Tensor [num_bodies]
        terminal_position_relative_to_center: Relative positions for connectors [num_bodies, 3], NaN for non-connectors
        max_bounds: Maximum position bounds
        min_bounds: Minimum position bounds
    """
    B = positions.shape[0]
    num_bodies = len(names)
    device = positions.device
    physical_states: PhysicalState = torch.stack([PhysicalState(device=device)] * B)
    physical_info = PhysicalStateInfo(device=device)

    assert B > 0 and num_bodies > 0

    # Validate inputs
    assert positions.shape == (B, num_bodies, 3), (
        f"Expected positions shape [B, num_bodies, 3] [{B}, {num_bodies}, 3], got {positions.shape}"
    )
    assert rotations.shape == (B, num_bodies, 4), (
        f"Expected rotations shape [B, num_bodies, 4] [{B}, {num_bodies}, 4], got {rotations.shape}"
    )
    assert fixed.ndim == 1 and fixed.shape[0] == num_bodies, (
        f"Expected `fixed` shape [{num_bodies}], got {tuple(fixed.shape)}"
    )

    # Move to device
    positions = positions.to(device)
    rotations = rotations.to(device)
    fixed = fixed.to(device)
    max_bounds = max_bounds.to(device)
    min_bounds = min_bounds.to(device)

    # Validate bounds
    assert (positions >= min_bounds).all() and (positions <= max_bounds).all(), (
        f"Some positions are out of bounds [{min_bounds.tolist()}, {max_bounds.tolist()}]"
    )

    # Validate name formats and check for conflicts
    for name in names:
        assert name.endswith(("@solid", "@connector", "@fixed_solid")), (
            f"Body name must end with @solid, @connector or @fixed_solid. Got {name}"
        )
        # assert name not in physical_info.body_indices, f"Body {name} already registered"
    # assert isinstance(physical_info.body_indices, dict)
    assert len(names) == len(set(names)), "Duplicate body names."

    # Update body indices for all bodies
    start_body_idx = 0
    for i, name in enumerate(names):
        body_idx = start_body_idx + i
        physical_info.body_indices[name] = body_idx
        physical_info.inverse_body_indices[body_idx] = name

    # Set tensors directly for all environments (register_bodies_batch is called once)
    fastener_count = torch.zeros((B, num_bodies), dtype=torch.int8, device=device)

    physical_states.position = positions
    physical_states.quat = rotations
    physical_states.count_fasteners_held = fastener_count
    physical_info.fixed = fixed

    # Handle connectors using the batched connector update function
    if terminal_position_relative_to_center is not None:
        # Filter out connector bodies and their data
        connector_indices = [
            i for i, name in enumerate(names) if name.endswith("@connector")
        ]

        if connector_indices:
            connector_names = [names[i] for i in connector_indices]
            connector_positions = positions[
                :, connector_indices
            ]  # [B, num_connectors, 3]
            connector_rotations = rotations[
                :, connector_indices
            ]  # [B, num_connectors, 4]
            terminal_rel_positions = terminal_position_relative_to_center[
                connector_indices
            ]  # [num_connectors, 3]

            # Validate that connector positions are not NaN
            for i, name in enumerate(connector_names):
                terminal_rel_pos = terminal_rel_positions[i]
                if name.endswith(("_male@connector", "_female@connector")):
                    assert not torch.isnan(terminal_rel_pos).any(), (
                        f"Connector {name} must have valid terminal_position_relative_to_center, got NaN"
                    )

            # Use the batched connector update function
            male_terminal_positions, female_terminal_positions = (
                update_terminal_def_pos_batch(
                    connector_positions,
                    connector_rotations,
                    terminal_rel_positions,
                    connector_names,
                )
            )

            # Set connector positions on PhysicalState (batched)
            physical_states.male_terminal_positions = male_terminal_positions
            physical_states.female_terminal_positions = female_terminal_positions

            # Set up connector indices and batches
            male_indices = [
                i
                for i, name in enumerate(connector_names)
                if name.endswith("_male@connector")
            ]
            female_indices = [
                i
                for i, name in enumerate(connector_names)
                if name.endswith("_female@connector")
            ]

            # Set up male connector batch and indices
            if male_indices:
                male_names = [connector_names[i] for i in male_indices]
                male_body_indices = [
                    physical_info.body_indices[name] for name in male_names
                ]
                # Create batched tensor with shape [B, num_male_connectors]
                male_batch_tensor = torch.tensor(
                    male_body_indices, dtype=torch.long, device=device
                )
                # Expand to batch dimension if needed
                if male_batch_tensor.ndim == 1:
                    # FIXME: this is wrong. male_batch_tensor is meant to be ndim=1!
                    male_batch_tensor = male_batch_tensor.unsqueeze(0).expand(B, -1)
                physical_info.male_terminal_batch = male_batch_tensor
                # Update connector indices for all batch elements
                for i, name in enumerate(male_names):
                    physical_info.terminal_indices_from_name[name] = i
            else:
                # Set empty batched tensor
                physical_info.male_terminal_batch = torch.empty(
                    (B, 0), dtype=torch.long, device=device
                )

            # Set up female connector batch and indices
            if female_indices:
                female_names = [connector_names[i] for i in female_indices]
                female_body_indices = [
                    physical_info.body_indices[name] for name in female_names
                ]
                # Create batched tensor with shape [B, num_female_connectors]
                female_batch_tensor = torch.tensor(
                    female_body_indices, dtype=torch.long, device=device
                )
                # Expand to batch dimension if needed
                if len(female_batch_tensor.shape) == 1:
                    # FIXME: this is wrong. female_batch_tensor is meant to be ndim=1!
                    female_batch_tensor = female_batch_tensor.unsqueeze(0).expand(B, -1)
                physical_info.female_terminal_batch = female_batch_tensor
                # Update connector indices for all batch elements
                for i, name in enumerate(female_names):
                    physical_info.terminal_indices_from_name[name] = i
            else:
                # Set empty batched tensor
                physical_info.female_terminal_batch = torch.empty(
                    (B, 0), dtype=torch.long, device=device
                )

    return physical_states, physical_info


def update_bodies_batch(
    physical_states: "PhysicalState",
    physical_info: "PhysicalStateInfo",
    names: list[str],
    positions: torch.Tensor,  # [B, num_update, 3]
    rotations: torch.Tensor,  # [B, num_update, 4]
    terminal_position_relative_to_center: torch.Tensor
    | None = None,  # [num_update, 3] with NaNs for non-connectors
    max_bounds: torch.Tensor = torch.tensor([0.32, 0.32, 0.64]),
    min_bounds: torch.Tensor = torch.tensor([-0.32, -0.32, 0.0]),
) -> "PhysicalState":
    """Update multiple bodies across multiple environments in batch.

    - Applies updates only where bodies are not fixed (per-environment).
    - Validates bounds and quaternion normalization.
    - Updates connector positions in-place for the updated connectors.

    Args:
        physical_states: Batched PhysicalState to update.
        names: List of body names to update.
        positions: [B, num_update, 3]
        rotations: [B, num_update, 4]
        terminal_position_relative_to_center: [num_update, 3] relative positions; NaN for non-connectors.
        max_bounds, min_bounds: Bounds for normalized coordinates.

    Returns:
        Updated PhysicalState.
    """
    assert positions.ndim == 3 and positions.shape[-1] == 3, (
        f"Expected positions shape [B, num_update, 3], got {positions.shape}"
    )
    assert rotations.ndim == 3 and rotations.shape[-1] == 4, (
        f"Expected rotations shape [B, num_update, 4], got {rotations.shape}"
    )
    B, num_update, _ = positions.shape

    # Sanity checks for names
    assert len(names) == num_update, (
        f"Got {len(names)} names but positions/rotations have {num_update} bodies"
    )
    assert len(set(names)) == len(names), (
        "Duplicate names in update list are not allowed"
    )
    for name in names:
        assert name in physical_info.body_indices, (
            f"Body {name} not registered. Registered: {list(physical_info.body_indices.keys())}"
        )

    device = physical_states.device
    positions = positions.to(device)
    rotations = rotations.to(device)
    max_bounds = max_bounds.to(device)
    min_bounds = min_bounds.to(device)

    # Validate bounds and rotations
    assert (positions >= min_bounds).all() and (positions <= max_bounds).all(), (
        f"Some positions are out of bounds [{min_bounds.tolist()}, {max_bounds.tolist()}]"
    )
    sanitize_quaternion(rotations)

    # Map names -> body indices
    idxs = torch.tensor(
        [physical_info.body_indices[n] for n in names],
        dtype=torch.long,
        device=device,
    )

    # Update positions/quaternions where not fixed
    not_fixed = ~physical_info.fixed[idxs]  # [num_update]
    # Positions
    cur_pos = physical_states.position[:, idxs, :]
    upd_pos = torch.where(not_fixed.unsqueeze(-1), positions, cur_pos)
    physical_states.position[:, idxs, :] = upd_pos
    # Rotations
    cur_quat = physical_states.quat[:, idxs, :]
    upd_quat = torch.where(not_fixed.unsqueeze(-1), rotations, cur_quat)
    physical_states.quat[:, idxs, :] = upd_quat

    # Handle connectors: recompute positions for connectors among the updated names
    connector_indices_local = [
        i for i, n in enumerate(names) if n.endswith("@connector")
    ]
    if connector_indices_local:
        assert terminal_position_relative_to_center is not None, (
            "Updated connectors must provide terminal_position_relative_to_center"
        )
        assert terminal_position_relative_to_center.shape == (num_update, 3), (
            f"Expected terminal_position_relative_to_center shape [{num_update}, 3], got {terminal_position_relative_to_center.shape}"
        )

        connector_names = [names[i] for i in connector_indices_local]
        terminal_rel = terminal_position_relative_to_center[connector_indices_local].to(
            device
        )

        # Validate connector rel positions for the connectors
        for i, cname in enumerate(connector_names):
            if cname.endswith(("_male@connector", "_female@connector")):
                assert not torch.isnan(terminal_rel[i]).any(), (
                    f"Connector {cname} must have valid terminal_position_relative_to_center, got NaN"
                )
        assert (terminal_rel.abs() < 5).all(), (
            "Likely dimension error: it is unlikely that a connector is further than 5m from the center of the part."
        )

        # Use updated state tensors (with fixed masking applied) to recompute connector positions
        body_idxs_for_connectors = idxs[connector_indices_local]
        conn_positions = physical_states.position[
            :, body_idxs_for_connectors, :
        ]  # [B, Nc, 3]
        conn_rotations = physical_states.quat[
            :, body_idxs_for_connectors, :
        ]  # [B, Nc, 4]

        male_pos_new, female_pos_new = update_terminal_def_pos_batch(
            conn_positions, conn_rotations, terminal_rel, connector_names
        )

        # Write back into preallocated connector position tensors using stored indices
        # terminal_indices_from_name now lives in physical_info
        male_indices_local = [
            i for i, n in enumerate(connector_names) if n.endswith("_male@connector")
        ]
        female_indices_local = [
            i for i, n in enumerate(connector_names) if n.endswith("_female@connector")
        ]

        # Update male connector positions
        if male_indices_local:
            # male_pos_new corresponds to the male subset order
            name_to_idx = physical_info.terminal_indices_from_name
            for b in range(B):
                for j, local_i in enumerate(male_indices_local):
                    cname = connector_names[local_i]
                    assert cname in name_to_idx, (
                        f"Connector {cname} not found in state mapping"
                    )
                    dst_idx = name_to_idx[cname]
                    physical_states.male_terminal_positions[b, dst_idx, :] = (
                        male_pos_new[b, j, :]
                    )
        # Update female connector positions
        if female_indices_local:
            name_to_idx = physical_info.terminal_indices_from_name
            for b in range(B):
                for j, local_i in enumerate(female_indices_local):
                    cname = connector_names[local_i]
                    assert cname in name_to_idx, (
                        f"Connector {cname} not found in state mapping"
                    )
                    dst_idx = name_to_idx[cname]
                    physical_states.female_terminal_positions[b, dst_idx, :] = (
                        female_pos_new[b, j, :]
                    )

    return physical_states


def register_fasteners_batch(
    physical_states: "PhysicalState",
    physical_info: "PhysicalStateInfo",
    fastener_pos: torch.Tensor,
    fastener_quat: torch.Tensor,
    fastener_init_hole_a: torch.Tensor,
    fastener_init_hole_b: torch.Tensor,
    fastener_compound_names: list[str],
) -> tuple[PhysicalState, PhysicalStateInfo]:
    """Register a single fastener across multiple environments using tensor operations.

    Args:
        physical_states: Batched PhysicalState to update
        fastener_pos: Tensor of positions [B, num_fasteners, 3] for this fastener across environments
        fastener_quat: Tensor of quaternions [B, num_fasteners, 4] for this fastener across environments
        fastener_init_hole_a: hole A of fasteners [B, num_fasteners]
        fastener_init_hole_b: hole B of fasteners [B, num_fasteners]
        fastener_compound_names: List of compound names for this fastener across environments
    """
    B = physical_states.batch_size[0]
    # Determine batch size from physical_states
    assert B is not None and physical_states.ndim == 1
    num_fasteners = len(fastener_compound_names)
    num_bodies = len(physical_info.body_indices)
    assert fastener_pos.shape == (B, num_fasteners, 3), (
        f"Expected fastener_pos shape [batch_size, num_fasteners, 3], got {fastener_pos.shape}"
    )
    assert fastener_quat.shape == (B, num_fasteners, 4), (
        f"Expected fastener_quat shape [batch_size, num_fasteners, 4], got {fastener_quat.shape}"
    )
    assert fastener_init_hole_a.shape == (B, num_fasteners), (
        f"Expected fastener_init_hole_a shape [batch_size, num_fasteners], got {fastener_init_hole_a.shape}"
    )
    assert fastener_init_hole_b.shape == (B, num_fasteners), (
        f"Expected fastener_init_hole_b shape [batch_size, num_fasteners], got {fastener_init_hole_b.shape}"
    )
    assert (
        (fastener_init_hole_a != fastener_init_hole_b)
        | ((fastener_init_hole_a == -1) & (fastener_init_hole_b == -1))
    ).all(), "Fastener init holes must be different or empty (-1)."

    device = (
        physical_states.device
        if hasattr(physical_states, "device")
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # If there are no fasteners, set empty tensors/scalars and return early.
    if num_fasteners == 0:  # unnecessary? just code bloat.
        physical_info.fasteners_diam = torch.empty(
            (0,), dtype=torch.float32, device=device
        )
        physical_info.fasteners_length = torch.empty(
            (0,), dtype=torch.float32, device=device
        )
        physical_states.fasteners_pos = torch.empty(
            (B, 0, 3), dtype=torch.float32, device=device
        )
        physical_states.fasteners_quat = torch.empty(
            (B, 0, 4), dtype=torch.float32, device=device
        )
        physical_states.fasteners_attached_to_hole = torch.empty(
            (B, 0, 2), dtype=torch.int64, device=device
        )
        physical_states.fasteners_attached_to_body = torch.empty(
            (B, 0, 2), dtype=torch.int64, device=device
        )
        physical_states.count_fasteners_held = torch.zeros(
            (B, num_bodies), dtype=torch.int8, device=device
        )
        return physical_states, physical_info

    # PhysicalStateInfo is a singleton store; fastener scalars must be 1D [N]
    physical_info.fasteners_diam = torch.full(
        (num_fasteners,), fill_value=-1.0, dtype=torch.float32, device=device
    )
    physical_info.fasteners_length = torch.full(
        (num_fasteners,), fill_value=-1.0, dtype=torch.float32, device=device
    )  # fill with -1 just in case
    for i, name in enumerate(fastener_compound_names):
        diam, length = get_fastener_params_from_name(name)
        # get_fastener_params_from_name returns values in millimeters.
        # Convert to meters for consistency with PhysicalState units.
        physical_info.fasteners_diam[i] = torch.tensor(
            diam / 1000.0, dtype=torch.float32, device=device
        )
        physical_info.fasteners_length[i] = torch.tensor(
            length / 1000.0, dtype=torch.float32, device=device
        )

    # Runtime guardrails to catch unit mistakes (e.g., mm accidentally stored as meters)
    assert (physical_info.fasteners_diam > 0).all(), (
        "Fastener diameters must be positive (meters)."
    )
    # Typical fastener diameters are < 0.05 m (50 mm); use 0.1 m as a generous upper bound
    assert physical_info.fasteners_diam.max() < 0.1, (
        "Fastener diameters look too large; expected meters. Did you forget mm->m conversion?"
    )
    assert (physical_info.fasteners_length > 0).all(), (
        "Fastener lengths must be positive (meters)."
    )
    # Typical fastener lengths are < 0.25 m (250 mm); use 0.5 m as a generous upper bound
    assert physical_info.fasteners_length.max() < 0.5, (
        "Fastener lengths look too large; expected meters. Did you forget mm->m conversion?"
    )

    assert physical_info.part_hole_batch is not None, (
        "Part hole batch must be set before registering fasteners."
    )
    # PhysicalStateInfo is singleton (no batch). Enforce 1D mapping [H].
    assert physical_info.part_hole_batch.ndim == 1, (
        "part_hole_batch must be 1D [H] in PhysicalStateInfo"
    )
    H = physical_info.part_hole_batch.shape[0]
    assert all(
        name not in physical_info.body_indices for name in fastener_compound_names
    ), (
        f"Fasteners can't be registered as bodies! Fastener (compound!) names: {fastener_compound_names}"
    )

    assert ((fastener_init_hole_a >= -1) & (fastener_init_hole_a < H)).all(), (
        f"fastener_init_hole_a are out of range. Num holes: {H}. "
        f"Min: {physical_info.part_hole_batch.min()}. Max: {physical_info.part_hole_batch.max()}. All: {fastener_init_hole_a}"
    )
    assert ((fastener_init_hole_b >= -1) & (fastener_init_hole_b < H)).all(), (
        f"fastener_init_hole_b are out of range. Num holes: {H}. "
        f"Min: {physical_info.part_hole_batch.min()}. Max: {physical_info.part_hole_batch.max()}. All: {fastener_init_hole_b}"
    )

    # Update PhysicalState tensors for this environment
    physical_states.fasteners_pos = fastener_pos.to(device)
    physical_states.fasteners_quat = fastener_quat.to(device)
    physical_states.fasteners_attached_to_hole = torch.stack(
        [fastener_init_hole_a, fastener_init_hole_b], dim=-1
    )
    empty = physical_states.fasteners_attached_to_hole == -1
    physical_states.fasteners_attached_to_body = torch.full_like(
        physical_states.fasteners_attached_to_hole, -1, dtype=torch.int64
    )
    physical_states.fasteners_attached_to_body[~empty] = physical_info.part_hole_batch[
        physical_states.fasteners_attached_to_hole[~empty]
    ]

    # Update fastener counts for connected bodies
    # Count how many fasteners are attached to each body for each batch
    # physical_states.fasteners_attached_to_body shape: [B, num_fasteners, 2] (body_a, body_b)
    # Use any() to check if a fastener is attached to each body, then sum to count fasteners
    physical_states.count_fasteners_held = (
        (
            physical_states.fasteners_attached_to_body[:, :, None, :]
            == torch.arange(num_bodies)[None, None, :, None]
        )
        .any(dim=-1)
        .int()
        .sum(dim=1)
    ).to(torch.int8)  # [B, num_bodies]
    assert (physical_states.count_fasteners_held <= num_fasteners).all(), (
        "Bodies can't hold more fasteners than there are."
    )

    return physical_states, physical_info


def update_terminal_def_pos_batch(
    positions: torch.Tensor,  # [B, num_connectors, 3] positions for connector parts across environments
    rotations: torch.Tensor,  # [B, num_connectors, 4] quaternions for connector parts across environments
    terminal_positions_relative_to_center: torch.Tensor,  # [num_connectors, 3] relative positions
    names: list[str],  # List of connector names
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update connector positions in batch for multiple environments.

    Args:
        positions: Tensor of positions [B, num_connectors, 3] for connectors across environments
        rotations: Tensor of quaternions [B, num_connectors, 4] for connectors across environments
        terminal_positions_relative_to_center: Relative positions [num_connectors, 3]
        names: List of connector names

    Returns:
        Tuple of (male_terminal_positions, female_terminal_positions)
        - male_terminal_positions: [B, num_male_connectors, 3]
        - female_terminal_positions: [B, num_female_connectors, 3]
    """
    B, num_connectors, _ = positions.shape
    device = positions.device

    # Validate inputs
    assert rotations.shape == (B, num_connectors, 4), (
        f"Expected rotations shape [{B}, {num_connectors}, 4], got {rotations.shape}"
    )
    assert terminal_positions_relative_to_center.shape == (num_connectors, 3), (
        f"Expected terminal_positions_relative_to_center shape [{num_connectors}, 3], got {terminal_positions_relative_to_center.shape}"
    )
    assert len(names) == num_connectors, (
        f"Expected {num_connectors} names, got {len(names)}"
    )

    # Validate all names are connectors
    for name in names:
        assert name.endswith(("_male@connector", "_female@connector")), (
            f"Connector {name} must end with _male@connector or _female@connector."
        )

    # Validate relative positions are reasonable
    assert (terminal_positions_relative_to_center.abs() < 5).all(), (
        "Likely dimension error: it is unlikely that a connector is further than 5m from the center of the part."
    )

    # Move to device
    terminal_positions_relative_to_center = terminal_positions_relative_to_center.to(
        device
    )

    # Calculate terminal positions for all environments and connectors
    # We need to expand relative positions to match batch dimension
    terminal_rel_expanded = terminal_positions_relative_to_center.unsqueeze(0).expand(
        B, -1, -1
    )  # [B, num_connectors, 3]

    # Calculate terminal positions using get_connector_pos for each connector
    all_terminal_positions = torch.zeros_like(positions)  # [B, num_connectors, 3]

    for i in range(num_connectors):
        terminal_pos = get_connector_pos(
            positions[:, i],  # [B, 3]
            rotations[:, i],  # [B, 4]
            terminal_rel_expanded[:, i],  # [B, 3]
        )
        all_terminal_positions[:, i] = terminal_pos

    # Separate male and female connectors
    male_indices = [
        i for i, name in enumerate(names) if name.endswith("_male@connector")
    ]
    female_indices = [
        i for i, name in enumerate(names) if name.endswith("_female@connector")
    ]

    # male_names = [names[i] for i in male_indices]  # Not used in this function
    # female_names = [names[i] for i in female_indices]  # Not used in this function

    if male_indices:
        male_terminal_positions = all_terminal_positions[
            :, male_indices
        ]  # [B, num_male, 3]
    else:
        male_terminal_positions = torch.empty((B, 0, 3), device=device)

    if female_indices:
        female_terminal_positions = all_terminal_positions[
            :, female_indices
        ]  # [B, num_female, 3]
    else:
        female_terminal_positions = torch.empty((B, 0, 3), device=device)

    return male_terminal_positions, female_terminal_positions


def connect_fastener_to_one_body(
    physical_state: PhysicalState,
    physical_info: PhysicalStateInfo,
    fastener_id: int,
    body_name: str,
    hole_id: int,
    env_idx: torch.Tensor,
):
    """Connect a fastener to a body and a specific hole.

    Contract:
    - hole_id must be valid and belong to body_name (via part_hole_batch)
    - hole diameter must closely match fastener diameter (meters)
    - fasteners_attached_to_body and fasteners_attached_to_hole are updated
      in the same free slot (0 or 1)
    """
    # Validate available slot
    assert (
        physical_state.fasteners_attached_to_body[env_idx, fastener_id] == -1
    ).any(), "Fastener is already connected to two bodies."

    # Validate hole_id
    H = int(physical_info.part_hole_batch.shape[0])
    assert 0 <= hole_id < H, f"hole_id out of range: {hole_id} not in [0, {H})"

    # Validate hole belongs to the target body
    target_body_idx = physical_info.body_indices[body_name]
    assert physical_info.part_hole_batch[hole_id].item() == target_body_idx, (
        f"Hole {hole_id} does not belong to body '{body_name}' (expected part index {target_body_idx}, "
        f"got {int(physical_info.part_hole_batch[hole_id].item())})"
    )

    # Assert hole/fastener diameter closeness in meters
    # Note: diameters are stored in meters across the project
    fastener_d = float(physical_info.fasteners_diam[fastener_id].item())
    hole_d = float(physical_info.hole_diameter[hole_id].item())
    tol = 0.0005  # 0.5 mm
    assert abs(hole_d - fastener_d) <= tol, (
        f"Hole/fastener diameter mismatch: hole_d={hole_d}m, fastener_d={fastener_d}m, tol={tol}m"
    )

    # Choose slot 0 if it is free, otherwise use slot 1
    slot0_free = (
        physical_state.fasteners_attached_to_body[env_idx, fastener_id, 0] == -1
    )
    # Ensure we have a Python bool even if this is a 0-dim tensor
    if isinstance(slot0_free, torch.Tensor):
        slot0_free = bool(slot0_free.item())
    free_slot = 0 if slot0_free else 1

    # Update body and hole slots consistently
    physical_state.fasteners_attached_to_body[env_idx, fastener_id, free_slot] = (
        target_body_idx
    )
    # Ensure hole slot is also free in the same position
    assert (
        physical_state.fasteners_attached_to_hole[env_idx, fastener_id, free_slot] == -1
    ), "Target hole slot already occupied for this fastener."
    physical_state.fasteners_attached_to_hole[env_idx, fastener_id, free_slot] = hole_id

    return physical_state
