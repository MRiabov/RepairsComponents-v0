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


# @tensorclass # complains for some reason.
class PhysicalState(TensorClass):
    # device: torch.device = field(
    #     default_factory=lambda: torch.device(
    #         "cuda" if torch.cuda.is_available() else "cpu"
    #     )
    # ) #already included in tensordict
    # Node attributes (previously in graph)

    # TODO: batch size field?
    position: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Solids positions (not fasteners)"""
    quat: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 4), dtype=torch.float32)
    )
    """Solids quaternions (not fasteners)"""
    fixed: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.bool)
    )
    """Fixed body flags (not fasteners)"""
    count_fasteners_held: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.int8)
    )
    """Count of fasteners held by each body (not fasteners)"""

    # Edge attributes (previously in graph)
    edge_index: torch.Tensor = field(
        default_factory=lambda: torch.empty((2, 0), dtype=torch.long)
    )
    """Edge connections (fastener connections) [2, num_edges]"""
    edge_attr: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 12), dtype=torch.float32)
    )
    """Edge attributes (fastener connections) [num_edges, edge_feature_dim]
    Includes:
    - fastener diameter (1)
    - fastener length (1)
    - fastener position (3)
    - fastener quaternion (4)
    - is_connected_a/b (2)

    """
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
    fasteners_diam: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Fastener diameters in meters [num_fasteners]"""
    fasteners_length: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    """Fastener lengths in meters [num_fasteners]"""
    # TODO encode mass, and possibly velocity.
    # note: fasteners_inserted_into_holes is not meant to be exported. for internal ref in screw in logic only.

    # body_indices and inverse_body_indices
    body_indices: dict[str, int] = field(default_factory=dict)
    inverse_body_indices: dict[int, str] = field(default_factory=dict)

    # fastener_ids: dict[int, str] = field(default_factory=dict) # fixme: I don't remember how, but this is unused.
    hole_indices_from_name: dict[str, int] = field(default_factory=dict)
    """Hole indices per part name."""
    part_hole_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
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

    # Connector attributes (tensor-based batching)
    connector_indices_from_name: dict[str, int] = field(default_factory=dict)
    """Connector indices per connector name."""
    male_connector_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Male connector indices per part in the batch."""
    female_connector_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Female connector indices per part in the batch."""
    male_connector_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Male connector positions per part, batched with male_connector_batch."""
    female_connector_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )
    """Female connector positions per part, batched with female_connector_batch."""

    permanently_constrained_parts: list[list[str]] = field(default_factory=list)
    """List of lists of permanently constrained parts (linked_groups from EnvSetup)"""

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

    def __post_init__(self):
        # Handle batched states where body_indices is a list of dicts
        assert isinstance(self.body_indices, (list, dict)), (
            "Expected list or dict for body_indices"
        )
        assert self.device is not None, "Device must be set"
        if isinstance(self.body_indices, list):
            assert self.batch_size and self.batch_size[0] >= 1, "Batch size must be set"
            # For batched states, use the first dict as reference
            # (assuming all batches have the same structure)
            assert len(self.body_indices) == self.batch_size[0], (
                f"batch dim mismatch! Got {len(self.body_indices)}, expected {self.batch_size}"
            )
            first_item = self.body_indices[0]
            self.body_indices = first_item
            assert len(first_item) == self.fixed.shape[1], (
                f"Dim mismatch! Got {len(first_item)}, expected {self.fixed.shape[1]}"
            )  # since `fixed` has an extra bdim yet.
            # ^ note: this could be batch dim there, so if it is, set to shape[1]
            assert isinstance(first_item, dict)
            self.inverse_body_indices = {v: k for k, v in first_item.items()}

            # # this also means that all other bodies are stacked, so...
            # self.fixed = self.fixed[0]
            # self.part_hole_batch = self.part_hole_batch[0]
            # note: tensorclass typechecks for batch dim, and I can't set it. So, in singleton dimensions simply use [env_idx] syntax.

        else:
            # For single states, use the dict directly
            self.inverse_body_indices = {v: k for k, v in self.body_indices.items()}

    def export_graph(self):
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
        edge_attr = self._build_fastener_edge_attr()
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
            edge_index=self.edge_index,
            edge_attr=edge_attr,  # e.g. fastener size.
            num_nodes=len(self.body_indices),
            global_feat=global_feat_export,
        )
        # print("debug: graph global feat shape", graph.global_feat.shape)
        return graph

    def register_body(
        self,
        name: str,
        position: tuple,
        rotation: tuple,
        fixed: bool = False,
        rot_as_quat: bool = False,  # mostly for convenience in testing
        _expect_unnormalized_coordinates: bool = True,  # mostly for tests...
        connector_position_relative_to_center: torch.Tensor | None = None,
        max_bounds: torch.Tensor = torch.tensor([0.32, 0.32, 0.64]),
        min_bounds: torch.Tensor = torch.tensor([-0.32, -0.32, 0.0]),
    ):
        assert name not in self.body_indices, f"Body {name} already registered"
        assert name.endswith(("@solid", "@connector", "@fixed_solid")), (
            f"Body name must end with @solid, @connector or @fixed_solid. Got {name}"
        )  # note: fasteners don't go here, they go in `register_fastener`.
        assert len(position) == 3, f"Position must be 3D vector, got {position}"

        min_bounds = min_bounds.to(self.device)
        max_bounds = max_bounds.to(self.device)

        # position_sim because that's what we put into sim state.
        if _expect_unnormalized_coordinates:
            position_sim = compound_pos_to_sim_pos(
                torch.tensor(position, device=self.device).unsqueeze(0)
            )
        else:  # else to genesis (note: flip the var name to normalized coords.)
            position_sim = torch.tensor(position, device=self.device).unsqueeze(0)

        assert (position_sim >= min_bounds).all() and (
            position_sim <= max_bounds
        ).all(), (
            f"Expected register position to be in [{min_bounds.tolist()}, {max_bounds.tolist()}], got {(position_sim.tolist())} at body {name}"
        )
        if not rot_as_quat:
            assert len(rotation) == 3, (
                f"Rotation must be 3D vector, got {rotation}. If you want to pass a quaternion, set rot_as_quat to True."
            )
            rotation = euler_deg_to_quat_wxyz(torch.tensor(rotation))
        else:
            rotation = sanitize_quaternion(rotation)
        rotation = rotation.to(self.device).unsqueeze(0)

        idx = len(self.body_indices)
        self.body_indices[name] = idx
        self.inverse_body_indices[idx] = name

        # Update dataclass fields directly
        self.position = torch.cat([self.position, position_sim], dim=0)
        self.quat = torch.cat([self.quat, rotation], dim=0)
        self.count_fasteners_held = torch.cat(
            [
                self.count_fasteners_held,
                torch.zeros(1, dtype=torch.int8, device=self.device),
            ],
            dim=0,
        )
        self.fixed = torch.cat(
            [
                self.fixed,
                torch.tensor([fixed], dtype=torch.bool, device=self.device).tile(
                    (self.batch_size[0],)
                ),
            ],
            dim=0,
        )  # maybe will put this into hint instead as -1s or something.

        # handle male and female connector positions.
        if name.endswith("@connector"):
            assert connector_position_relative_to_center is not None, (
                f"Connector {name} must have a connector position relative to center."
            )
            self._update_connector_def_pos(
                name, position_sim, rotation, connector_position_relative_to_center
            )
        return self

    def update_body(
        self,
        name: str,
        position: tuple,
        rotation: tuple,
        connector_position_relative_to_center: torch.Tensor | None = None,
    ):
        "Note: expects normalized coordinates."
        assert name in self.body_indices, (
            f"Body {name} not registered. Registered bodies: {self.body_indices.keys()}"
        )
        pos_tensor = torch.tensor(position, device=self.device)
        assert (
            pos_tensor >= torch.tensor([-0.32, -0.32, 0.0]).to(self.device)
        ).all() and (
            pos_tensor <= torch.tensor([0.32, 0.32, 0.64]).to(self.device)
        ).all(), (
            f"Position {position} out of bounds. Expected [-0.32, 0.32] for update."
        )
        pos_tensor = pos_tensor
        rotation = sanitize_quaternion(rotation).to(device=self.device)
        if self.fixed[self.body_indices[name]]:
            # assert torch.isclose(
            #     torch.tensor(position, device=self.device),
            #     self.position[self.body_indices[name]],
            #     atol=1e-6,
            # ).all(), f"Body {name} is fixed and cannot be moved."
            # FIXME: fix
            return

        idx = self.body_indices[name]
        self.position[idx] = pos_tensor
        self.quat[idx] = rotation

        if name.endswith("@connector"):
            assert connector_position_relative_to_center is not None, (
                f"Connector {name} must have a connector position relative to center."
            )
            self._update_connector_def_pos(
                name,
                pos_tensor.unsqueeze(0),
                rotation.unsqueeze(0),
                connector_position_relative_to_center,
            )

        return self  # maybe that would fix view issues.

    def register_fastener(self, fastener: Fastener):
        """A fastener method to register fasteners and add all necessary components.
        Handles constraining to bodies and adding to graph.

        Args:
        - count_holes: Number of holes in the batch. If None, uses the number of holes in the batch (necessary during initial population)
        """
        assert fastener.name not in self.body_indices, (
            f"Fasteners can't be registered as bodies! Attempted at {fastener.name}"
        )
        assert self.part_hole_batch is not None, (
            "Part hole batch must be set before registering fasteners."
        )

        # Convert hole IDs to body names using hole_indices_batch
        initial_body_a = None
        initial_body_b = None

        if fastener.initial_hole_id_a is not None:
            assert 0 <= fastener.initial_hole_id_a < self.part_hole_batch.shape[0], (
                f"Hole ID {fastener.initial_hole_id_a} is out of range. Num holes: {self.part_hole_batch.shape[0]}"
            )
            body_idx_a = int(self.part_hole_batch[fastener.initial_hole_id_a].item())
            initial_body_a = self.inverse_body_indices[body_idx_a]

        if fastener.initial_hole_id_b is not None:
            assert 0 <= fastener.initial_hole_id_b < self.part_hole_batch.shape[0], (
                f"Hole ID {fastener.initial_hole_id_b} is out of range. Num holes: {self.part_hole_batch.shape[0]}"
                f"Hole ID {fastener.initial_hole_id_b} is out of range. Available holes: 0-{self.part_hole_batch.shape[0] - 1}"
            )
            body_idx_b = int(self.part_hole_batch[fastener.initial_hole_id_b].item())
            initial_body_b = self.inverse_body_indices[body_idx_b]

        fastener_id = len(self.fasteners_pos)
        self.fasteners_pos = torch.cat(
            [self.fasteners_pos, torch.zeros((1, 3), device=self.device)], dim=0
        )

        self.fasteners_quat = torch.cat(
            [self.fasteners_quat, torch.zeros((1, 4), device=self.device)], dim=0
        )

        self.fasteners_diam = torch.cat(
            [
                self.fasteners_diam,
                torch.tensor(fastener.diameter, device=self.device).unsqueeze(0),
            ],
            dim=0,
        )
        self.fasteners_length = torch.cat(
            [
                self.fasteners_length,
                torch.tensor(fastener.length, device=self.device).unsqueeze(0),
            ],
            dim=0,
        )
        self.fasteners_attached_to_body = torch.cat(
            [
                self.fasteners_attached_to_body,
                torch.full((1, 2), -1, device=self.device),
            ],
            dim=0,
        )  # note: technically fasteners_attached_to_body is a junk value as it simply repeats fasteners_attached_to_hole except with part IDs.
        self.fasteners_attached_to_hole = torch.cat(
            [
                self.fasteners_attached_to_hole,
                torch.full((1, 2), -1, device=self.device),
            ],
            dim=0,
        )

        if initial_body_a is not None:
            self.connect_fastener_to_one_body(fastener_id, initial_body_a)
        if initial_body_b is not None:
            self.connect_fastener_to_one_body(fastener_id, initial_body_b)
        return self  # maybe that would fix view issues.

    def connect_fastener_to_one_body(self, fastener_id: int, body_name: str):
        """Connect a fastener to a body. Used during screw-in and initial construction."""
        # FIXME: but where to get/store fastener ids I'll need to think.
        assert (self.fasteners_attached_to_body[fastener_id] == -1).any(), (
            "Fastener is already connected to two bodies."
        )
        free_slot = (
            self.fasteners_attached_to_body[fastener_id][0] == -1
        ).int()  # 0 or 1 # bad syntax, but incidentally it works.

        self.fasteners_attached_to_body[fastener_id][free_slot] = self.body_indices[
            body_name
        ]

        return self

    def disconnect(self, fastener_id: int, disconnected_body: str):
        body_id = self.body_indices[disconnected_body]

        # if (self.fasteners_attached_to_body[fastener_id]>0).all():
        #     # both slots are occupied, so we need to remove the edge.
        # NOTE: let us not have edge index before export at all. it is unnecessary.

        matching_mask = self.fasteners_attached_to_body[fastener_id] == body_id
        assert matching_mask.any(), (
            f"Body {disconnected_body} not attached to fastener {fastener_id}"
        )
        assert not matching_mask.all(), (
            f"Body {disconnected_body} attached to both slots of fastener {fastener_id}, which can not happen"
        )
        # ^ actually it can, but should not
        self.fasteners_attached_to_body[fastener_id][matching_mask] = -1
        return self

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
        assert len(self.body_indices.keys()) > 0, "Physical state must not be empty."
        assert len(other.body_indices.keys()) > 0, (
            "Compared physical state must not be empty."
        )
        assert set(self.body_indices.keys()) == set(other.body_indices.keys()), (
            "Compared physical states must have equal bodies in them."
        )
        # Get node differences
        body_diff, body_diff_count = _diff_body_features(self, other)

        # Get edge differences
        fastener_diff, fastener_diff_count = _diff_fastener_features(self, other)

        # Calculate total differences
        total_diff_count = body_diff_count + fastener_diff_count

        # Create diff graph with same nodes as original
        diff_graph = Data()
        num_nodes = len(self.body_indices.keys())

        # Node features
        diff_graph.position = torch.zeros((num_nodes, 3), device=self.device)
        diff_graph.quat = torch.zeros((num_nodes, 4), device=self.device)
        diff_graph.node_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device
        )

        changed_indices = body_diff["changed_indices"].to(self.device)
        # Store full per-node diff tensors (zeros for unchanged nodes)
        diff_graph.position = body_diff["pos_diff"].to(self.device)
        diff_graph.quat = body_diff["quat_diff"].to(self.device)
        diff_graph.node_mask[changed_indices] = True

        # Edge features and mask
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_attr = torch.empty(
            (0, 3), device=self.device
        )  # [is_added, is_removed, is_changed]
        edge_mask = torch.empty((0,), dtype=torch.bool, device=self.device)

        if fastener_diff["added"] or fastener_diff["removed"]:
            added_edges = (
                torch.tensor(fastener_diff["added"], device=self.device).t()
                if fastener_diff["added"]
                else torch.empty((2, 0), device=self.device, dtype=torch.long)
            )
            removed_edges = (
                torch.tensor(fastener_diff["removed"], device=self.device).t()
                if fastener_diff["removed"]
                else torch.empty((2, 0), device=self.device, dtype=torch.long)
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
        diff_graph.num_nodes = len(self.body_indices)
        # ^ could be shape of positions... could be shape too.

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

    def _update_connector_def_pos(
        self,
        name: str,
        position_sim: torch.Tensor,
        rotation: torch.Tensor,
        connector_position_relative_to_center: torch.Tensor,
    ):
        assert name.endswith(("_male@connector", "_female@connector")), (
            f"Connector {name} must end with _male@connector or _female@connector."
        )
        assert (connector_position_relative_to_center < 5).all(), (
            f"Likely dimension error: it is unlikely that a connector is further than 5m from the center of the "
            f"part. Failed at {name} with position {connector_position_relative_to_center}"
        )
        assert position_sim.ndim == 2 and rotation.ndim == 2, (
            f"Position and rotation must be 2D tensors, got {position_sim.shape} and {rotation.shape}"
        )  # 2dim because it's just for convenience.
        assert connector_position_relative_to_center.ndim == 1, (
            f"Connector position relative to center must be 1D tensor, got {connector_position_relative_to_center.shape}"
        )
        connector_pos = get_connector_pos(
            position_sim,
            rotation,
            connector_position_relative_to_center.to(self.device).unsqueeze(0),
        ).squeeze(0)

        # For connectors, the body index is the connector itself since connectors are registered as bodies
        if name not in self.body_indices:
            raise ValueError(
                f"Connector body {name} not found in body_indices. Available: {list(self.body_indices.keys())}"
            )

        body_idx = self.body_indices[name]

        if name.endswith("_male@connector"):
            # Check if connector already exists
            if name in self.connector_indices_from_name:
                # Update existing connector
                connector_idx = self.connector_indices_from_name[name]
                self.male_connector_positions[connector_idx] = connector_pos
            else:
                # Add new connector
                connector_idx = len(self.male_connector_positions)
                self.connector_indices_from_name[name] = connector_idx
                self.male_connector_batch = torch.cat(
                    [
                        self.male_connector_batch,
                        torch.tensor([body_idx], dtype=torch.long, device=self.device),
                    ]
                )
                self.male_connector_positions = torch.cat(
                    [
                        self.male_connector_positions.to(self.device),
                        connector_pos.unsqueeze(0),
                    ]
                )
        else:
            # Check if connector already exists
            if name in self.connector_indices_from_name:
                # Update existing connector
                connector_idx = self.connector_indices_from_name[name]
                self.female_connector_positions[connector_idx] = connector_pos
            else:
                # Add new connector
                connector_idx = len(self.female_connector_positions)
                self.connector_indices_from_name[name] = connector_idx
                self.female_connector_batch = torch.cat(
                    [
                        self.female_connector_batch,
                        torch.tensor([body_idx], dtype=torch.long, device=self.device),
                    ]
                )
                self.female_connector_positions = torch.cat(
                    [
                        self.female_connector_positions.to(self.device),
                        connector_pos.unsqueeze(0),
                    ]
                )
        return self

    def _build_fastener_edge_attr(self):
        # cat shapes: 1,1,3,4,2 = 11
        # expected shape: [num_fasteners, 11]
        return torch.cat(
            [
                self.fasteners_diam.unsqueeze(-1),  #
                self.fasteners_length.unsqueeze(-1),
                self.fasteners_pos,
                self.fasteners_quat,
                (
                    self.fasteners_attached_to_body != -1
                ).float(),  # if attached at a or b.
            ],
            dim=1,
        )


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
    data_a: "PhysicalState", data_b: "PhysicalState"
) -> tuple[dict, int]:
    """Compare fastener features between two PhysicalState instances."""

    # FIXME: this does not check for (graph) edge feature difference.
    # Stack edge pairs for easy comparison
    def to_sorted_tuple_tensor(edge_index):
        sorted_idx = edge_index.sort(dim=0)[0]
        return sorted_idx.transpose(-2, -1)

    edges_a = to_sorted_tuple_tensor(data_a.edge_index)
    edges_b = to_sorted_tuple_tensor(data_b.edge_index)

    a_set = set(map(tuple, edges_a.tolist()))
    b_set = set(map(tuple, edges_b.tolist()))

    added = list(b_set - a_set)  # well, this is sloppy, but let it be.
    removed = list(a_set - b_set)

    return {"added": added, "removed": removed}, len(added) + len(removed)


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
    physical_states: "PhysicalState",
    names: list[str],  # List of body names to register across all environments
    positions: torch.Tensor,  # [B, num_bodies, 3] positions for bodies across environments
    rotations: torch.Tensor,  # [B, num_bodies, 4] quaternions for bodies across environments
    fixed: torch.Tensor,  # [B, num_bodies]
    connector_position_relative_to_center: torch.Tensor
    | None = None,  # [num_bodies, 3] for connectors, can contain NaN for non-connectors
    max_bounds: torch.Tensor = torch.tensor([0.32, 0.32, 0.64]),
    min_bounds: torch.Tensor = torch.tensor([-0.32, -0.32, 0.0]),
) -> "PhysicalState":
    """Register multiple bodies across multiple environments using tensor operations.

    Args:
        physical_states: Batched PhysicalState to update
        names: List of body names to register across all environments
        positions: Tensor of positions [B, num_bodies, 3] for bodies across environments
        rotations: Tensor of quaternions [B, num_bodies, 4] for bodies across environments
        fixed: Tensor [B, num_bodies]
        connector_position_relative_to_center: Relative positions for connectors [num_bodies, 3], NaN for non-connectors
        max_bounds: Maximum position bounds
        min_bounds: Minimum position bounds
    """
    B = (
        physical_states.batch_size[0]
        if hasattr(physical_states, "batch_size") and physical_states.batch_size
        else positions.shape[0]
    )
    num_bodies = len(names)
    device = (
        physical_states.device
        if hasattr(physical_states, "device")
        else positions.device
    )

    assert B > 0 and num_bodies > 0

    # Validate inputs
    assert positions.shape == (B, num_bodies, 3), (
        f"Expected positions shape [{B}, {num_bodies}, 3], got {positions.shape}"
    )
    assert rotations.shape == (B, num_bodies, 4), (
        f"Expected rotations shape [{B}, {num_bodies}, 4], got {rotations.shape}"
    )

    # Handle fixed parameter - can be [B, num_bodies] or [num_bodies]
    if isinstance(fixed, torch.Tensor):
        if fixed.shape == (num_bodies,):
            # Broadcast to [B, num_bodies]
            fixed = fixed.unsqueeze(0).expand(B, num_bodies)
        elif fixed.shape == (B, num_bodies):
            pass  # Already correct shape
        else:
            raise ValueError(
                f"Expected fixed shape [{num_bodies}] or [{B}, {num_bodies}], got {fixed.shape}"
            )
    else:
        raise ValueError("fixed must be a torch.Tensor")

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
        assert name not in physical_states.body_indices, (
            f"Body {name} already registered"
        )
    assert isinstance(physical_states.body_indices, dict)

    # Update body indices for all bodies
    start_body_idx = len(physical_states.body_indices)
    for i, name in enumerate(names):
        body_idx = start_body_idx + i
        physical_states.body_indices[name] = body_idx
        physical_states.inverse_body_indices[body_idx] = name

    # Set tensors directly for all environments (register_bodies_batch is called once)
    fastener_count = torch.zeros((B, num_bodies), dtype=torch.int8, device=device)

    physical_states.position = positions
    physical_states.quat = rotations
    physical_states.fixed = fixed
    physical_states.count_fasteners_held = fastener_count

    # Handle connectors using the batched connector update function
    if connector_position_relative_to_center is not None:
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
            connector_rel_positions = connector_position_relative_to_center[
                connector_indices
            ]  # [num_connectors, 3]

            # Validate that connector positions are not NaN
            for i, name in enumerate(connector_names):
                connector_rel_pos = connector_rel_positions[i]
                if name.endswith(("_male@connector", "_female@connector")):
                    assert not torch.isnan(connector_rel_pos).any(), (
                        f"Connector {name} must have valid connector_position_relative_to_center, got NaN"
                    )

            # Use the batched connector update function
            male_connector_positions, female_connector_positions = (
                update_connector_def_pos_batch(
                    connector_positions,
                    connector_rotations,
                    connector_rel_positions,
                    connector_names,
                )
            )

            # Set connector positions and indices directly
            physical_states.male_connector_positions = male_connector_positions
            physical_states.female_connector_positions = female_connector_positions

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
                    physical_states.body_indices[name] for name in male_names
                ]
                # Create batched tensor with shape [B, num_male_connectors]
                male_batch_tensor = torch.tensor(
                    male_body_indices, dtype=torch.long, device=device
                )
                # Expand to batch dimension if needed
                if len(male_batch_tensor.shape) == 1:
                    male_batch_tensor = male_batch_tensor.unsqueeze(0).expand(B, -1)
                physical_states.male_connector_batch = male_batch_tensor
                # Update connector indices for all batch elements
                for batch_idx in range(B):
                    for i, name in enumerate(male_names):
                        physical_states[batch_idx].connector_indices_from_name[name] = i
            else:
                # Set empty batched tensor
                physical_states.male_connector_batch = torch.empty(
                    (B, 0), dtype=torch.long, device=device
                )

            # Set up female connector batch and indices
            if female_indices:
                female_names = [connector_names[i] for i in female_indices]
                female_body_indices = [
                    physical_states.body_indices[name] for name in female_names
                ]
                # Create batched tensor with shape [B, num_female_connectors]
                female_batch_tensor = torch.tensor(
                    female_body_indices, dtype=torch.long, device=device
                )
                # Expand to batch dimension if needed
                if len(female_batch_tensor.shape) == 1:
                    female_batch_tensor = female_batch_tensor.unsqueeze(0).expand(B, -1)
                physical_states.female_connector_batch = female_batch_tensor
                # Update connector indices for all batch elements
                for batch_idx in range(B):
                    for i, name in enumerate(female_names):
                        physical_states[batch_idx].connector_indices_from_name[name] = i
            else:
                # Set empty batched tensor
                physical_states.female_connector_batch = torch.empty(
                    (B, 0), dtype=torch.long, device=device
                )

    return physical_states


def register_fasteners_batch(
    physical_states: "PhysicalState",
    fastener_pos: torch.Tensor,
    fastener_quat: torch.Tensor,
    fastener_init_hole_a: torch.Tensor,
    fastener_init_hole_b: torch.Tensor,
    fastener_compound_names: list[str],
) -> "PhysicalState":
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
    assert B is not None and len(physical_states.batch_size) >= 1
    num_fasteners = len(fastener_compound_names)
    num_bodies = len(physical_states.body_indices)
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

    device = physical_states.device if hasattr(physical_states, "device") else "cuda"
    physical_states.fasteners_diam = torch.full(
        (B, num_fasteners), fill_value=-1.0, dtype=torch.float32, device=device
    )
    physical_states.fasteners_length = torch.full(
        (B, num_fasteners), fill_value=-1.0, dtype=torch.float32, device=device
    )  # fill with -1 just in case
    for i, name in enumerate(fastener_compound_names):
        diam, length = get_fastener_params_from_name(name)
        # get_fastener_params_from_name returns values in millimeters.
        # Convert to meters for consistency with PhysicalState units.
        physical_states.fasteners_diam[:, i] = torch.tensor(
            diam / 1000.0, dtype=torch.float32, device=device
        )
        physical_states.fasteners_length[:, i] = torch.tensor(
            length / 1000.0, dtype=torch.float32, device=device
        )
    
    # Runtime guardrails to catch unit mistakes (e.g., mm accidentally stored as meters)
    assert (physical_states.fasteners_diam > 0).all(), "Fastener diameters must be positive (meters)."
    # Typical fastener diameters are < 0.05 m (50 mm); use 0.1 m as a generous upper bound
    assert (
        physical_states.fasteners_diam.max() < 0.1
    ), "Fastener diameters look too large; expected meters. Did you forget mm->m conversion?"
    assert (physical_states.fasteners_length > 0).all(), "Fastener lengths must be positive (meters)."
    # Typical fastener lengths are < 0.25 m (250 mm); use 0.5 m as a generous upper bound
    assert (
        physical_states.fasteners_length.max() < 0.5
    ), "Fastener lengths look too large; expected meters. Did you forget mm->m conversion?"

    assert physical_states.part_hole_batch is not None, (
        "Part hole batch must be set before registering fasteners."
    )
    assert all(
        name not in physical_states.body_indices for name in fastener_compound_names
    ), (
        f"Fasteners can't be registered as bodies! Fastener (compound!) names: {fastener_compound_names}"
    )

    assert (
        (fastener_init_hole_a >= -1)
        & (fastener_init_hole_a < physical_states.part_hole_batch.shape[1])
    ).all(), (
        f"fastener_init_hole_a are out of range. Num holes: {physical_states.part_hole_batch.shape[0]}. "
        f"Min: {physical_states.part_hole_batch.min()}. Max: {physical_states.part_hole_batch.max()}. All: {fastener_init_hole_a}"
    )
    assert (
        (fastener_init_hole_b >= -1)
        & (fastener_init_hole_b < physical_states.part_hole_batch.shape[1])
    ).all(), (
        f"fastener_init_hole_b are out of range. Num holes: {physical_states.part_hole_batch.shape[0]}. "
        f"Min: {physical_states.part_hole_batch.min()}. Max: {physical_states.part_hole_batch.max()}. All: {fastener_init_hole_b}"
    )

    # Update PhysicalState tensors for this environment
    physical_states.fasteners_pos = fastener_pos.to(device)
    physical_states.fasteners_quat = fastener_quat.to(device)
    physical_states.fasteners_attached_to_hole = torch.stack(
        [fastener_init_hole_a, fastener_init_hole_b], dim=-1
    )
    empty = physical_states.fasteners_attached_to_hole == -1
    physical_states.fasteners_attached_to_body = physical_states.part_hole_batch[
        0, physical_states.fasteners_attached_to_hole
    ]
    physical_states.fasteners_attached_to_body[empty] = -1

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

    return physical_states


def update_connector_def_pos_batch(
    positions: torch.Tensor,  # [B, num_connectors, 3] positions for connectors across environments
    rotations: torch.Tensor,  # [B, num_connectors, 4] quaternions for connectors across environments
    connector_positions_relative_to_center: torch.Tensor,  # [num_connectors, 3] relative positions
    names: list[str],  # List of connector names
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update connector positions in batch for multiple environments.

    Args:
        positions: Tensor of positions [B, num_connectors, 3] for connectors across environments
        rotations: Tensor of quaternions [B, num_connectors, 4] for connectors across environments
        connector_positions_relative_to_center: Relative positions [num_connectors, 3]
        names: List of connector names

    Returns:
        Tuple of (male_connector_positions, female_connector_positions)
        - male_connector_positions: [B, num_male_connectors, 3]
        - female_connector_positions: [B, num_female_connectors, 3]
    """
    B, num_connectors, _ = positions.shape
    device = positions.device

    # Validate inputs
    assert rotations.shape == (B, num_connectors, 4), (
        f"Expected rotations shape [{B}, {num_connectors}, 4], got {rotations.shape}"
    )
    assert connector_positions_relative_to_center.shape == (num_connectors, 3), (
        f"Expected connector_positions_relative_to_center shape [{num_connectors}, 3], got {connector_positions_relative_to_center.shape}"
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
    assert (connector_positions_relative_to_center.abs() < 5).all(), (
        "Likely dimension error: it is unlikely that a connector is further than 5m from the center of the part."
    )

    # Move to device
    connector_positions_relative_to_center = connector_positions_relative_to_center.to(
        device
    )

    # Calculate connector positions for all environments and connectors
    # We need to expand relative positions to match batch dimension
    connector_rel_expanded = connector_positions_relative_to_center.unsqueeze(0).expand(
        B, -1, -1
    )  # [B, num_connectors, 3]

    # Calculate connector positions using get_connector_pos for each connector
    all_connector_positions = torch.zeros_like(positions)  # [B, num_connectors, 3]

    for i in range(num_connectors):
        connector_pos = get_connector_pos(
            positions[:, i],  # [B, 3]
            rotations[:, i],  # [B, 4]
            connector_rel_expanded[:, i],  # [B, 3]
        )
        all_connector_positions[:, i] = connector_pos

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
        male_connector_positions = all_connector_positions[
            :, male_indices
        ]  # [B, num_male, 3]
    else:
        male_connector_positions = torch.empty((B, 0, 3), device=device)

    if female_indices:
        female_connector_positions = all_connector_positions[
            :, female_indices
        ]  # [B, num_female, 3]
    else:
        female_connector_positions = torch.empty((B, 0, 3), device=device)

    return male_connector_positions, female_connector_positions
