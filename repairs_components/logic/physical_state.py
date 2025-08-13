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
    # --- bodies ---

    fixed: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.bool)
    )
    # body_indices and inverse_body_indices
    body_indices: dict[str, int] = field(default_factory=dict)
    inverse_body_indices: dict[int, str] = field(default_factory=dict)
    permanently_constrained_parts: list[list[str]] = field(default_factory=list)
    """List of lists of permanently constrained parts (linked_groups from EnvSetup)"""

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
    part_hole_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    "Part index for every hole. Equal over the batch. Shape: (H)"
    hole_indices_from_name: dict[str, int] = field(default_factory=dict)
    """Hole indices per part name."""

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
        assert (self.hole_depth > 0).all(), "Hole depths must be positive."
        assert self.hole_is_through.shape == (self.hole_count,), (
            "Hole is through must have shape (H,)"
        )


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
        edge_attr = self._build_fastener_edge_attr(physical_info)
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
            num_nodes=len(physical_info.body_indices),
            global_feat=global_feat_export,
        )
        # print("debug: graph global feat shape", graph.global_feat.shape)
        return graph

        # @deprecated("Use register_bodies_batch instead.")
        # def register_body(
        #     self,
        #     name: str,
        #     position: tuple,
        #     rotation: tuple,
        #     fixed: bool = False,
        #     rot_as_quat: bool = False,  # mostly for convenience in testing
        #     _expect_unnormalized_coordinates: bool = True,  # mostly for tests...
        #     terminal_position_relative_to_center: torch.Tensor | None = None,
        #     max_bounds: torch.Tensor = torch.tensor([0.32, 0.32, 0.64]),
        #     min_bounds: torch.Tensor = torch.tensor([-0.32, -0.32, 0.0]),
        # ):
        #     assert name not in physical_state_info.body_indices, (
        #         f"Body {name} already registered"
        #     )
        #     assert name.endswith(("@solid", "@connector", "@fixed_solid")), (
        #         f"Body name must end with @solid, @connector or @fixed_solid. Got {name}"
        #     )  # note: fasteners don't go here, they go in `register_fastener`.
        #     assert len(position) == 3, f"Position must be 3D vector, got {position}"

        #     min_bounds = min_bounds.to(self.device)
        #     max_bounds = max_bounds.to(self.device)

        #     # position_sim because that's what we put into sim state.
        #     if _expect_unnormalized_coordinates:
        #         position_sim = compound_pos_to_sim_pos(
        #             torch.tensor(position, device=self.device).unsqueeze(0)
        #         )
        #     else:  # else to genesis (note: flip the var name to normalized coords.)
        #         position_sim = torch.tensor(position, device=self.device).unsqueeze(0)

        #     assert (position_sim >= min_bounds).all() and (
        #         position_sim <= max_bounds
        #     ).all(), (
        #         f"Expected register position to be in [{min_bounds.tolist()}, {max_bounds.tolist()}], got {(position_sim.tolist())} at body {name}"
        #     )
        #     if not rot_as_quat:
        #         assert len(rotation) == 3, (
        #             f"Rotation must be 3D vector, got {rotation}. If you want to pass a quaternion, set rot_as_quat to True."
        #         )
        #         rotation = euler_deg_to_quat_wxyz(torch.tensor(rotation))
        #     else:
        #         rotation = sanitize_quaternion(rotation)
        #     rotation = rotation.to(self.device).unsqueeze(0)

        #     idx = len(physical_state_info.body_indices)
        #     physical_info.body_indices[name] = idx
        #     physical_info.inverse_body_indices[idx] = name

        #     # Update dataclass fields directly
        #     self.position = torch.cat([self.position, position_sim], dim=0)
        #     self.quat = torch.cat([self.quat, rotation], dim=0)
        #     self.count_fasteners_held = torch.cat(
        #         [
        #             self.count_fasteners_held,
        #             torch.zeros(1, dtype=torch.int8, device=self.device),
        #         ],
        #         dim=0,
        #     )
        #     physical_state.fixed = torch.cat(
        #         [
        #             self.fixed,
        #             torch.tensor([fixed], dtype=torch.bool, device=self.device),
        #         ],
        #         dim=0,
        #     )  # maybe will put this into hint instead as -1s or something.

        #     # handle male and female connector positions.
        #     if name.endswith("@connector"):
        #         assert terminal_position_relative_to_center is not None, (
        #             f"Connector {name} must have a connector position relative to center."
        #         )
        #         self._update_terminal_def_pos(
        #             name, position_sim, rotation, terminal_position_relative_to_center
        #         )
        #     return self

        # @deprecated("Use update_bodies_batch instead.")
        # def update_body(
        #     self,
        #     name: str,
        #     position: tuple,
        #     rotation: tuple,
        #     terminal_position_relative_to_center: torch.Tensor | None = None,
        # ):
        #     "Note: expects normalized coordinates."
        #     assert name in physical_state_info.body_indices, (
        #         f"Body {name} not registered. Registered bodies: {physical_state_info.body_indices.keys()}"
        #     )
        #     pos_tensor = torch.tensor(position, device=self.device)
        #     assert (
        #         pos_tensor >= torch.tensor([-0.32, -0.32, 0.0]).to(self.device)
        #     ).all() and (
        #         pos_tensor <= torch.tensor([0.32, 0.32, 0.64]).to(self.device)
        #     ).all(), (
        #         f"Position {position} out of bounds. Expected [-0.32, 0.32] for update."
        #     )
        #     pos_tensor = pos_tensor
        #     rotation = sanitize_quaternion(rotation).to(device=self.device)
        #     if self.fixed[physical_state_info.body_indices[name]]:
        #         # assert torch.isclose(
        #         #     torch.tensor(position, device=self.device),
        #         #     self.position[physical_state_info.body_indices[name]],
        #         #     atol=1e-6,
        #         # ).all(), f"Body {name} is fixed and cannot be moved."
        #         # FIXME: fix
        #         return

        #     idx = physical_state_info.body_indices[name]
        #     self.position[idx] = pos_tensor
        #     self.quat[idx] = rotation

        #     if name.endswith("@connector"):
        #         assert terminal_position_relative_to_center is not None, (
        #             f"Connector {name} must have a connector position relative to center."
        #         )
        #         self._update_terminal_def_pos(
        #             name,
        #             pos_tensor.unsqueeze(0),
        #             rotation.unsqueeze(0),
        #             terminal_position_relative_to_center,
        #         )

        return self  # maybe that would fix view issues.

    # @deprecated("Use register_fasteners_batch instead.")
    # def register_fastener(self, fastener: Fastener):
    #     """A fastener method to register fasteners and add all necessary components.
    #     Handles constraining to bodies and adding to graph.

    #     Args:
    #     - count_holes: Number of holes in the batch. If None, uses the number of holes in the batch (necessary during initial population)
    #     """
    #     assert fastener.name not in physical_state_info.body_indices, (
    #         f"Fasteners can't be registered as bodies! Attempted at {fastener.name}"
    #     )
    #     assert self.part_hole_batch is not None, (
    #         "Part hole batch must be set before registering fasteners."
    #     )

    #     # Convert hole IDs to body names using hole_indices_batch
    #     initial_body_a = None
    #     initial_body_b = None

    #     if fastener.initial_hole_id_a is not None:
    #         assert 0 <= fastener.initial_hole_id_a < self.part_hole_batch.shape[0], (
    #             f"Hole ID {fastener.initial_hole_id_a} is out of range. Num holes: {self.part_hole_batch.shape[0]}"
    #         )
    #         body_idx_a = int(self.part_hole_batch[fastener.initial_hole_id_a].item())
    #         initial_body_a = physical_state_info.inverse_body_indices[body_idx_a]

    #     if fastener.initial_hole_id_b is not None:
    #         assert 0 <= fastener.initial_hole_id_b < self.part_hole_batch.shape[0], (
    #             f"Hole ID {fastener.initial_hole_id_b} is out of range. Num holes: {self.part_hole_batch.shape[0]}"
    #             f"Hole ID {fastener.initial_hole_id_b} is out of range. Available holes: 0-{self.part_hole_batch.shape[0] - 1}"
    #         )
    #         body_idx_b = int(self.part_hole_batch[fastener.initial_hole_id_b].item())
    #         initial_body_b = physical_state_info.inverse_body_indices[body_idx_b]

    #     fastener_id = len(self.fasteners_pos)
    #     self.fasteners_pos = torch.cat(
    #         [self.fasteners_pos, torch.zeros((1, 3), device=self.device)], dim=0
    #     )

    #     self.fasteners_quat = torch.cat(
    #         [self.fasteners_quat, torch.zeros((1, 4), device=self.device)], dim=0
    #     )

    #     physical_info.fasteners_diam = torch.cat(
    #         [
    #             physical_info.fasteners_diam,
    #             torch.tensor(fastener.diameter, device=self.device).unsqueeze(0),
    #         ],
    #         dim=0,
    #     )
    #     physical_info.fasteners_length = torch.cat(
    #         [
    #             physical_info.fasteners_length,
    #             torch.tensor(fastener.length, device=self.device).unsqueeze(0),
    #         ],
    #         dim=0,
    #     )
    #     physical_info.fasteners_attached_to_body = torch.cat(
    #         [
    #             physical_info.fasteners_attached_to_body,
    #             torch.full((1, 2), -1, device=self.device),
    #         ],
    #         dim=0,
    #     )  # note: technically fasteners_attached_to_body is a junk value as it simply repeats fasteners_attached_to_hole except with part IDs.
    #     physical_info.fasteners_attached_to_hole = torch.cat(
    #         [
    #             physical_info.fasteners_attached_to_hole,
    #             torch.full((1, 2), -1, device=self.device),
    #         ],
    #         dim=0,
    #     )

    #     if initial_body_a is not None:
    #         self.connect_fastener_to_one_body(fastener_id, initial_body_a)
    #     if initial_body_b is not None:
    #         self.connect_fastener_to_one_body(fastener_id, initial_body_b)
    #     return self  # maybe that would fix view issues.

    # TODO deprecate and set to functions
    def connect_fastener_to_one_body(
        self, fastener_id: int, body_name: str, env_idx: torch.Tensor
    ):
        """Connect a fastener to a body. Used during screw-in and initial construction."""
        # FIXME: but where to get/store fastener ids I'll need to think.
        assert (self.fasteners_attached_to_body[env_idx, fastener_id] == -1).any(), (
            "Fastener is already connected to two bodies."
        )
        # Choose slot 0 if it is free, otherwise use slot 1
        slot0_free = self.fasteners_attached_to_body[env_idx, fastener_id, 0] == -1
        # Ensure we have a Python bool even if this is a 0-dim tensor
        if isinstance(slot0_free, torch.Tensor):
            slot0_free = bool(slot0_free.item())
        free_slot = 0 if slot0_free else 1

        self.fasteners_attached_to_body[env_idx, fastener_id, free_slot] = (
            physical_state_info.body_indices[body_name]
        )

        return self

    # TODO deprecate and set to functions
    def disconnect(self, fastener_id: int, disconnected_body: str):
        body_id = physical_state_info.body_indices[disconnected_body]

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
        assert self.position.shape[1] > 0, "Physical state must not be empty."
        assert other.position.shape[1] > 0, "Compared physical state must not be empty."
        assert self.position.shape[1] == other.position.shape[1], (
            "Compared physical states must have equal number of bodies."
        )
        # Get node and edge differences
        body_diff, body_diff_count = _diff_body_features(self, other)
        fastener_diff, fastener_diff_count = _diff_fastener_features(self, other)
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

    # def _update_terminal_def_pos(
    #     self,
    #     name: str,
    #     position_sim: torch.Tensor,
    #     rotation: torch.Tensor,
    #     terminal_position_relative_to_center: torch.Tensor,
    # ):
    #     assert name.endswith(("_male@connector", "_female@connector")), (
    #         f"Connector {name} must end with _male@connector or _female@connector."
    #     )
    #     assert (terminal_position_relative_to_center < 5).all(), (
    #         f"Likely dimension error: it is unlikely that a connector is further than 5m from the center of the "
    #         f"part. Failed at {name} with position {terminal_position_relative_to_center}"
    #     )
    #     assert position_sim.ndim == 2 and rotation.ndim == 2, (
    #         f"Position and rotation must be 2D tensors, got {position_sim.shape} and {rotation.shape}"
    #     )  # 2dim because it's just for convenience.
    #     assert terminal_position_relative_to_center.ndim == 1, (
    #         f"Connector position relative to center must be 1D tensor, got {terminal_position_relative_to_center.shape}"
    #     )
    #     terminal_pos = get_connector_pos(
    #         position_sim,
    #         rotation,
    #         terminal_position_relative_to_center.to(self.device).unsqueeze(0),
    #     ).squeeze(0)

    #     # For connectors, the body index is the connector itself since connectors are registered as bodies
    #     if name not in physical_info.body_indices:
    #         raise ValueError(
    #             f"Connector body {name} not found in body_indices. Available: {list(physical_state_info.body_indices.keys())}"
    #         )

    #     body_idx = physical_info.body_indices[name]

    #     if name.endswith("_male@connector"):
    #         # Check if connector already exists
    #         if name in physical_info.terminal_indices_from_name:
    #             # Update existing connector
    #             terminal_idx = physical_info.terminal_indices_from_name[name]
    #             physical_info.male_terminal_positions[terminal_idx] = terminal_pos
    #         else:
    #             # Add new connector
    #             terminal_idx = len(self.male_terminal_positions)
    #             physical_info.terminal_indices_from_name[name] = terminal_idx
    #             physical_info.male_terminal_batch = torch.cat(
    #                 [
    #                     self.male_terminal_batch,
    #                     torch.tensor([body_idx], dtype=torch.long, device=self.device),
    #                 ]
    #             )
    #             physical_info.male_terminal_positions = torch.cat(
    #                 [
    #                     self.male_terminal_positions.to(self.device),
    #                     terminal_pos.unsqueeze(0),
    #                 ]
    #             )
    #     else:
    #         # Check if connector already exists
    #         if name in physical_info.terminal_indices_from_name:
    #             # Update existing connector
    #             terminal_idx = physical_info.terminal_indices_from_name[name]
    #             physical_info.female_terminal_positions[terminal_idx] = terminal_pos
    #         else:
    #             # Add new connector
    #             terminal_idx = len(self.female_terminal_positions)
    #             physical_info.terminal_indices_from_name[name] = terminal_idx
    #             physical_info.female_terminal_batch = torch.cat(
    #                 [
    #                     physical_info.female_terminal_batch,
    #                     torch.tensor([body_idx], dtype=torch.long, device=self.device),
    #                 ]
    #             )
    #             physical_info.female_terminal_positions = torch.cat(
    #                 [
    #                     physical_info.female_terminal_positions.to(self.device),
    #                     terminal_pos.unsqueeze(0),
    #                 ]
    #             )
    #     return self

    def _build_fastener_edge_attr(self, physical_info: PhysicalStateInfo):
        # cat shapes: 1,1,3,4,2 = 11
        # expected shape: [num_fasteners, 11]
        return torch.cat(
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
    """Compare fastener features between two PhysicalState instances.

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

    # Reconstruct edges from fastener attachments.
    # We consider an edge to exist when a fastener is attached to two bodies (both slots are valid).
    # Each edge is undirected and represented as a sorted (u, v) tuple of body indices.

    # Normalize potentially batched tensors:
    # - For attachments/poses that can vary per environment, flatten across batch to union info across B
    # - For scalar fastener params (diam/length) that are equal over batch, validate equality and reduce
    def _unbatch_fastener_tensors(ps: PhysicalState):
        atb = ps.fasteners_attached_to_body
        ath = ps.fasteners_attached_to_hole
        # diam = pi.fasteners_diam
        # length = pi.fasteners_length
        pos = ps.fasteners_pos
        quat = ps.fasteners_quat

        if atb.ndim == 3:  # [B, N, 2] -> [B*N, 2]
            atb = atb.reshape(-1, 2)
        if ath.ndim == 3:  # [B, N, 2] -> [B*N, 2]
            ath = ath.reshape(-1, 2)
        # if diam.ndim == 2:  # [B, N] -> [B*N] (validate equality across B)
        #     assert torch.allclose(diam, diam[:1].expand_as(diam)), (
        #         "fasteners_diam must be equal across batch"
        #     )
        #     diam = diam.reshape(-1)
        # if length.ndim == 2:  # [B, N] -> [B*N] (validate equality across B)
        #     assert torch.allclose(length, length[:1].expand_as(length)), (
        #         "fasteners_length must be equal across batch"
        #     )
        #     length = length.reshape(-1)
        if pos.ndim == 3:  # [B, N, 3] -> [B*N, 3]
            pos = pos.reshape(-1, 3)
        if quat.ndim == 3:  # [B, N, 4] -> [B*N, 4]
            quat = quat.reshape(-1, 4)

        return atb, ath, pos, quat

    a_atb, a_ath, a_pos, a_quat = _unbatch_fastener_tensors(data_a)
    b_atb, b_ath, b_pos, b_quat = _unbatch_fastener_tensors(data_b)

    def attachments_to_edge_set(attached_to_body: torch.Tensor) -> set[tuple[int, int]]:
        if attached_to_body.numel() == 0:
            return set()
        # attached_to_body: [num_fasteners, 2] with -1 for unattached
        # Select rows where both bodies are valid
        valid = (attached_to_body[:, 0] >= 0) & (attached_to_body[:, 1] >= 0)
        pairs = attached_to_body[valid].to(torch.long)
        if pairs.numel() == 0:
            return set()
        # Sort each pair to make edges undirected-consistent
        u = torch.minimum(pairs[:, 0], pairs[:, 1])
        v = torch.maximum(pairs[:, 0], pairs[:, 1])
        edges = torch.stack([u, v], dim=1)
        return set(map(tuple, edges.tolist()))

    a_set = attachments_to_edge_set(a_atb)
    b_set = attachments_to_edge_set(b_atb)

    # Compute differences for connectivity
    added = list(b_set - a_set)
    removed = list(a_set - b_set)

    # Attribute diffs: match fasteners by hole pairs (assumes at most one fastener per hole pair)
    def build_holepair_map(
        attached_to_hole: torch.Tensor,
        attached_to_body: torch.Tensor,
        diam: torch.Tensor,
        length: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> dict[tuple[int, int], dict]:
        hole_map: dict[tuple[int, int], dict] = {}
        if attached_to_hole.numel() == 0:
            return hole_map
        valid = (attached_to_hole[:, 0] >= 0) & (attached_to_hole[:, 1] >= 0)
        idxs = torch.nonzero(valid, as_tuple=False).squeeze(1)
        for i in idxs.tolist():
            ha, hb = (
                int(attached_to_hole[i, 0].item()),
                int(attached_to_hole[i, 1].item()),
            )
            key = (ha, hb) if ha <= hb else (hb, ha)
            # Also record body pair for this fastener
            ba, bb = (
                int(attached_to_body[i, 0].item()),
                int(attached_to_body[i, 1].item()),
            )
            bu, bv = (ba, bb) if ba <= bb else (bb, ba)
            # For batched inputs flattened across B, multiple entries for the same
            # hole-pair may appear (from different environments). Prefer the first
            # occurrence to avoid arbitrary overwrites; diam/length are equal across
            # batch by design.
            hole_map.setdefault(
                key,
                {
                    "body_pair": (bu, bv),
                    "diam": float(diam[i].item()),
                    "length": float(length[i].item()),
                    "pos": pos[i].detach(),
                    "quat": quat[i].detach(),
                },
            )
        return hole_map

    a_holes = build_holepair_map(
        a_ath,
        a_atb,
        physical_info.fasteners_diam,
        physical_info.fasteners_length,
        a_pos,
        a_quat,
    )
    b_holes = build_holepair_map(
        b_ath,
        b_atb,
        physical_info.fasteners_diam,
        physical_info.fasteners_length,
        b_pos,
        b_quat,
    )

    common_hole_pairs = set(a_holes.keys()) & set(b_holes.keys())

    changed_edges: list[tuple[int, int]] = []
    diam_changed: list[bool] = []
    length_changed: list[bool] = []
    pos_diffs: list[torch.Tensor] = []
    quat_deltas: list[torch.Tensor] = []

    # thresholds consistent with body diffs
    pos_threshold = 5.0 / 1000
    deg_threshold = 5.0

    for k in common_hole_pairs:
        a_v = a_holes[k]
        b_v = b_holes[k]
        # Use body pair from 'b' (should match 'a')
        changed_edges.append(tuple(b_v["body_pair"]))

        # Diameter/length: scalar compare with small tolerance
        d_diam = abs(b_v["diam"] - a_v["diam"]) > 1e-6
        d_len = abs(b_v["length"] - a_v["length"]) > 1e-6

        # Position: vector diff with threshold
        p_raw = b_v["pos"] - a_v["pos"]
        p_dist = torch.linalg.norm(p_raw)
        p_diff = torch.zeros_like(p_raw)
        if p_dist > pos_threshold:
            p_diff = p_raw

        # Quaternion: use delta and angular threshold
        q_delta = quaternion_delta(
            a_v["quat"].unsqueeze(0), b_v["quat"].unsqueeze(0)
        ).squeeze(0)
        q_changed = ~are_quats_within_angle(
            a_v["quat"].unsqueeze(0),
            b_v["quat"].unsqueeze(0),
            torch.tensor(deg_threshold, device=q_delta.device),
        ).squeeze(0)
        if q_changed.ndim > 0:
            q_changed = bool(q_changed.item())

        diam_changed.append(bool(d_diam))
        length_changed.append(bool(d_len))
        pos_diffs.append(p_diff)
        quat_deltas.append(q_delta if q_changed else torch.zeros_like(q_delta))

    if changed_edges:
        changed_edge_tensor = torch.tensor(changed_edges, dtype=torch.long).t()
        pos_diff_tensor = torch.stack(pos_diffs, dim=0)
        quat_delta_tensor = torch.stack(quat_deltas, dim=0)
    else:
        changed_edge_tensor = torch.empty((2, 0), dtype=torch.long)
        pos_diff_tensor = torch.empty((0, 3))
        quat_delta_tensor = torch.empty((0, 4))

    changed = {
        "changed_edges": changed_edge_tensor,
        "diam_changed": torch.tensor(diam_changed, dtype=torch.bool),
        "length_changed": torch.tensor(length_changed, dtype=torch.bool),
        "pos_diff": pos_diff_tensor,
        "quat_delta": quat_delta_tensor,
    }

    total_count = len(added) + len(removed) + changed_edge_tensor.size(1)
    out = {"added": added, "removed": removed, **changed}
    return out, total_count


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
    physical_states: PhysicalState = torch.stack(
        [PhysicalState(device=device) for _ in range(B)]
    )
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
                if len(male_batch_tensor.shape) == 1:
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
    assert B is not None and len(physical_states.batch_size) == 1
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

    device = physical_states.device if hasattr(physical_states, "device") else "cuda"
    physical_info.fasteners_diam = torch.full(
        (B, num_fasteners), fill_value=-1.0, dtype=torch.float32, device=device
    )
    physical_info.fasteners_length = torch.full(
        (B, num_fasteners), fill_value=-1.0, dtype=torch.float32, device=device
    )  # fill with -1 just in case
    for i, name in enumerate(fastener_compound_names):
        diam, length = get_fastener_params_from_name(name)
        # get_fastener_params_from_name returns values in millimeters.
        # Convert to meters for consistency with PhysicalState units.
        physical_info.fasteners_diam[:, i] = torch.tensor(
            diam / 1000.0, dtype=torch.float32, device=device
        )
        physical_info.fasteners_length[:, i] = torch.tensor(
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
    env_idx: torch.Tensor,
):
    """Connect a fastener to a body. Used during screw-in and initial construction."""
    # TODO: will batch in the near future.
    assert (
        physical_state.fasteners_attached_to_body[env_idx, fastener_id] == -1
    ).any(), "Fastener is already connected to two bodies."
    # Choose slot 0 if it is free, otherwise use slot 1
    slot0_free = (
        physical_state.fasteners_attached_to_body[env_idx, fastener_id, 0] == -1
    )
    # Ensure we have a Python bool even if this is a 0-dim tensor
    if isinstance(slot0_free, torch.Tensor):
        slot0_free = bool(slot0_free.item())
    free_slot = 0 if slot0_free else 1

    physical_state.fasteners_attached_to_body[env_idx, fastener_id, free_slot] = (
        physical_info.body_indices[body_name]
    )

    return physical_state
