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
from repairs_components.geometry.fasteners import Fastener
from torch_geometric.data import Data
from dataclasses import dataclass, field


@dataclass
class PhysicalState:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Graph storing part nodes and fastener edges
    graph: Data = field(default_factory=Data)
    """The graph storing part nodes and fastener edges.
    *Note*: don't use this graph for ML purposes. Use `export_graph` instead.
    
    Graph features:
    - position
    - quat
    - free_fasteners_loc
    - free_fasteners_quat
    - free_fasteners_attached_to
    """  # this is kind of unnecessary... again.
    # TODO encode mass, and possibly velocity.

    body_indices: dict[str, int] = field(default_factory=dict)
    inverse_indices: dict[int, str] = field(default_factory=dict)
    # fastener_ids: dict[int, str] = field(default_factory=dict) # fixme: I don't remember how, but this is unused.
    hole_indices_from_name: dict[str, int] = field(default_factory=dict)
    """Hole indices per part name."""
    hole_indices_batch: torch.Tensor = field(default_factory=torch.empty)
    """Hole indices per part in the batch."""
    hole_positions: torch.Tensor = field(default_factory=torch.empty)  # [H, 3]
    """Hole positions per part, batched with hole_indices_batch."""
    hole_quats: torch.Tensor = field(default_factory=torch.empty)  # [H, 4]
    """Hole quats per part, batched with hole_indices_batch."""

    # Fastener metadata (shared across batch)
    # fastener: dict[str, Fastener] = field(default_factory=dict)
    "When the fastener is inserted only into one body, it is stored here. Otherwise, it is -1."
    # ^ self.fastener deprecated? graph stores everything.

    # # Free fasteners - not attached to two parts, but to one or none
    # free_fasteners_loc: torch.Tensor = field(
    #     default_factory=lambda: torch.zeros((0, 3), dtype=torch.float32)
    # )
    # free_fasteners_quat: torch.Tensor = field(
    #     default_factory=lambda: torch.zeros((0, 4), dtype=torch.float32)
    # )
    # free_fasteners_attached_to: torch.Tensor = field(
    #     default_factory=lambda: torch.zeros((0,), dtype=torch.int8)
    # )

    permanently_constrained_parts: list[list[str]] = field(default_factory=list)
    """List of lists of permanently constrained parts (linked_groups from EnvSetup)"""

    # TODO electronics connection positions?

    def __init__(
        self,
        graph: Data | None = None,
        indices: dict[str, int] | None = None,
        fasteners: dict[str, Fastener] | None = None,
        fastener_id_to_name: dict[int, str] | None = None,
        device: torch.device = torch.device(
            "cpu"
        ),  # should be always on CPU to my understanding. it's not buffer.
    ):
        assert graph is None or (
            graph.fasteners_loc.shape[0]
            == graph.fasteners_quat.shape[0]
            == graph.fasteners_attached_to.shape[0]
        ), "Free fasteners must have the same shape"
        ### unsure end.
        # for empty creation (which is the case in online loading), do not require a graph
        if graph is None:
            self.graph = Data()
            # Initialize graph attributes

            self.device = device  # Set device before initializing tensors
            self.graph.edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            # self.graph.edge_attr = torch.empty(
            #     (0, 12), dtype=torch.float32, device=self.device
            # )  # placeholder size  # note: shouldn't exist, until merge at least.

            # Node attributes
            self.graph.position = torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            )  # note: torch_geomeric conventionally uses `pos` for 2d or 3d positions. You could too.
            self.graph.quat = torch.empty(
                (0, 4), dtype=torch.float32, device=self.device
            )
            self.graph.fixed = torch.empty((0,), dtype=torch.bool, device=self.device)
            self.graph.count_fasteners_held = torch.empty(
                (0,), dtype=torch.int8, device=self.device
            )  # NOTE: 0 info about fasteners?

            self.graph.fasteners_loc = torch.empty(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self.graph.fasteners_quat = torch.empty(
                (0, 4), dtype=torch.float32, device=self.device
            )
            self.graph.fasteners_attached_to = torch.empty(
                (0, 2), dtype=torch.int16, device=self.device
            )  # to which 2 bodies attached. -1 if not attached.

            # note: free_fasteners_id does not go into graph, it is only used for
            # fastener_id_to_name mapping.
            # self.graph.fasteners_id = torch.empty((0,), dtype=torch.int16)
            self.graph.fasteners_diam = torch.empty(
                (0,), dtype=torch.float32, device=self.device
            )
            self.graph.fasteners_length = torch.empty(
                (0,), dtype=torch.float32, device=self.device
            )

            if indices is None:
                self.body_indices = {}
                self.inverse_indices = {}
            else:
                self.body_indices = indices
                self.inverse_indices = {v: k for k, v in indices.items()}
            # self.fastener = {}
        else:
            assert indices is not None, "Indices must be provided if graph is not None"
            # assert fastener_id_to_name is not None, (
            #     "Fastener names must be provided if graph is not None"
            # )
            self.graph = graph
            self.body_indices = indices
            self.inverse_indices = {v: k for k, v in indices.items()}
            self.device = device
            # self.fastener = {}

            assert (
                graph.fasteners_loc is not None
                and graph.fasteners_quat is not None
                and graph.fasteners_attached_to is not None
            ), "Passed graph can't have None fasteners."

            # TODO logic for fastener rebuild...
            # for edge_index, edge_attr in zip(graph.edge_index.t(), graph.edge_attr):
            #     fastener_id = edge_attr[0]  # assume it is the first element
            #     fastener_size = edge_attr[
            #         1
            #     ]  # TODO: should be in separate graphs before merge!
            #     fastener_name = fastener_id_to_name[fastener_id]
            #     connected_to_1 = self.reverse_indices[edge_index[0].item()]
            #     connected_to_2 = self.reverse_indices[edge_index[1].item()]
            #     # fastener = Fastener(  # FIXME: constraint_b_active always True. It is not selected
            #     #     constraint_b_active=True,
            #     #     initial_body_a=connected_to_1,
            #     #     initial_body_b=connected_to_2,
            #     #     name=fastener_name,
            #     #     # fastener_size=fastener_size, # FIXME: size to actual params mapping
            #     # )
            #     # self.fastener[fastener_name] = fastener

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

    def export_graph(self):
        """Export the graph to a torch_geometric Data object usable by ML."""

        # only export fasteners which aren not attached to nothing.
        global_feat_mask = (self.graph.fasteners_attached_to == -1).any(dim=-1)
        global_feat_export = torch.cat(
            [
                self.graph.fasteners_loc[global_feat_mask],
                self.graph.fasteners_quat[global_feat_mask],
                (self.graph.fasteners_attached_to[global_feat_mask] == -1).float(),
                # export which are attached and which not, but not their ids.
            ],
            dim=-1,
        )
        graph = Data(  # expected len of x - 8.
            x=torch.cat(
                [
                    self.graph.position,
                    self.graph.quat,
                    self.graph.count_fasteners_held.float().unsqueeze(-1),
                    # TODO: construct count_fasteners_held on export.
                ],
                dim=1,
            ).bfloat16(),
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,  # e.g. fastener size.
            num_nodes=len(self.body_indices),
            global_feat=global_feat_export,
            # batch=self.graph.batch,
            # global_feat_count=self.graph.fasteners_loc.shape[0],
            # ^export global fastener features as a part of graph.
        )
        # print("debug: graph global feat shape", graph.global_feat.shape)
        return graph

    def register_body(
        self, name: str, position: tuple, rotation: tuple, fixed: bool = False
    ):
        assert name not in self.body_indices, f"Body {name} already registered"
        # assert position.shape == (3,) and rotation.shape == (4,) # was a tensor.
        assert len(position) == 3 and len(rotation) == 3, (
            f"Position must be 3D vector, got {position}"
            f"Rotation must be 3D vector, got {rotation}. Note: transformation to quat already happens in this function."
        )
        assert (
            torch.tensor(position).min() >= 0 and torch.tensor(position).max() <= 640
        ), f"Expected register position to be in [0,640], got {position}"
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_euler("xyz", rotation).as_quat()

        idx = len(self.body_indices)
        self.body_indices[name] = idx
        self.inverse_indices[idx] = name

        # Ensure all graph tensors are on the correct device before concatenation
        self.graph = self.graph.to(self.device)

        self.graph.position = torch.cat(
            [
                self.graph.position,
                compound_pos_to_sim_pos(
                    torch.tensor(position, device=self.device).unsqueeze(0)
                ),
            ],
            dim=0,
        )
        self.graph.quat = torch.cat(
            [self.graph.quat, torch.tensor(rotation, device=self.device).unsqueeze(0)],
            dim=0,
        )
        self.graph.count_fasteners_held = torch.cat(
            [
                self.graph.count_fasteners_held,
                torch.zeros(1, dtype=torch.int8, device=self.device),
            ],
            dim=0,
        )
        # set num_nodes manually because otherwise there is no way for PyG to know the number of nodes.
        self.graph.num_nodes = len(self.body_indices)
        self.graph.fixed = torch.cat(
            [
                self.graph.fixed,
                torch.tensor([fixed], dtype=torch.bool, device=self.device).unsqueeze(
                    0
                ),
            ],
            dim=0,
        )  # maybe will put this into hint instead as -1s or something.

    def update_body(self, name: str, position: tuple, rotation: tuple):
        assert name in self.body_indices, f"Body {name} not registered"
        pos_tensor = torch.tensor(position, device=self.device)
        assert pos_tensor.min() >= -0.32 and pos_tensor.max() <= 0.32, (
            f"Position {position} out of bounds. Expected [-0.32, 0.32] for update."
        )
        if self.graph.fixed[self.body_indices[name]]:
            # assert torch.isclose(
            #     torch.tensor(position, device=self.device),
            #     self.graph.position[self.body_indices[name]],
            #     atol=1e-6,
            # ).all(), f"Body {name} is fixed and cannot be moved."
            # FIXME: fix
            return

        idx = self.body_indices[name]
        self.graph.position[idx] = pos_tensor
        self.graph.quat[idx] = torch.tensor(rotation, device=self.device)

    def register_fastener(self, fastener: Fastener):
        """A fastener method to register fasteners and add all necessary components.
        Handles constraining to bodies and adding to graph."""

        assert fastener.name not in self.body_indices, (
            f"Fasteners can't be registered as bodies!"
        )
        if fastener.initial_body_a is not None:
            assert fastener.initial_body_a in self.body_indices, (
                f"Body {fastener.initial_body_a} marked as connected to fastener {fastener.name} is not registered"
            )
        if fastener.initial_body_b is not None:
            assert fastener.initial_body_b in self.body_indices, (
                f"Body {fastener.initial_body_b} marked as connected to fastener {fastener.name} is not registered"
            )
        fastener_id = len(self.graph.fasteners_loc)
        self.graph.fasteners_loc = torch.cat(
            [self.graph.fasteners_loc, torch.zeros((1, 3), device=self.device)], dim=0
        )
        self.graph.fasteners_quat = torch.cat(
            [self.graph.fasteners_quat, torch.zeros((1, 4), device=self.device)], dim=0
        )
        self.graph.fasteners_attached_to = torch.cat(
            [
                self.graph.fasteners_attached_to,
                torch.full((1, 2), -1, device=self.device),
            ],
            dim=0,
        )
        self.graph.fasteners_diam = torch.cat(
            [
                self.graph.fasteners_diam,
                torch.tensor(fastener.diameter, device=self.device).unsqueeze(0),
            ],
            dim=0,
        )
        self.graph.fasteners_length = torch.cat(
            [
                self.graph.fasteners_length,
                torch.tensor(fastener.length, device=self.device).unsqueeze(0),
            ],
            dim=0,
        )

        if fastener.initial_body_a is not None:
            self.connect_fastener_to_one_body(fastener_id, fastener.initial_body_a)
        if fastener.initial_body_b is not None:
            self.connect_fastener_to_one_body(fastener_id, fastener.initial_body_b)

    def connect_fastener_to_one_body(self, fastener_id: int, body_name: str):
        """Connect a fastener to a body. Used during screw-in and initial construction."""
        # FIXME: but where to get/store fastener ids I'll need to think.
        assert (self.graph.fasteners_attached_to[fastener_id] == -1).any(), (
            "Fastener is already connected to two bodies."
        )
        free_slot = (
            self.graph.fasteners_attached_to[fastener_id][0] == -1
        ).int()  # 0 or 1 # bad syntax, but incidentally it works.

        self.graph.fasteners_attached_to[fastener_id][free_slot] = self.body_indices[
            body_name
        ]

        # design choice - no edge index before export (`export_graph()`).
        # # If both slots were now occupied, add a new edge to edge_index
        # if free_slot == 1:
        #     src, dst = self.graph.fasteners_attached_to[fastener_id]
        #     new_edges = torch.tensor(
        #         [[src, dst], [dst, src]], dtype=torch.long, device=self.device
        #     )
        #     self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)
        #     # edge index and attr will be constructed later.

    def disconnect(self, fastener_id: int, disconnected_body: str):
        body_id = self.body_indices[disconnected_body]

        # if (self.graph.fasteners_attached_to[fastener_id]>0).all():
        #     # both slots are occupied, so we need to remove the edge.
        # NOTE: let us not have edge index before export at all. it is unnecessary.

        matching_mask = self.graph.fasteners_attached_to[fastener_id] == body_id
        assert matching_mask.any(), (
            f"Body {disconnected_body} not attached to fastener {fastener_id}"
        )
        assert not matching_mask.all(), (
            f"Body {disconnected_body} attached to both slots of fastener {fastener_id}, which can not happen"
        )
        # ^ actually it can, but should not
        self.graph.fasteners_attached_to[fastener_id][matching_mask] = -1

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
        num_nodes = self.graph.num_nodes

        # Node features
        diff_graph.position = torch.zeros((num_nodes, 3), device=self.device)
        diff_graph.quat = torch.zeros((num_nodes, 4), device=self.device)
        diff_graph.node_mask = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device
        )

        changed_indices = node_diff["changed_indices"].to(self.device)
        # Store full per-node diff tensors (zeros for unchanged nodes)
        diff_graph.position = node_diff["pos_diff"].to(self.device)
        diff_graph.quat = node_diff["quat_diff"].to(self.device)
        diff_graph.node_mask[changed_indices] = True

        # Edge features and mask
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_attr = torch.empty(
            (0, 3), device=self.device
        )  # [is_added, is_removed, is_changed]
        edge_mask = torch.empty((0,), dtype=torch.bool, device=self.device)

        if edge_diff["added"] or edge_diff["removed"]:
            added_edges = (
                torch.tensor(edge_diff["added"], device=self.device).t()
                if edge_diff["added"]
                else torch.empty((2, 0), device=self.device, dtype=torch.long)
            )
            removed_edges = (
                torch.tensor(edge_diff["removed"], device=self.device).t()
                if edge_diff["removed"]
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

        Args:indices
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

        # FIXME: this does not account for a) batch of environments, b) batch of parts.
        # ^batch of environments would be cool, but batch of parts would barely ever happen.
        return (changed == part_id).any()

    @staticmethod
    def rebuild_from_graph(graph: Data, indices: dict[str, int]) -> "PhysicalState":
        "Rebuild an ElectronicsState from a graph"
        assert graph.num_nodes == len(indices), (
            f"Graph and indices do not match: {graph.num_nodes} != {len(indices)}"
        )
        # kind of pointless method tbh
        # new_graph.position = graph.position
        # new_graph.quat = graph.quat
        # new_graph.count_fasteners_held = graph.count_fasteners_held
        # new_graph.fasteners_loc = graph.fasteners_loc
        # new_graph.fasteners_quat = graph.fasteners_quat
        # new_graph.fasteners_attached_to = graph.fasteners_attached_to
        # note: fasteners are not reconstructed because hopefully, they should not be necessary after offline reconstruction.
        new_state = PhysicalState(graph=graph, indices=indices)
        return new_state


def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a quaternion (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


#FIXME: quat_conj, quat_multiply, quat_angle_diff are equally implemented in `translation.py`
def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.  Inputs (...,4) → output (...,4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quaternion_delta(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    """Returns the quaternion that rotates *q_from* into *q_to* (element-wise)."""
    return _quaternion_multiply(q_to, _quaternion_conjugate(q_from))


def _quaternion_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two quaternions, degrees."""
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1.0, 1.0).abs()
    angle_rad = 2.0 * torch.acos(dot)
    return torch.rad2deg(angle_rad)
    # q1, q2: [N, 4]
    dot = torch.sum(q1 * q2, dim=1).clamp(-1.0, 1.0).abs()
    return 2 * torch.acos(dot).rad2deg()


def _diff_node_features(
    data_a: Data,
    data_b: Data,
    pos_threshold: float = 5.0 / 1000,
    deg_threshold: float = 5.0,
):
    """Return per-node diffs (static shape) and changed indices/count."""
    # Position diff
    pos_raw = data_b.position - data_a.position  # [N, 3]
    pos_dist = torch.linalg.norm(pos_raw, dim=1)
    pos_mask = pos_dist > pos_threshold  # [N]
    pos_diff = torch.zeros_like(pos_raw)
    pos_diff[pos_mask] = pos_raw[pos_mask]

    # Quaternion diff
    quat_delta = _quaternion_delta(data_a.quat, data_b.quat)  # [N,4]
    ang_deg = _quaternion_angle_diff(data_a.quat, data_b.quat)  # [N]
    rot_mask = ang_deg > deg_threshold  # [N]
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


def compound_pos_to_sim_pos(compound_pos: torch.Tensor, env_size_mm=(640, 640, 640)):
    "Pos in compound to pos in sim/sim state"
    env_size_mm = torch.tensor(env_size_mm, device=compound_pos.device)
    compound_pos_xy = (compound_pos[:, :2] - env_size_mm[:2] / 2) / 1000
    compound_pos_z = compound_pos[:, 2] / 1000  # z needs not go lower.
    return torch.cat([compound_pos_xy, compound_pos_z.unsqueeze(1)], dim=1)
