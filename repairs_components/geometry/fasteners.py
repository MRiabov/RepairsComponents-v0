"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

from pathlib import Path

from ocp_vscode import show
from repairs_components.geometry.base import Component
from build123d import *
from dataclasses import dataclass
import genesis as gs
from genesis.engine.entities import RigidEntity
import numpy as np
import torch

from repairs_components.logic.tools.tool import Tool
from repairs_components.processing.geom_utils import get_connector_pos


@dataclass
class Fastener(Component):
    """Fastener class. All values are in millimeters (mm)."""

    def __init__(
        self,
        initial_hole_id_a: int | None = None,  # hole ID for constraint A
        initial_hole_id_b: int | None = None,  # hole ID for constraint B
        # note: initial_hole_a and b are not used in bd geometry.
        length: float = 15.0,  # mm
        diameter: float = 5.0,  # mm
        *,
        expected_body_name_a: str | None = None,
        expected_body_name_b: str | None = None,
        b_depth: float = 5.0,
        head_diameter: float = 7.5,  # mm
        head_height: float = 3.0,  # mm
        thread_pitch: float = 0.5,  # mm
    ):
        """
        Args:
        - initial_hole_id_a: ID of the hole for constraint A
        - initial_hole_id_b: ID of the hole for constraint B
        - expected_body_name_a: Optional(!) Name of the body for constraint A. If provided, constraint mechanism can check that id of holes corresponds to ids of bodies, and those to names.
        - expected_body_name_b: Optional(!) Name of the body for constraint B. If provided, constraint mechanism can check that id of holes corresponds to ids of bodies, and those to names.
        """  # maybe TODO: expected_body_name_a and b.
        # assert initial_hole_id_a is not None, "initial_hole_id_a must be provided"
        assert head_diameter > diameter, (
            "head_diameter of a fastener must be greater than diameter"
        )
        assert b_depth > 0, "b_depth of a fastener must be greater than 0"
        self.initial_hole_id_a = initial_hole_id_a
        self.initial_hole_id_b = initial_hole_id_b
        self.thread_pitch = thread_pitch
        self.length = length
        self.b_depth = b_depth
        self.diameter = diameter
        self.head_diameter = head_diameter
        self.head_height = head_height
        # self.a_constraint_active = True # note: a_constraint_active is always True now.
        self.b_constraint_active = (
            initial_hole_id_b is not None
        )  # deprecated, for backwards compatibility
        self.name = get_fastener_singleton_name(self.diameter, self.length)

    def get_mjcf(self):
        """Get MJCF of a screw. MJCF is preferred because it is faster and more reliable than meshes.

        Args:
            thread_pitch: Distance between threads in mm.
            length: Total length of the screw in mm.
            diameter: Outer diameter of the screw thread in mm.
            head_diameter: Diameter of the screw head in mm.
            head_height: Height of the screw head in mm.
        """
        # MJCF expects meters, Build123d uses mm, so convert mm to m
        shaft_radius = self.diameter / 2 / 1000
        shaft_length = self.length / 1000
        head_radius = self.head_diameter / 2 / 1000
        head_height = self.head_height / 1000
        # Head base at z=0, head centered at head_height/2, shaft centered at -shaft_length/2
        # Tip body at z=-shaft_length
        return f"""
    <mujoco>
    <worldbody>
        <body name="{self.name}">
            <!-- <joint name="{self.name}_base_joint" type="weld"/> -->
            <geom name="{self.name}_head" type="cylinder"
                  size="{head_radius} {head_height / 2}"
                  pos="0 0 {head_height / 2}"
                  rgba="0.5 0.5 0.5 1"
                  density="7800"/>

            <geom name="{self.name}_shaft" type="cylinder"
                  size="{shaft_radius} {shaft_length / 2}"
                  pos="0 0 {-shaft_length / 2}"
                  rgba="0.8 0.8 0.8 1"
                  density="7800"/>

            <body name="{self.name}_tip" pos="0 0 {-shaft_length}">
                <!-- Tip is located at the end of the shaft (z=-shaft_length) -->
                <site name="{self.name}_tip_site" pos="0 0 0" size="0.001" rgba="1 0 0 1"/>
            </body>
        </body>
    </worldbody>
    </mujoco>
"""

    def bd_geometry(self) -> Part:
        """Create a build123d geometry for the fastener.

        Returns:
            A build123d Solid representing the screw with shaft and head.
        """
        from build123d import BuildPart, Cylinder, Align

        with BuildPart() as fastener:
            with Locations((0, 0, -self.length / 2)):
                # Create the shaft (main cylinder)
                shaft = Cylinder(
                    radius=self.diameter / 2,  # Convert diameter to radius
                    height=self.length,
                )

            # Create the head (wider cylinder)
            with Locations((0, 0, self.head_height / 2)):
                head = Cylinder(
                    radius=self.head_diameter / 2,  # Convert diameter to radius
                    height=self.head_height,
                )
            head.faces().filter_by(Axis.Z).sort_by(Axis.Z).last

            RigidJoint(
                "fastener_joint_a",
                joint_location=shaft.faces()
                .filter_by(Axis.Z)
                .sort_by(Axis.Z)
                .last.center_location,
            )  # lowest point of head/top of shaft (top of shaft because there were rotation issues)
            RigidJoint(
                "fastener_joint_b",
                joint_location=shaft.faces()
                .filter_by(Axis.Z)
                .sort_by(Axis.Z)
                .last.offset(amount=-self.b_depth)
                .center_location,
            )  # lowest point of the head + offset
            RigidJoint(
                "fastener_joint_tip",
                joint_location=fastener.faces()
                .filter_by(Axis.Z)
                .sort_by(Axis.Z)
                .first.center_location,
            )  # lowest point of the fastener

        fastener = fastener.part
        fastener.color = Color(0.58, 0.44, 0.86, 0.8)
        fastener.label = self.name

        # # set collision detection position at the tip of the fastener
        # fastener_collision_detection_position = tuple(
        #     shaft.faces().sort_by(Axis.Z).last.center()
        # )

        return fastener

    @staticmethod
    def get_tip_pos_relative_to_center(length: float = 15.0 / 1000):
        return torch.tensor([0, 0, -length])
        # note: fastener head relative to center is a pointless function because 1mm offset or whatnot is insignificant.

    @staticmethod
    def fastener_name_in_simulation(fastener_id_in_genesis: int):
        """Return the name of a fastener that is used in the genesis simulation. Note that
        this is not the name of the fastener in build123d, but the name of the fastener
        in the genesis simulation."""
        return f"{fastener_id_in_genesis}@fastener"
        # the inverse is int(fastener_name.split("@")[0]), it's spinkled throughout the code.


def check_fastener_possible_insertion(
    active_fastener_tip_position: torch.Tensor,  # [B,3]
    part_hole_positions: torch.Tensor,  # [B,num_holes,3]
    part_hole_batch: torch.Tensor,  # [num_holes] #NOTE: part_hole_batch is static amongst batch!
    connection_dist_threshold: float = 0.75,  # meters (!)
    connection_angle_threshold: float = 30,  # degrees
    part_hole_quats: torch.Tensor | None = None,  # [B,num_holes,4]
    active_fastener_quat: torch.Tensor | None = None,  # [B,4]
    ignore_part_idx: torch.Tensor | None = None,  # [B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    active_fastener_tip_position: [B,3] tensor of tip positions
    part_hole_positions: torch.Tensor of hole positions - per every hole, a tensor of shape [B, num_holes,3]
    part_hole_quats: torch.Tensor of hole quaternions - per every hole, a tensor of shape [B, num_holes,4]
    part_hole_batch: torch.Tensor of hole batch indices - per every hole, it's corresponding part index. Static amongst batch. [num_holes]
    connection_threshold: float
    ignore_part_idx: torch.Tensor, indices of the part to ignore (if fastener is already inserted in that part in that batch, ignore.) [B]

    Batch check: for each env, find first hole within threshold or -1. If part_hole_quats is not None, check that the hole is not rotated too much.
    Returns:
    - Tuple of tensors `(part_idx, hole_idx)`, each of shape `[batch]`, where `-1` indicates no match and value >= 0 indicates the hole index, with its part index.
    """
    from repairs_components.processing.geom_utils import are_quats_within_angle

    assert part_hole_positions.shape[:2] == part_hole_batch.shape, (
        f"part_hole_positions, part_hole_batch must have the same batch and hole counts.\n"
        f"part_hole_positions: {part_hole_positions.shape}, part_hole_batch: {part_hole_batch.shape}"
    )
    assert part_hole_batch.ndim == 2, (
        f"part_hole_batch must be a 2D tensor of shape [B, H], got {part_hole_batch.shape}"
    )
    part_hole_batch = part_hole_batch[
        0
    ]  # since they are equal over the batch, simplify by [0]
    dist = torch.norm(
        part_hole_positions - active_fastener_tip_position, dim=-1
    )  # [B, H]
    # ignore holes that this fastener is already attached to
    if ignore_part_idx is not None:
        assert ignore_part_idx.shape == (part_hole_positions.shape[0],), (
            f"ignore_part_idx must be a 1D tensor of shape [B], got {ignore_part_idx.shape}"
        )
        mask = ignore_part_idx != -1
        batch_idx = torch.arange(len(ignore_part_idx), device=ignore_part_idx.device)[
            mask
        ]
        part_idx = ignore_part_idx[mask]
        dist[batch_idx, part_idx] = float("inf")

    # mask out holes that are not within angle threshold
    if active_fastener_quat is not None:
        assert part_hole_quats is not None, (
            "part_hole_quats must be provided if active_fastener_quat is provided"
        )
        assert part_hole_quats.shape[:2] == part_hole_positions.shape[:2], (
            "part_hole_quats, part_hole_positions must have the same batch and hole counts"
        )
        angle_mask = are_quats_within_angle(
            part_hole_quats,
            active_fastener_quat.unsqueeze(1),
            connection_angle_threshold,
        )
        dist[~angle_mask] = float("inf")
    # find the closest hole
    hole_min = torch.min(dist, dim=-1)
    # if it's close enough, return the hole index
    close_enough = hole_min.values < connection_dist_threshold  # [B]
    # if it's not close enough, return -1
    hole_idx = torch.where(close_enough, hole_min.indices, -1)
    part_idx = torch.where(
        hole_idx != -1, part_hole_batch[hole_idx], torch.full_like(hole_idx, -1)
    )

    return part_idx, hole_idx


def attach_fastener_to_screwdriver(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    screwdriver_entity: RigidEntity,
    tool_state_to_update: Tool,
    fastener_id: int,
    env_id: int,
):
    # avoid circular import
    from repairs_components.logic.tools.screwdriver import Screwdriver

    # ^ note: could be batched, but fastener_entity are not batchable (easily) and it doesn't matter.
    rigid_solver = scene.sim.rigid_solver
    screwdriver_link = screwdriver_entity.base_link.idx
    fastener_head_joint = fastener_entity.base_link.idx
    # compute screwdriver grip position
    screwdriver_xyz = screwdriver_entity.get_pos(env_id)
    screwdriver_quat = screwdriver_entity.get_quat(env_id)
    screwdriver_grip_xyz = (
        screwdriver_xyz
        + Screwdriver.fastener_connector_pos_relative_to_center().unsqueeze(0)
    )  # ^ `-` because we want the fastener to be at the screwdriver grip position
    # ^ hmm. The tests pass this way, but is this expected?

    # Align the fastener before constraining
    fastener_entity.set_pos(screwdriver_grip_xyz, env_id)
    fastener_entity.set_quat(screwdriver_quat, env_id)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, screwdriver_link, env_id
    )  # works is genesis's examples.
    assert isinstance(tool_state_to_update, Screwdriver), "Tool must be a Screwdriver"
    tool_state_to_update.picked_up_fastener_tip_position = screwdriver_grip_xyz
    tool_state_to_update.picked_up_fastener_quat = screwdriver_quat
    tool_state_to_update.picked_up_fastener_name = Fastener.fastener_name_in_simulation(
        fastener_id
    )


def detach_fastener_from_screwdriver(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    screwdriver_entity: RigidEntity,
    tool_state_to_update: Tool,
    env_id: int,
):
    from repairs_components.logic.tools.screwdriver import Screwdriver

    rigid_solver = scene.sim.rigid_solver
    hand_link = screwdriver_entity.base_link.idx
    fastener_head_joint = fastener_entity.base_link.idx
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, hand_link, [env_id]
    )  # works in genesis's examples. #[env_id] because it wants a collection.
    assert isinstance(tool_state_to_update, Screwdriver), "Tool must be a Screwdriver"
    tool_state_to_update.picked_up_fastener_name = None
    tool_state_to_update.picked_up_fastener_tip_position = None
    tool_state_to_update.picked_up_fastener_quat = None


def attach_fastener_to_part(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    inserted_into_hole_pos: torch.Tensor,
    inserted_into_hole_quat: torch.Tensor,
    inserted_to_hole_depth: torch.Tensor,
    inserted_into_part_entity: RigidEntity,
    inserted_into_hole_is_through: torch.Tensor,
    top_hole_is_through: torch.Tensor,  # only for assertion.
    already_inserted_into_one_hole: torch.Tensor,
    top_hole_depth: torch.Tensor,
    fastener_length: torch.Tensor,
    envs_idx: torch.Tensor,
):
    """Attach a fastener to a part at a target hole and add a weld constraint.

    This is invoked from `step_repairs -> step_screw_in_or_out` when the policy decides to
    screw in. It computes the fastener pose from the target hole world pose and depth, taking
    into account whether the hole is through/blind and any existing partial insertion into a
    previously connected ("top") hole in the same environment.

    Behavior:
    - Position is computed via `recalculate_fastener_pos_with_offset_to_hole(...)`, which:
      * for a through hole with no prior insertion: aligns the fastener head with the hole top
      * for a blind hole with no prior insertion: offsets by (fastener_length - hole_depth)
        along the hole axis
      * for a second connection when partially inserted into a top hole: offsets by
        (fastener_length - top_hole_depth)
    - Orientation is set equal to `inserted_into_hole_quat`.
    - A rigid weld constraint is added between the fastener base link and the part base link
      (temporary simplification; will be replaced by a revolute/cylindrical joint later).

    Args:
        scene (gs.Scene): Genesis scene.
        fastener_entity (RigidEntity): The fastener to attach.
        inserted_into_hole_pos (torch.Tensor): World-frame position of the target hole.
            Shape [B, 3] where B == len(envs_idx). In `step_repairs` this is passed for a
            single environment (B=1).
        inserted_into_hole_quat (torch.Tensor): World-frame quaternion [w, x, y, z] of the
            target hole. Shape [B, 4].
        inserted_to_hole_depth (torch.Tensor): Target hole depth in meters (> 0). Shape [B].
        inserted_into_part_entity (RigidEntity): Part owning the target hole.
        inserted_into_hole_is_through (torch.Tensor): Whether the target hole is through.
            Shape [B] (bool).
        top_hole_is_through (torch.Tensor): Whether the already-connected "top" hole is through.
            Used only for validation. Where `already_inserted_into_one_hole` is True, all entries
            must be True. Shape [B] (bool).
        already_inserted_into_one_hole (torch.Tensor): Whether the fastener is already inserted
            into a (top) hole in the same environment. Shape [B] (bool).
        top_hole_depth (torch.Tensor): Depth inserted into the top hole in meters (>= 0). If
            `already_inserted_into_one_hole` is True, must be > 0; otherwise must be 0. Shape [B].
        fastener_length (torch.Tensor): Total fastener length in meters (> 0). Shape [B].
        envs_idx (torch.Tensor): Indices of environments to apply the update to. Shape [B].

    Notes:
        - All tensor arguments must be filtered to the same batch as `envs_idx` (B items).
        - Units are meters throughout.
        - Preconditions enforced by assertions:
            * Where `already_inserted_into_one_hole` is True: `top_hole_is_through` must be True
              and `top_hole_depth` > 0
            * Where `already_inserted_into_one_hole` is False: `top_hole_depth` == 0
            * `inserted_to_hole_depth` > 0 and `fastener_length` > 0

    Returns:
        None. Side effects: updates the fastener pose and adds a weld constraint to
        `inserted_into_part_entity` in the specified environments.
    """
    assert (top_hole_is_through[already_inserted_into_one_hole]).all(), (
        f"Where already inserted, must be inserted into a through hole (can't insert when the top hole is blind).\n"
        f"already_inserted_into_one_hole: {already_inserted_into_one_hole}, top_hole_is_through: {top_hole_is_through}"
    )
    assert (already_inserted_into_one_hole == (top_hole_depth > 0)).all(), (
        f"Where marked as uninserted, top hole depth must be 0, and where inserted, >0.\n"
        f"already_inserted_into_one_hole: {already_inserted_into_one_hole}, top_hole_depth: {top_hole_depth}"
    )
    assert fastener_length > 0, (
        f"Fastener length must be positive. Fastener_length: {fastener_length}"
    )
    assert inserted_to_hole_depth > 0, (
        f"Inserted to hole depth must be positive. Inserted_to_hole_depth: {inserted_to_hole_depth}"
    )

    # TODO: make fastener insertion more smooth.
    rigid_solver = scene.sim.rigid_solver
    # fastener_pos = fastener_entity.get_pos(envs_idx)
    # fastener_quat = fastener_entity.get_quat(envs_idx)
    # fastener_head_joint = np.array(fastener_entity.base_link.idx)
    fastener_joint = fastener_entity.base_link.idx
    other_body_link = inserted_into_part_entity.base_link.idx

    # TODO: This needs to be updated to pass proper hole_is_through and top_hole_depth values
    # For now, assuming blind holes with no partial insertion
    # top_hole_depth = torch.zeros_like(hole_depth)  # No partial insertion

    fastener_pos = recalculate_fastener_pos_with_offset_to_hole(
        inserted_into_hole_pos,
        inserted_into_hole_quat,
        inserted_to_hole_depth,
        inserted_into_hole_is_through,
        fastener_length,
        top_hole_depth,
    )

    fastener_entity.set_pos(fastener_pos, envs_idx)
    fastener_entity.set_quat(inserted_into_hole_quat, envs_idx)

    rigid_solver.add_weld_constraint(
        fastener_joint, other_body_link, envs_idx
    )  # works in genesis's examples.
    # FIXME: rigid weld constraint would make screwing the other part impossible. it should be a circular joint first.


def detach_fastener_from_part(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    part_entity: RigidEntity,
    envs_idx: torch.Tensor,
):
    rigid_solver = scene.sim.rigid_solver
    fastener_head_joint = fastener_entity.base_link.idx
    other_body_hole_link = part_entity.base_link.idx
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, other_body_hole_link, envs_idx
    )  # works in genesis's examples.


class FastenerHolder(Component):
    fastener_sizes_held: torch.Tensor  # float 1D tensor of fastener sizes held

    # so, should each body hold fasteners? heterogenous graph?
    # no, simply create a "count_loose_fasteners_inside" as an integer and node feature and fasteners
    # will be constrained.

    # and the fastener holder is a specialty component for holding loose fasteners.

    # so if fasteners are loose, how do we reconstruct them? Probably SAVE the fasteners metadata to bodies graph.
    # however it is not used in `x`.

    def __init__(self, name: str, fastener_sizes_held: torch.Tensor):
        super().__init__(name)
        self.fastener_sizes_held = fastener_sizes_held
        self.count_fasteners_held = torch.nonzero(fastener_sizes_held).shape[0]

    def bd_geometry():
        # TODO: copy bd_geometry from tooling stand... tooling stand kind of is the fastener holder.
        raise NotImplementedError


# NOTE: see https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/rigid/suction_cup.py for rigid constraint added in time

# To remove A/B constraint and activate screwdriver constraint
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_ab"))] = 0
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_screwdriver"))] = 1


def get_fastener_singleton_name(diameter: float, length: float) -> str:
    """Return the name for a fastener singleton based on its diameter and length."""
    diameter_str = f"{diameter:.2f}"
    length_str = f"{length:.2f}"
    return f"fastener_d{diameter_str}_l{length_str}@fastener"


def get_fastener_params_from_name(name: str) -> tuple[float, float]:
    """Return the diameter and lengthF of a fastener singleton based on its name."""
    diameter_str = name.split("_")[1][1:]  # [1:] - remove 'd'
    length_str = name.split("_")[2][1:]  # [1:] - remove 'h'
    length_str = length_str.split("@")[0]  # remove everything after '@'
    return float(diameter_str), float(length_str)


def get_singleton_fastener_save_path(
    diameter: float, length: float, base_dir: Path
) -> Path:
    """Return the save path for a fastener singleton based on its diameter and length."""
    return (
        base_dir
        / "shared"
        / "fasteners"
        / f"fastener_d{diameter:.2f}_h{length:.2f}.xml"
    )


def get_fastener_save_path_from_name(name: str, base_dir: Path) -> Path:
    """Return the save path for a fastener singleton based on its name."""
    return base_dir / "shared" / "fasteners" / (name + ".xml")


def recalculate_fastener_pos_with_offset_to_hole(
    hole_pos: torch.Tensor,  # [B, 3]
    hole_quat: torch.Tensor,  # [B, 4]
    hole_depth: torch.Tensor,  # [B] - always positive
    hole_is_through: torch.Tensor,  # [B] - boolean mask
    fastener_length: torch.Tensor,  # [B]
    top_hole_depth: torch.Tensor,  # [B] # zero or positive
):
    """Handle three cases:
    1. Attaching a free fastener to a through hole, so connect the top of the fastener
    at the top of the hole
    2. Attaching a free fastener to a non-through (blind) hole, so connect the bottom
    of the fastener at the bottom of the hole
    3. Attaching a fastener partially inserted into a through hole, so top_hole_depth
    is not None, so connect the bottom (B) hole to the center of the fastener.
    - in the third case, the remaining length to be accommodated by the bottom hole is
      (fastener_length - top_hole_depth). Therefore, offset along the hole axis by
      (fastener_length - top_hole_depth), i.e. (hole_pos + (fastener_length - top_hole_depth))@quat.
    - in the third case, the hole_pos, hole_quat and hole_depth are params of a bottom hole.
    - in the third case, the fastener inserted into a blind hole can not be connected nowhere else.
    """
    assert hole_depth >= 0, "Hole depth must be positive."
    assert top_hole_depth >= 0, "top_hole_depth must be positive or zero."
    assert fastener_length > top_hole_depth, (
        "fastener_length must be greater than top_hole_depth."
    )

    through_hole = hole_is_through
    has_partial_insertion = top_hole_depth > 0

    # Start with hole_pos for all cases
    fastener_pos = hole_pos.clone()

    # Case 1: Through hole without partial insertion - fastener_pos = hole_pos (no offset)
    # This is already handled by starting with hole_pos.clone()

    # Case 2: Blind hole without partial insertion - offset by (fastener_length - hole_depth)
    blind_hole_no_partial = ~through_hole & ~has_partial_insertion
    if blind_hole_no_partial.any():
        # For blind holes, we need to offset the fastener position
        # The offset should be applied in the hole's coordinate system (using quaternion)
        offset = torch.zeros_like(hole_pos)
        offset[blind_hole_no_partial, 2] = (fastener_length - hole_depth)[
            blind_hole_no_partial
        ]

        # Apply quaternion transformation using get_connector_pos
        transformed_pos = get_connector_pos(
            hole_pos[blind_hole_no_partial],
            hole_quat[blind_hole_no_partial],
            offset[blind_hole_no_partial],
        )
        fastener_pos[blind_hole_no_partial] = transformed_pos

    # Case 3: Partial insertion - offset by the remaining length (L - top_hole_depth)
    # (applies to both through and blind holes for the second connection)
    if has_partial_insertion.any():
        # For partial insertion, offset by the depth the fastener is already inserted
        offset = torch.zeros_like(hole_pos)
        offset[has_partial_insertion, 2] = (fastener_length - top_hole_depth)[
            has_partial_insertion
        ]

        # Apply quaternion transformation using get_connector_pos
        transformed_pos = get_connector_pos(
            hole_pos[has_partial_insertion],
            hole_quat[has_partial_insertion],
            offset[has_partial_insertion],
        )
        fastener_pos[has_partial_insertion] = transformed_pos

    return fastener_pos


if __name__ == "__main__":
    show(Fastener().bd_geometry())
