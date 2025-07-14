"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

from pathlib import Path
from repairs_components.geometry.base import Component
from build123d import *
from dataclasses import dataclass
import genesis as gs
from genesis.engine.entities import RigidEntity
import numpy as np
import torch

from repairs_components.logic.tools.tool import Tool


@dataclass
class Fastener(Component):
    """Fastener class. All values are in millimeters (mm)."""

    def __init__(
        self,
        constraint_b_active: bool,
        initial_body_a: str
        | None = None,  # how will it be constrained in case of hole?
        initial_body_b: str | None = None,
        length: float = 15.0,  # mm
        diameter: float = 5.0,  # mm
        *,
        b_depth: float = 5.0,
        head_diameter: float = 7.5,  # mm
        head_height: float = 3.0,  # mm
        thread_pitch: float = 0.5,  # mm
        screwdriver_name: str = "screwdriver",
    ):
        # assert initial_body_a is not None, "initial_body_a must be provided"
        assert head_diameter > diameter, (
            "head_diameter of a fastener must be greater than diameter"
        )
        assert b_depth > 0, "b_depth of a fastener must be greater than 0"
        self.initial_body_a = initial_body_a
        self.initial_body_b = initial_body_b
        self.thread_pitch = thread_pitch
        self.length = length
        self.b_depth = b_depth
        self.diameter = diameter
        self.head_diameter = head_diameter
        self.head_height = head_height
        self.screwdriver_name = screwdriver_name
        # self.a_constraint_active = True # note: a_constraint_active is always True now.
        self.b_constraint_active = constraint_b_active
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

    def bd_geometry(self) -> tuple[Part, tuple]:
        """Create a build123d geometry for the fastener.

        Returns:
            A build123d Solid representing the screw with shaft and head.
        """
        from build123d import BuildPart, Cylinder, Pos, Align

        with BuildPart() as fastener:
            with Locations(Pos(0, 0, -self.length + self.head_height / 2)):
                # Create the shaft (main cylinder)
                shaft = Cylinder(
                    radius=self.diameter / 2,  # Convert diameter to radius
                    height=self.length,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                )

            # Create the head (wider cylinder)
            with Locations(Pos(0, 0, self.head_height / 2)):
                head = Cylinder(
                    radius=self.head_diameter / 2,  # Convert diameter to radius
                    height=self.head_height,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
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

        # set collision detection position at the tip of the fastener
        fastener_collision_detection_position = tuple(
            shaft.faces().sort_by(Axis.Z).last.center()
        )

        return fastener, fastener_collision_detection_position

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
    active_fastener_tip_position: torch.Tensor,
    part_hole_positions: torch.Tensor,
    part_hole_batch: torch.Tensor,
    connection_dist_threshold: float = 0.75,
    connection_angle_threshold: float = 30,  # degrees
    part_hole_quats: torch.Tensor | None = None,
    active_fastener_quat: torch.Tensor | None = None,
    ignore_part_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    active_fastener_tip_position: [B,3] tensor of tip positions
    part_hole_positions: torch.Tensor of hole positions - per every hole, a tensor of shape [num_holes,3]
    part_hole_quats: torch.Tensor of hole quaternions - per every hole, a tensor of shape [num_holes,4]
    part_hole_batch: torch.Tensor of hole batch indices - per every hole, it's corresponding part index [num_holes]
    connection_threshold: float
    ignore_hole_idx: torch.Tensor, indices of the hole to ignore (if fastener is already inserted in that hole in that batch, ignore.) [B]

    Batch check: for each env, find first hole within threshold or -1. If part_hole_quats is not None, check that the hole is not rotated too much.
    Returns:
    - Tuple of tensors `(part_idx, hole_idx)`, each of shape `[batch]`, where `-1` indicates no match.`
    """
    from repairs_components.processing.translation import are_quats_within_angle

    dist = torch.norm(
        part_hole_positions - active_fastener_tip_position, dim=-1
    )  # [B,H]
    # ignore holes that this fastener is already attached to
    if ignore_part_idx is not None:  # what about in ignore_hole_idx are -1?
        # because ignore_hole_idx is actually a [B, 2] tensor which will only ever have one
        # non-negative value, we can say that mask can be aggregated with `any()`.
        #
        mask = (ignore_part_idx != -1).any(dim=-1)
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


def activate_fastener_to_screwdriver_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    screwdriver_entity: RigidEntity,
    reposition_to_xyz: torch.Tensor,  # [3]
    env_id: int,
    tool_state_to_update: Tool,
    fastener_id: int,
):
    # avoid circular import
    from repairs_components.processing.translation import get_connector_pos
    from repairs_components.logic.tools.screwdriver import Screwdriver

    # ^ note: could be batched, but fastener_entity are not batchable (easily) and it doesn't matter.
    rigid_solver = scene.sim.rigid_solver
    screwdriver_link = screwdriver_entity.base_link.idx
    fastener_head_joint = fastener_entity.base_link.idx
    # Align the fastener before constraining
    screwdriver_quat = screwdriver_entity.get_quat(env_id)
    fastener_entity.set_pos(reposition_to_xyz, env_id)
    fastener_entity.set_quat(screwdriver_quat, env_id)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, screwdriver_link, env_id
    )  # works is genesis's examples.
    assert isinstance(tool_state_to_update, Screwdriver), "Tool must be a Screwdriver"
    tool_state_to_update.picked_up_fastener_name = Fastener.fastener_name_in_simulation(
        fastener_id
    )
    tool_state_to_update.picked_up_fastener_tip_position = get_connector_pos(
        reposition_to_xyz,  # note that we know the position of fastener already
        screwdriver_quat,
        Fastener.get_tip_pos_relative_to_center().unsqueeze(0),
    )
    tool_state_to_update.has_picked_up_fastener = True


def deactivate_fastener_to_screwdriver_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    screwdriver_entity: RigidEntity,
    env_id: int,
    tool_state_to_update: Tool,
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
    tool_state_to_update.has_picked_up_fastener = False


def activate_part_to_fastener_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    hole_pos: torch.Tensor,
    hole_quat: torch.Tensor,
    part_entity: RigidEntity,
    envs_idx: torch.Tensor,
):
    # TODO: make fastener insertion more smooth.
    rigid_solver = scene.sim.rigid_solver
    # fastener_pos = fastener_entity.get_pos(envs_idx)
    # fastener_quat = fastener_entity.get_quat(envs_idx)
    # fastener_head_joint = np.array(fastener_entity.base_link.idx)
    fastener_joint = fastener_entity.base_link.idx
    other_body_link = part_entity.base_link.idx

    fastener_entity.set_pos(hole_pos, envs_idx)
    fastener_entity.set_quat(hole_quat, envs_idx)
    # not exactly the hole position!!!
    # how to prevent insertion too deeply?

    rigid_solver.add_weld_constraint(
        fastener_joint, other_body_link, envs_idx
    )  # works in genesis's examples.


def deactivate_part_connection(
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
