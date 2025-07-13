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


def check_fastener_possible_insertion(
    active_fastener_tip_position: torch.Tensor,
    part_hole_positions: dict[str, torch.Tensor],
    connection_threshold: float = 0.75,
    ignore_hole_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    active_fastener_tip_position: [B,3] tensor of tip positions
    part_hole_positions: dict[str, torch.Tensor] of hole positions - per every part, a tensor of shape [num_holes,3]
    connection_threshold: float
    ignore_hole_idx: torch.Tensor, indices of the hole to ignore (if fastener is already inserted in that hole in that batch, ignore.) [B]

    Batch check: for each env, find first hole within threshold or -1.
    Returns:
    - Tuple of tensors `(part_idx, hole_idx)`, each of shape `[batch]`, where `-1` indicates no match.`
    """
    # flatten all holes across parts
    part_names = list(part_hole_positions.keys())
    B = active_fastener_tip_position.shape[0]
    holes_list = []
    hps = []
    for val in part_hole_positions.values():
        pos = val
        # add batch dim if needed
        if pos.dim() == active_fastener_tip_position.dim() - 1:
            pos = pos.unsqueeze(0).expand(B, *pos.shape)
        holes_list.append(pos)
        hps.append(pos.shape[1])
    # concat to [B, total_holes, 3]
    holes_all = torch.cat(holes_list, dim=1)
    total_h = holes_all.shape[1]
    # build part_id map and prefix offsets
    device = active_fastener_tip_position.device
    part_id_map = torch.cat(
        [
            torch.full((hp,), idx, dtype=torch.long, device=device)
            for idx, hp in enumerate(hps)
        ],
        dim=0,
    )
    prefix = torch.tensor(
        [0] + list(np.cumsum(hps)[:-1]), dtype=torch.long, device=device
    )
    # compute squared distances and mask
    sq_dist = torch.sum(
        (holes_all - active_fastener_tip_position.unsqueeze(1)) ** 2, dim=-1
    )
    mask = sq_dist < (connection_threshold**2)
    # optionally ignore a global hole index
    if (
        ignore_hole_idx is not None
        and ignore_hole_idx >= 0
        and ignore_hole_idx < total_h
    ):
        mask[:, ignore_hole_idx] = False
    any_match = mask.any(dim=1)
    first_global = mask.float().argmax(dim=1)
    # derive part and local hole indices
    part_idx = part_id_map[first_global]
    hole_idx = first_global - prefix[part_idx]
    # set -1 where no match
    part_idx = torch.where(any_match, part_idx, torch.full_like(part_idx, -1))
    hole_idx = torch.where(any_match, hole_idx, torch.full_like(hole_idx, -1))
    return part_idx, hole_idx


def activate_fastener_to_hand_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    franka_arm: RigidEntity,
    reposition_to_xyz: torch.Tensor,
    rotate_to_quat: torch.Tensor,
    envs_idx: torch.Tensor,
    tool_state_to_update: list[Tool],
):
    # avoid circular import
    from repairs_components.processing.translation import get_connector_pos

    assert (
        reposition_to_xyz.shape[0]
        == rotate_to_quat.shape[0]
        == envs_idx.shape[0]
        == len(tool_state_to_update)
    ), (
        "Reposition_to, rotate_to_quat, envs_idx, tool_state_to_update must have the same shape"
    )
    # TODO: align the fastener to the hand before constraining
    rigid_solver = scene.sim.rigid_solver
    hand_link = franka_arm.get_link("hand")
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    fastener_entity.set_pos(reposition_to_xyz, envs_idx)
    fastener_entity.set_quat(rotate_to_quat, envs_idx)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, hand_link, envs_idx
    )  # works is genesis's examples.
    tool_state_to_update[envs_idx].picked_up_fastener_name = fastener_entity.name
    tool_state_to_update[envs_idx].picked_up_fastener_tip_position = get_connector_pos(
        fastener_entity.get_pos(envs_idx),
        fastener_entity.get_quat(envs_idx),
        Fastener.get_tip_pos_relative_to_center(),
    )
    tool_state_to_update[envs_idx].has_picked_up_fastener = True


def deactivate_fastener_to_hand_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    franka_arm: RigidEntity,
    envs_idx: torch.Tensor,
    tool_state_to_update: list[Tool],
):
    assert envs_idx.shape[0] == len(tool_state_to_update), (
        "envs_idx and tool_state_to_update must have the same shape"
    )
    rigid_solver = scene.sim.rigid_solver
    hand_link = franka_arm.get_link("hand")
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, hand_link, envs_idx
    )  # works is genesis's examples.
    tool_state_to_update[envs_idx].picked_up_fastener_name = None
    tool_state_to_update[envs_idx].picked_up_fastener_tip_position = None
    tool_state_to_update[envs_idx].has_picked_up_fastener = False


def activate_part_to_fastener_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    part_entity: RigidEntity,
    hole_link_name: str,
    envs_idx: torch.Tensor,
):
    # TODO: align the fastener to the hole before constraining
    rigid_solver = scene.sim.rigid_solver
    # fastener_head_joint = np.array(fastener_entity.base_link.idx)
    fastener_head_joint = np.array(fastener_entity.base_joint.idx)
    other_body_hole_link = np.array(part_entity.get_link(hole_link_name).idx)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, other_body_hole_link, envs_idx
    )  # works is genesis's examples.


def deactivate_part_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    part_entity: RigidEntity,
    hole_link_name: str,
    envs_idx: torch.Tensor,
):
    rigid_solver = scene.sim.rigid_solver
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    other_body_hole_link = np.array(part_entity.get_link(hole_link_name).idx)
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, other_body_hole_link, envs_idx
    )  # works is genesis's examples.


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
