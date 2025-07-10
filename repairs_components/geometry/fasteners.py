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
from typing import Mapping
from typing_extensions import deprecated


@dataclass
class Fastener(Component):
    """Fastener class. All values are in millimeters (mm)."""

    def __init__(
        self,
        constraint_b_active: bool,
        initial_body_a: str
        | None = None,  # how will it be constrained in case of hole?
        initial_body_b: str | None = None,
        thread_pitch: float = 0.5,  # mm
        length: float = 15.0,  # mm
        diameter: float = 5.0,  # mm
        head_diameter: float = 7.5,  # mm
        head_height: float = 3.0,  # mm
        screwdriver_name: str = "screwdriver",
    ):
        # assert initial_body_a is not None, "initial_body_a must be provided"
        assert head_diameter > diameter, (
            "head_diameter of a fastener must be greater than diameter"
        )
        self.initial_body_a = initial_body_a
        self.initial_body_b = initial_body_b
        self.thread_pitch = thread_pitch
        self.length = length
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
        # units are mm
        shaft_radius = self.diameter / 2  # mm
        shaft_length = self.length  # mm
        head_radius = self.head_diameter / 2  # mm
        head_height = self.head_height  # mm
        # Head base at z=0, head centered at head_height/2, shaft centered at -shaft_length/2
        # Tip body at z=-shaft_length
        density = 7.8 / 1000  # g/mm3
        return f"""
    <mujoco>
    <worldbody>
        <body name="{self.name}">
            <!-- <joint name="{self.name}_base_joint" type="weld"/> -->
            <geom name="{self.name}_head" type="cylinder"
                  size="{head_radius} {head_height / 2}"
                  pos="0 0 {head_height / 2}"
                  rgba="0.5 0.5 0.5 1"
                  density="{density}"/>

            <geom name="{self.name}_shaft" type="cylinder"
                  size="{shaft_radius} {shaft_length / 2}"
                  pos="0 0 {-shaft_length / 2}"
                  rgba="0.8 0.8 0.8 1"
                  density="{density}"/>

            <body name="{self.name}_tip" pos="0 0 {-shaft_length}">
                <!-- Tip is located at the end of the shaft (z=-shaft_length) -->
                <site name="{self.name}_tip_site" pos="0 0 0" size="1" rgba="1 0 0 1"/>
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

            RigidJoint("fastener_joint_a")
            RigidJoint("fastener_joint_b")
            RigidJoint("fastener_joint_tip")

        fastener = fastener.part
        fastener.color = Color(0.58, 0.44, 0.86, 0.8)
        fastener.label = self.name

        # set collision detection position at the tip of the fastener
        fastener_collision_detection_position = tuple(
            shaft.faces().sort_by(Axis.Z).last.center()
        )

        return fastener, fastener_collision_detection_position


def check_fastener_possible_insertion(
    active_fastener_tip_position: Mapping[str, np.ndarray] | torch.Tensor | np.ndarray,
    hole_positions: Mapping[str, torch.Tensor | np.ndarray],
    connection_threshold: float = 0.75,
) -> torch.Tensor:
    """
    Batch check: for each env, find first hole index within threshold or -1.
    Returns tensor of shape [batch] with hole index or -1.
    """
    # prepare tip tensor [B,3]
    tip = active_fastener_tip_position
    tip = tip if isinstance(tip, torch.Tensor) else torch.tensor(tip)
    # gather hole names and stack positions [B,H,3]
    hole_keys = list(hole_positions.keys())
    hole_vals = [hole_positions[k] for k in hole_keys]
    hole_vals = [
        v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in hole_vals
    ]
    holes = torch.stack(hole_vals, dim=1)
    # compute squared distances [B,H]
    sq_dist = torch.sum((holes - tip.unsqueeze(1)) ** 2, dim=-1)
    mask = sq_dist < (connection_threshold**2)
    # find first match idx or -1
    first_idx = mask.float().argmax(dim=1)
    any_match = mask.any(dim=1)
    return torch.where(any_match, first_idx, torch.full_like(first_idx, -1))


def activate_hand_connection(
    scene: gs.Scene,
    fastener_entity: RigidEntity,
    franka_arm: RigidEntity,
):
    rigid_solver = scene.sim.rigid_solver
    hand_link = franka_arm.get_link("hand")
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, hand_link
    )  # works is genesis's examples.


def deactivate_hand_connection(
    scene: gs.Scene,
    fastener: Fastener,
    fastener_entity: RigidEntity,
    franka_arm: RigidEntity,
):
    rigid_solver = scene.sim.rigid_solver
    hand_link = franka_arm.get_link("hand")
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, hand_link
    )  # works is genesis's examples.


def activate_part_connection(
    scene: gs.Scene,
    fastener: Fastener,
    fastener_entity: RigidEntity,
    other_entity: RigidEntity,
    hole_link_name: str,
):
    rigid_solver = scene.sim.rigid_solver
    # fastener_head_joint = np.array(fastener_entity.base_link.idx)
    fastener_head_joint = np.array(fastener_entity.base_joint.idx)
    other_body_hole_link = np.array(other_entity.get_link(hole_link_name).idx)
    rigid_solver.add_weld_constraint(
        fastener_head_joint, other_body_hole_link
    )  # works is genesis's examples.


def deactivate_part_connection(
    scene: gs.Scene,
    fastener: Fastener,
    fastener_entity: RigidEntity,
    other_entity: RigidEntity,
    hole_link_name: str,
):
    rigid_solver = scene.sim.rigid_solver
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
    other_body_hole_link = np.array(other_entity.get_link(hole_link_name).idx)
    rigid_solver.delete_weld_constraint(
        fastener_head_joint, other_body_hole_link
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
