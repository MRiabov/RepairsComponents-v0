"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

import math
from typing import Optional, TYPE_CHECKING
from repairs_components.geometry.base import Component
from build123d import *
from dataclasses import dataclass
import genesis as gs
from genesis.engine.entities import RigidEntity


class Fastener(Component):
    def __init__(
        self,
        constraint_a_active: bool,
        constraint_b_active: bool,
        initial_body_a: str
        | None = None,  # how will it be constrained in case of hole?
        initial_body_b: str | None = None,
        name: str = "",  # just name them by int ids. except special cases.
        thread_pitch: float = 0.5,
        length: float = 10.0,
        diameter: float = 3.0,
        head_diameter: float = 5.5,
        head_height: float = 2.0,
        screwdriver_name: str = "screwdriver",
    ):
        self.initial_body_a = initial_body_a
        self.initial_body_b = initial_body_b
        self.thread_pitch = thread_pitch
        self.length = length
        self.diameter = diameter
        self.head_diameter = head_diameter
        self.head_height = head_height
        self.name = name
        self.screwdriver_name = screwdriver_name
        self.a_constraint_active = constraint_a_active
        self.b_constraint_active = constraint_b_active

    def get_mjcf(self):
        """Get MJCF of a screw.

        Args:
            thread_pitch: Distance between threads in mm.
            length: Total length of the screw in mm.
            diameter: Outer diameter of the screw thread in mm.
            head_diameter: Diameter of the screw head in mm.
            head_height: Height of the screw head in mm.
            name: Optional name for the screw.
        """

        return f"""
            <body name="{self.name}">
                <freejoint name="{self.name}_joint"/>
                <geom name="{self.name}_shaft" type="cylinder" size="{self.diameter / 2000} {self.length / 2000}" 
                    pos="0 0 {self.length / 2000}" rgba="0.8 0.8 0.8 1" mass="0.1"/>
                <geom name="{self.name}_head" type="cylinder" size="{self.head_diameter / 2000} {self.head_height / 2000}" 
                    pos="0 0 {(self.length + self.head_height / 2) / 2000}" rgba="0.5 0.5 0.5 1" mass="0.05"/>
                <body name="{self.name}_tip" pos="0 0 0">
                    <!-- Tip is located at 0,0,0 -->
                </body>
            </body>
            
            <!-- NOTE: possibly unnecessary! there is rigid_solver.add_weld_constraint.-->

            <!-- Weld constraints to A and B (both active at spawn)-->
            <equality name="{self.name}_to_A" active="{self.a_constraint_active}">
                <weld body1="{self.name}" body2="{self.initial_body_a}" relpose="true"/>
            </equality>

            <equality name="{self.name}_to_B" active="{self.b_constraint_active}">
                <weld body1="{self.name}" body2="{self.initial_body_b}" relpose="true"/>
            </equality>

            <!-- Equality constraint to attach to screwdriver -->
            <equality name="{self.name}_to_screwdriver" active="false">
                <weld body1="{self.name}" body2="{self.screwdriver_name}" relpose="true"/>
            </equality>
            """

    def bd_geometry(self):
        """Create a build123d geometry for the fastener.

        Returns:
            A build123d Solid representing the screw with shaft and head.
        """
        from build123d import BuildPart, Cylinder, Pos, Align

        with BuildPart() as screw:
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

            CylindricalJoint("fastener_joint_a", to_part=None, axis=Axis.Z)
            CylindricalJoint("fastener_joint_b", to_part=None, axis=Axis.Z)
            CylindricalJoint("fastener_joint_tip", to_part=None, axis=Axis.Z)

        screw = screw.part
        screw.color = Color(0.58, 0.44, 0.86, 0.8)
        screw.label = self.name + "@fastener"

        # set collision detection position at the tip of the fastener
        fastener_collision_detection_position = (
            shaft.faces().sort_by(Axis.Z).last.center().to_tuple()
        )

        return screw, fastener_collision_detection_position


import numpy as np
import torch
from typing import Mapping


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
    fastener: Fastener,
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
    fastener_head_joint = np.array(fastener_entity.base_link.idx)
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
