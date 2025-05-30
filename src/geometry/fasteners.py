"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

import math
from typing import Optional, TYPE_CHECKING
from src.geometry.base import Component
from build123d import *
from dataclasses import dataclass
import genesis as gs
from genesis.engine.entities import RigidEntity


class Fastener(Component):
    def __init__(
        self,
        name: str,  # just name them by int ids. except special cases.
        constraint_a_active: bool,
        constraint_b_active: bool,
        initial_body_a: str
        | None = None,  # how will it be constrained in case of hole?
        initial_body_b: str | None = None,
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


def check_fastener_possible_insertion(
    active_fastener_tip_position: tuple[float, float, float],
    hole_positions: dict[str, tuple[float, float, float]],
    connection_threshold: float = 0.75,
):
    # Convert inputs to numpy arrays
    tip_pos = np.array(active_fastener_tip_position)
    hole_names = list(hole_positions.keys())
    hole_pos_array = np.array(list(hole_positions.values()))

    # Calculate squared distances (avoids sqrt for better performance) # equal to math.dist but without sqrt, just a small trick
    squared_distances = np.sum((hole_pos_array - tip_pos) ** 2, axis=1)

    # Find the closest hole within threshold
    within_threshold = squared_distances < (connection_threshold**2)
    if np.any(within_threshold):
        # Get the index of the first hole within threshold
        first_match_idx = np.argmax(within_threshold)
        return hole_names[first_match_idx]
    return None


def activate_connection(fastener_entity: RigidEntity, activate_joint_name: str):
    # scene.sim.rigid_solver.get_joint()
    joint: RigidJoint = fastener_entity.get_joint(activate_joint_name)
    joint.active = True


# To remove A/B constraint and activate screwdriver constraint
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_ab"))] = 0
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_screwdriver"))] = 1
