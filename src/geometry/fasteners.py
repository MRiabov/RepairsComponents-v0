"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

from typing import Optional, TYPE_CHECKING
from geometry.base import Component
from build123d import *
from dataclasses import dataclass

if TYPE_CHECKING:
    from genesis.sim import Model, Data  # noqa: F401


class Fastener(Component):
    def __init__(
        self,
        initial_body_a: str,  # how will it be constrained in case of hole?
        initial_body_b: str,
        name: str,  # just name them by int ids. except special cases.
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

    def screw_mjcf(self):
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
            </body>
            
            <!-- Weld constraints to A and B (both active at spawn) -->
            <equality name="{self.name}_to_A" active="true">
                <weld body1="{self.name}" body2="{self.initial_body_a}" relpose="true"/>
            </equality>

            <equality name="{self.name}_to_B" active="true">
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
        screw = screw.part
        screw.color = Color(0.58, 0.44, 0.86, 0.8)
        screw.label = self.name + "@fastener"

        return screw


# To remove A/B constraint and activate screwdriver constraint
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_ab"))] = 0
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_screwdriver"))] = 1
