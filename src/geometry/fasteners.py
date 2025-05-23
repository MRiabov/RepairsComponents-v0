"""Fastener components like screws, bolts, and nuts.

This module provides fastener components that can be used with the Genesis physics simulator.
Genesis supports MJCF (MuJoCo) format for model definition, allowing for easy migration
from MuJoCo-based simulations.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import genesis
from .base import Component

if TYPE_CHECKING:
    from genesis.sim import Model, Data


def screw_mjcf(
    initial_body_a: str,
    initial_body_b: str,
    thread_pitch: float = 0.5,
    length: float = 10.0,
    diameter: float = 3.0,
    head_diameter: float = 5.5,
    head_height: float = 2.0,
    name: Optional[str] = None,
    screwdriver_name: str = "screwdriver",
):
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
        <body name="{name}">
            <freejoint name="{name}_joint"/>
            <geom name="{name}_shaft" type="cylinder" size="{diameter / 2000} {length / 2000}" 
                  rgba="0.8 0.8 0.8 1" mass="0.1"/>
            <geom name="{name}_head" type="cylinder" size="{head_diameter / 2000} {head_height / 2000}" 
                  pos="0 0 {-(length + head_height) / 2000}" rgba="0.5 0.5 0.5 1" mass="0.05"/>
        </body>
        
        <!-- Weld constraints to A and B (both active at spawn) -->
        <equality name="{name}_to_A" active="true">
            <weld body1="{name}" body2="{initial_body_a}" relpose="true"/>
        </equality>

        <equality name="{name}_to_B" active="true">
            <weld body1="{name}" body2="{initial_body_b}" relpose="true"/>
        </equality>


        <!-- Equality constraint to attach to screwdriver -->
        <equality name="{name}_to_screwdriver" active="false">
            <weld body1="{name}" body2="{screwdriver_name}" relpose="true"/>
        </equality>
        """


# To remove A/B constraint and activate screwdriver constraint
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_ab"))] = 0
# model.eq_active[model.equality(name_to_id(model, "equality", f"{name}_to_screwdriver"))] = 1
