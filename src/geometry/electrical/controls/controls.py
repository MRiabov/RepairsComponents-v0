"""Control components like buttons and switches."""

from typing import Dict, Any, Optional, Callable, Union

# import mujoco #TODO replace mujoco with Genesis!


def button_mjcf(
    button_radius: float = 10.0, button_thickness: float = 2.0, press_range: float = 0.5
):
    return f"""
    <body name="button_base" pos="0 0 0">
      <geom type="cylinder" size="{button_radius} {button_thickness}" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
      
      <!-- Pressable part -->
      <body name="button_top" pos="0 0 {button_thickness / 2}">
        <joint name="press_joint" type="slide" axis="0 0 1" range="-{press_range} {press_range}"/>
        <geom type="cylinder" size="{button_radius} {press_range}" rgba="1 0 0 1"/>
        
        <!-- Contact site to detect pressing -->
        <site name="button_site" pos="0 0 0" size="{press_range}" />
      </body>
    </body>
    """
