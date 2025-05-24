"""Control components like buttons and switches."""

from dataclasses import dataclass
from geometry.base import Component
from build123d import *


# import mujoco #TODO replace mujoco with Genesis!
@dataclass
class Button(Component):
    name: str
    button_radius: float = 10.0
    button_thickness: float = 4.0
    press_range: float = 6
    base_thickness = 5

    def get_mjcf(self):
        return f"""
      <body name="{self.name}_base" pos="0 0 0">
        <geom type="cylinder" size="{self.button_radius} {self.button_thickness}" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
        
        <!-- Pressable part -->
        <body name="{self.name}_top" pos="0 0 {self.button_thickness / 2}">
          <joint name="{self.name}_press_joint" type="slide" axis="0 0 1" range="-{self.press_range} {self.press_range}"/>
          <geom type="cylinder" size="{self.button_radius} {self.press_range}" rgba="1 0 0 1"/>
          
          <!-- Contact site to detect pressing -->
          <site name="{self.name}_site" pos="0 0 0" size="{self.press_range}" />
        </body>
      </body>
      """

    def bd_geometry(self):
        # Create the base cylinder
        base = Cylinder(self.button_radius, self.base_thickness)

        # Create the pressable top part positioned above the base

        top = Cylinder(self.button_radius, self.button_thickness)
        top = top.moved(
            Pos(
                0, 0, self.press_range + self.base_thickness + self.button_thickness / 2
            )
        )

        # color and comprehension information:
        base.color = Color(0.4, 0.4, 0.4, 1)
        top.color = Color(1, 0, 0, 1)
        base.label = f"{self.name}_base@solid"
        top.label = f"{self.name}_top@button"

        # Return the combined geometry
        return Compound(children=[base, top])


@dataclass
class Switch(Component):
    name: str
    switch_radius: float = 10.0
    switch_thickness: float = 6.0
    base_thickness: float = 8

    def get_mjcf(self):
        lever_length = self.switch_radius * 2
        lever_width = self.switch_thickness
        lever_height = self.switch_thickness * 1.2
        lever_pos_z = self.switch_thickness * 0.75
        lever_rot_deg = 20  # rotate around y
        base_x = self.switch_radius * 2 + 2
        base_y = self.base_thickness
        base_z = self.base_thickness
        # MJCF does not support boolean subtraction, so use a single box for the base
        return f'''
  <body name="{self.name}_base" pos="0 0 0">
    <geom type="box" size="{base_x/2} {base_y/2} {base_z/2}" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
    <!-- Lever part -->
    <body name="{self.name}_lever" pos="0 0 {lever_pos_z}">
      <joint name="{self.name}_hinge" type="hinge" axis="0 1 0" pos="0 0 0" range="-30 30" damping="1" frictionloss="0.1"/>
      <geom type="box" size="{lever_length/2} {lever_width/2} {lever_height/2}" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"
        euler="0 {lever_rot_deg} 0"/>
      <!-- Contact site to detect lever press -->
      <site name="{self.name}_site" pos="{lever_length/2} 0 {lever_height/2}" size="1"/>
      <!-- Sensor to read hinge angle -->
      <sensor name="{self.name}_hinge_pos" type="jointpos" joint="{self.name}_hinge"/>
    </body>
  </body>'''

    def bd_geometry(self):
        # Create the moving part of the switch as a lever
        with BuildPart() as switch_moving_part:
            with Locations(Pos(0, 0, self.switch_thickness * 0.75)):
                Box(
                    self.switch_radius * 2,
                    self.switch_thickness,
                    self.switch_thickness * 1.2,
                    rotation=(0, 20, 0),
                )
            switch_moving_part.part.color = Color(0.8, 0.2, 0.2, 1)  # dark red
            switch_moving_part.part.label = f"{self.name}_moving@switch"

        # Create the base of the switch
        with BuildPart() as switch_base:
            # Outer box
            Box(
                self.switch_radius * 2 + 2,
                self.base_thickness,
                self.base_thickness,
                mode=Mode.ADD,
            )
            # Inner cutout
            Box(
                self.switch_radius * 2,
                self.base_thickness - 2,
                self.base_thickness - 2,
                mode=Mode.SUBTRACT,
            )
            switch_base.part.color = Color(0.4, 0.4, 0.4, 1)
            switch_base.part.label = f"{self.name}_base@solid"

        # Combine both parts
        return Compound(children=[switch_base.part, switch_moving_part.part])
