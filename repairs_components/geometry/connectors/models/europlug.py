from build123d import *
from repairs_components.geometry.connectors.connectors import Connector
import ocp_vscode
import build123d as bd
import numpy as np


# Define the dimensions of the IEC plug
plug_length = 50.0
male_plug_width = 30.0
female_plug_width = 50
male_plug_height = 20.0
female_plug_height = 50.0
hole_diameter = 5
pin_diameter = 4.0
pin_dist = 19
pin_len = 19


class Europlug(Connector):
    def bd_geometry_male(self, moved_to: bd.VectorLike):
        """
        Create a simple IEC plug model.

        Returns:
            Part: The IEC plug part.
        """

        with BuildPart() as plug_part:
            # Create the main body of the plug
            plug_body = Box(plug_length, male_plug_width, male_plug_height)
            chamfer(plug_body.edges().filter_by(Axis.Z).sort_by(Axis.X)[:2], 7)

            rightmost = plug_body.faces().sort_by(Axis.X)[-1]
            with BuildSketch(rightmost) as pin_sketch:
                with GridLocations(0, pin_dist, 1, 2):
                    Circle(pin_diameter / 2)
            pins = extrude(pin_sketch.sketch, pin_len)

        connector_def = Part()
        connector_def = extrude(
            pin_sketch.sketch, pin_len, mode=Mode.ADD, target=connector_def
        )
        connector_collision_detection_position = np.array(
            connector_def.center().to_tuple()
        )

        return (
            self.color_and_label(
                plug_part.part.moved(Location(moved_to)),
                connector_def.moved(Location(moved_to)),
            ),
            connector_collision_detection_position,
        )

    def bd_geometry_female(self, moved_to: bd.VectorLike):
        """
        Create a simple IEC socket model with holes for pins.

        Returns:
            Part: The IEC socket part with holes.
        """
        with BuildPart() as socket_part:
            # Create the main body of the socket
            socket_body = Box(plug_length, female_plug_width, female_plug_height)
            fillet(socket_body.faces().filter_by(Axis.X).last.edges(), 7)

            # Create holes for the pins on the rightmost face
            rightmost = socket_body.faces().sort_by(Axis.X)[-1]
            with BuildSketch(rightmost) as hole_sketch:
                with GridLocations(0, pin_dist, 1, 2):
                    Circle(hole_diameter / 2)
            # Extrude the holes into the socket body
            holes = extrude(hole_sketch.sketch, -pin_len - 2, mode=Mode.SUBTRACT)
            base_joint = RigidJoint(
                "native"
            )  # native "base" joint, so will hold despite perturbations.

        connector_def = Part()
        # The connector definition is the negative space where the pins would go
        connector_def = extrude(
            hole_sketch.sketch, -pin_len - 2, mode=Mode.ADD, target=connector_def
        )
        connector_collision_detection_position = np.array(
            connector_def.center().to_tuple()
        )  # a general rule that should work. # to debug if can't be matched.

        return self.color_and_label(
            socket_part.part.moved(Location(moved_to)),
            connector_def.moved(Location(moved_to)),
            connector_collision_detection_position,
        )
