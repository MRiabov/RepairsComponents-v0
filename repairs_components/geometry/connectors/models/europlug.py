from pathlib import Path
from build123d import *
from repairs_components.geometry.connectors.connectors import Connector, ConnectorsEnum
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
    @property
    def model_id(self) -> int:
        return ConnectorsEnum.EUROPLUG.value

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

        with BuildPart() as connector_center:
            with Locations(pins.center(CenterOf.BOUNDING_BOX)):
                Sphere(
                    self._connector_def_size
                )  # just get the center of this sphere later.

        plug_part = Compound(children=[plug_part.part, connector_center.part])

        return self.color_and_label(plug_part.moved(Location(moved_to)), male=True)

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
            rightmost = socket_body.faces().filter_by(Axis.X).sort_by(Axis.X).last
            with BuildSketch(rightmost) as hole_sketch:
                with GridLocations(0, pin_dist, 1, 2):
                    hole_circles = Circle(hole_diameter / 2)

            # Extrude the holes into the socket body
            holes = extrude(hole_sketch.sketch, -pin_len - 2, mode=Mode.SUBTRACT)

            base_joint = RigidJoint(
                "native"  # FIXME: joints were deprecated.
            )  # native "base" joint, so will hold despite perturbations.
            # connector_center = Locations(holes.center(CenterOf.BOUNDING_BOX))

        with BuildPart() as connector_center:
            x, y, z = holes.center(CenterOf.BOUNDING_BOX)
            x += 1.5  # make it easier to plug in.
            with Locations((x, y, z)):
                Sphere(
                    self._connector_def_size
                )  # just get the center of this sphere later.

        socket_part = Compound(
            children=[socket_part.part, connector_center.part],
            joints={"native": base_joint},
        )

        return self.color_and_label(socket_part.moved(Location(moved_to)), male=False)

    @property
    def connector_pos_relative_to_center_male(self) -> np.ndarray:
        return np.array([0, 0, 0])

    @property
    def connector_pos_relative_to_center_female(self) -> np.ndarray:
        return np.array([0, 0, 0])


if __name__ == "__main__":
    from ocp_vscode import show

    show(Europlug(0).bd_geometry((0, 0, 0), connected=True))
