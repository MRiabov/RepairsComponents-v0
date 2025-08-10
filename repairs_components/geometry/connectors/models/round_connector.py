from build123d import *
from ocp_vscode import *
import ocp_vscode
from repairs_components.geometry.connectors.connectors import Connector, ConnectorsEnum
import numpy as np


class RoundConnector(Connector):
    "The kind of connectors often found on pre-Type-C laptops."

    @property
    def model_id(self) -> int:
        return ConnectorsEnum.ROUND_LAPTOP.value

    def bd_geometry_male(self, moved_to: VectorLike) -> Part | Compound:
        # dimensions can be found on e.g. https://coywood.com/Asus-X541-charger-19v-237a#:~:text=Genuine%20Asus%20X541%20charger%2019v,match%20your%20Asus%20notebook%20computer.
        # the dimensions are mostly guessed.
        housing_len = 18
        housing_r = 9

        pin_len = 8
        pin_r = 3.5

        tongue_hole_len = 4.0
        tongue_hole_r = 1.35

        with BuildPart() as male:
            housing = Cylinder(housing_r, housing_len, rotation=(0, 90, 0))

            with Locations(housing.faces().sort_by(Axis.X).first.offset(pin_len / 2)):
                pin = Cylinder(pin_r, pin_len)
                hole = Hole(tongue_hole_r, pin_len)

            chamfer(male.faces().sort_by(Axis.X)[:2].edges(), 0.5)

        with BuildPart() as terminal_def:
            terminal_center_pos = pin.center(CenterOf.BOUNDING_BOX)
            terminal_center_pos = terminal_center_pos - Vector(1, 0, 0)
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        comp = Compound(children=[male.part, terminal_def.part])
        return self.color_and_label(comp, male=True).moved(Location(moved_to))

    def bd_geometry_female(self, moved_to: VectorLike) -> Part | Compound:
        housing_len = 14
        housing_r = 9

        pin_len = 8 - (1)
        pin_r = 3.5 - (0.25)

        tongue_hole_len = 4.0
        tongue_hole_r = 1.35 - (0.35)
        with BuildPart() as female:
            housing = Cylinder(housing_r, housing_len, rotation=(0, 90, 0))

            with Locations(housing.faces().sort_by(Axis.X).first):
                hole = Hole(pin_r, depth=pin_len, mode=Mode.SUBTRACT)
                # note: rotated.
            with Locations(hole.center(CenterOf.BOUNDING_BOX) + Vector(5, 0, 0)):
                inner_pin = Cylinder(
                    tongue_hole_r, tongue_hole_len, rotation=(0, 90, 0)
                )
            chamfer(female.faces().sort_by(Axis.X)[:3].edges(), 0.5)

        with BuildPart() as terminal_def:
            terminal_center_pos = (
                female.faces()
                .sort_by(Axis.X)
                .first.offset(-(pin_len / 2 + 0.5))
                .center(CenterOf.BOUNDING_BOX)
            )
            terminal_center_pos = terminal_center_pos
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        comp = Compound(children=[female.part, terminal_def.part])
        return self.color_and_label(comp, male=False).moved(Location(moved_to))

    @property
    def terminal_pos_relative_to_center_male(self) -> np.ndarray:
        return np.array([-22.625, 0, 0])

    @property
    def terminal_pos_relative_to_center_female(self) -> np.ndarray:
        return np.array([-4, 0, 0])

    @property
    def connected_at_angle(self) -> tuple[float, float, float]:
        return (0, 0, 180)


if __name__ == "__main__":
    show(RoundConnector(0).bd_geometry((0, 0, 0)))
