from build123d import *
from ocp_vscode import show
from repairs_components.geometry.connectors.connectors import Connector, ConnectorsEnum
import numpy as np


class Powerpole(Connector):
    "Anderson Powerpole connectors. Note, the male and female connectors equal."

    @property
    def model_id(self) -> int:
        return ConnectorsEnum.POWERPOLE.value

    def bd_geometry_male(self, moved_to: VectorLike) -> Part | Compound:
        # Basic Powerpole-like dimensions (mm)
        body_x = 24.6
        body_y = 7.9
        body_z = 7.9

        # top opening (rotated) (z=x)
        top_opening_x = body_x / 2
        top_opening_y = body_y / 2 - 0.5
        opening_z = body_z / 2

        with BuildPart() as powerpole:
            body = Box(body_x, body_y, body_z)
            with BuildSketch(body.faces().sort_by(Axis.X).first) as opening_sketch:
                with Locations((0, body_y / 4 - 1)):
                    Rectangle(body_y - 1, body_y / 2 + 1, mode=Mode.ADD)

                with Locations((0, -body_y / 4)):
                    Rectangle(body_y, body_y / 2 + 1, mode=Mode.ADD)
                    Rectangle(
                        body_y - 1 + (-1), body_y / 2 - 1 + (-1), mode=Mode.SUBTRACT
                    )
                    # +(-1) to account for insertion. so 1mm tolerance here.
            opening_sketch = opening_sketch.sketch.rotate(Axis.X, 90)
            opening = extrude(opening_sketch, -body_x / 2, mode=Mode.SUBTRACT)
            # rear opening (cable entry)
            with Locations((body_x / 4 + 1, 0, 0)):
                Box(body_x / 2 + 1, body_y - 1, body_z - 1, mode=Mode.SUBTRACT)
            with Locations((0, 0, body_z / 4 - 0.5)):
                Box(2, body_y - 1, body_z / 2, mode=Mode.SUBTRACT)

        with BuildPart() as terminal_def:
            # x -= 0.5  # can't make opening because equality.
            with Locations((-body_x / 4 - (0.5), 0, 0)):
                Sphere(
                    self._terminal_def_size
                )  # just get the center of this sphere later.

        powerpole_comp = Compound(children=[powerpole.part, terminal_def.part])
        self.color_and_label(powerpole_comp, male=True)
        return powerpole_comp.moved(Location(moved_to))

    def bd_geometry_female(self, moved_to: VectorLike) -> Part | Compound:
        male = self.bd_geometry_male((0, 0, 0))  # equal!
        # female = male.moved(Location(moved_to, (180, 0, 0)))
        # actually, it doesn't need to be rotated. It'll be rotated wherever necessary.
        self.color_and_label(male, male=False)

        return male

    @property
    def terminal_pos_relative_to_center_male(self) -> np.ndarray:
        return np.array([-24.6 / 4 - 0.5, 0.0, 0.0], dtype=np.float32)

    @property
    def terminal_pos_relative_to_center_female(self) -> np.ndarray:
        return self.terminal_pos_relative_to_center_male

    def connected_at_angle(self) -> tuple[float, float, float]:
        return (0, 0, 180)


if __name__ == "__main__":
    # show(Powerpole(0).bd_geometry_male((0, 0, 0)))
    show(Powerpole(0).bd_geometry((0, 0, 0)))
