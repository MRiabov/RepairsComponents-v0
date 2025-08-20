import numpy as np
from build123d import *
from ocp_vscode import show

from repairs_components.geometry.connectors.connectors import Connector, ConnectorsEnum


class USB_C(Connector):
    @property
    def model_id(self) -> int:
        return ConnectorsEnum.USB_TYPE_C.value

    def bd_geometry_male(self, moved_to: VectorLike) -> Part | Compound:
        # note: dimensions from https://www.usb.org/sites/default/files/USB%20Type-C%20Spec%20R2.0%20-%20August%202019.pdf
        # page 46, 47
        shell_h = 2.4
        shell_len = 6.51
        shell_w = 8.25  # incl shell t

        shell_t = 0.225

        # housing
        housing_w = 11
        housing_h = 6
        housing_len = 10

        tongue_h = 0.77 + (0.5)  # 0.5mm opening for easier insertion...
        tongue_w = 6
        with BuildPart() as usb_c:
            housing = Box(housing_len, housing_w, housing_h)
            with BuildSketch(housing.faces().sort_by(Axis.X).first) as opening_sketch:
                SlotOverall(width=shell_w, height=shell_h, mode=Mode.ADD)
                SlotOverall(width=tongue_w, height=tongue_h, mode=Mode.SUBTRACT)

            connector = extrude(opening_sketch.sketch.rotate(Axis.X, -90), shell_len)

        with BuildPart() as terminal_def:
            # x -= 0.5  # can't make opening because equality.
            terminal_center_pos = connector.center(CenterOf.BOUNDING_BOX)
            terminal_center_pos = terminal_center_pos - Vector(
                0.5, 0, 0
            )  # move a little closer
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        connector = Compound(children=[usb_c.part, terminal_def.part])
        self.color_and_label(connector, male=True)
        return connector.moved(Location(moved_to))

    def bd_geometry_female(self, moved_to: VectorLike) -> Part | Compound:
        # note: dimensions from https://www.usb.org/sites/default/files/USB%20Type-C%20Spec%20R2.0%20-%20August%202019.pdf
        # page 42
        shell_t = 0.225  # somewhere 0.20 to 0.25

        shell_h = 2.56 + shell_t * 2
        shell_len = 6.6
        shell_w = 8.34 + shell_t * 2

        # housing
        housing_w = 11
        housing_h = 6
        housing_len = 10

        tongue_h = 0.7
        tongue_w = 6.69

        tongue_offset_from_edge = 1
        with BuildPart() as usb_c:
            housing = Box(housing_w, housing_len, housing_h)
            edge_face = housing.faces().sort_by(Axis.X).first
            with BuildSketch(edge_face) as opening_sketch:
                SlotOverall(width=shell_w, height=shell_h, mode=Mode.ADD)
                SlotOverall(width=tongue_w, height=tongue_h, mode=Mode.SUBTRACT)

            connector = extrude(
                opening_sketch.sketch.rotate(Axis.X, -90),
                -shell_len,
                mode=Mode.SUBTRACT,
            )
            # clear the tongue a little backwards
            with Locations(
                edge_face.center(CenterOf.BOUNDING_BOX)
                + Vector(tongue_offset_from_edge / 2, 0, 0)
            ):
                Box(tongue_offset_from_edge, tongue_w, tongue_h, mode=Mode.SUBTRACT)

        with BuildPart() as terminal_def:
            terminal_center_pos = connector.center(CenterOf.BOUNDING_BOX)
            terminal_center_pos = terminal_center_pos + Vector(
                0.0, 0, 0
            )  # move a little farther
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        connector = Compound(children=[usb_c.part, terminal_def.part])
        self.color_and_label(connector, male=False)
        return connector.moved(Location(moved_to))

    @property
    def terminal_pos_relative_to_center_male(self) -> np.ndarray:
        return np.array([-8.755, 0, 0])

    @property
    def terminal_pos_relative_to_center_female(self) -> np.ndarray:
        return np.array([-2.2, 0, 0])

    @property
    def connected_at_angle(self) -> tuple[float, float, float]:
        return (0, 0, 180)


if __name__ == "__main__":
    show(USB_C(0).bd_geometry_male((0, 0, 0)))
