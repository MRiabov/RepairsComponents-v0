import numpy as np
from build123d import *
from ocp_vscode import show

from repairs_components.geometry.connectors.connectors import Connector, ConnectorsEnum


class USB_A(Connector):
    @property
    def model_id(self) -> int:
        return ConnectorsEnum.USB_TYPE_A.value

    def bd_geometry_male(self, moved_to: VectorLike) -> Part | Compound:
        # note: dimensions are from https://www.usb.org/sites/default/files/CCWG_A_Plug_Form_Factor_Guideline_Revision_1.0_.pdf
        # USB-A male metal shell
        shell_w = 12.0
        shell_h = 4.5
        shell_len = 11.5

        # Housing (outer user-touchable shell)
        housing_w = 31.75
        housing_h = 8.0
        housing_len = 25.0

        shell_t = 0.315

        tongue_h = shell_h - shell_t * 2 - 1.95

        # top holes
        top_hole_w = 2.5
        top_hole_len = 2
        top_hole_y_dist = 1 / 2 * 2 + 1 * 2 + 2.5 / 2 * 2  # calculated.
        # top_hole_dist_from_edge = 5.16 #should be but looks good enough.

        with BuildPart() as usb_a:
            housing = Box(housing_w, housing_len, housing_h)
            with BuildSketch(housing.faces().sort_by(Axis.X).first) as opening_sketch:
                RectangleRounded(shell_w, shell_h, radius=shell_t, mode=Mode.ADD)
                RectangleRounded(
                    shell_w - shell_t * 2,
                    shell_h - shell_t * 2,
                    radius=shell_t,
                    mode=Mode.SUBTRACT,
                )
                with Locations(Vector(0, housing_h / 4)):  # todo: proper tongue h.
                    Rectangle(shell_w - shell_t * 2, tongue_h, Mode.SUBTRACT)

            connector = extrude(opening_sketch.sketch.rotate(Axis.X, -90), shell_len)
            # two top holes
            with Locations(
                connector.faces()
                .filter_by(Axis.Z)
                .sort_by(Axis.Z)
                .last.offset(-shell_t / 2)
            ):
                with GridLocations(top_hole_y_dist, 0, 2, 1):
                    Box(top_hole_w, top_hole_len, shell_t, mode=Mode.SUBTRACT)

        with BuildPart() as terminal_def:
            terminal_center_pos = (
                tuple(connector.center(CenterOf.BOUNDING_BOX))[0] - 1,
                0,
                0,
            )  # for whichever reason, terminal def z is 0.355 and I don't need it. so reset it.
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        connector = Compound(children=[usb_a.part, terminal_def.part])
        self.color_and_label(connector, male=True)

        return connector.moved(Location(moved_to))

    def bd_geometry_female(self, moved_to: VectorLike) -> Part | Compound:
        # # USB-A female shell (receptacle)
        # usb_a_female_w = 12.2
        # usb_a_female_h = 5.0
        # usb_a_female_len = 13.7

        # # Housing
        # housing_w = 14.0
        # housing_h = 7.0
        # housing_len = 25.0

        # # Inner slot
        # tongue_w = 8.0
        # tongue_h = 2.0
        # tongue_len = 10

        # USB-A male metal shell
        shell_w = 12.2
        shell_h = 5.0
        shell_len = 14

        # Housing (outer user-touchable shell)
        housing_w = 20
        housing_h = 8.0
        housing_len = 25.0

        shell_t = 0.315

        tongue_h = 1
        tongue_offset_from_edge = 2
        tongue_len = shell_len - tongue_offset_from_edge
        tongue_w = shell_w - 2

        # # top holes
        # top_hole_w = 2.5
        # top_hole_len = 2
        # top_hole_y_dist = 1 / 2 * 2 + 1 * 2 + 2.5 / 2 * 2  # calculated.
        # # top_hole_dist_from_edge = 5.16 #should be but looks good enough.

        # tongue
        tongue_h = 2
        with BuildPart() as usb_a:
            housing = Box(housing_w, housing_len, housing_h)
            with BuildSketch(housing.faces().sort_by(Axis.X).first) as opening_sketch:
                RectangleRounded(shell_w, shell_h, radius=shell_t, mode=Mode.ADD)
            connector = extrude(
                opening_sketch.sketch.rotate(Axis.X, -90),
                amount=-shell_len,
                mode=Mode.SUBTRACT,
            )

            with Locations(
                (
                    tuple(connector.center(CenterOf.BOUNDING_BOX))[0]
                    + tongue_offset_from_edge,
                    0,
                    (1.5),
                )
            ):
                Box(tongue_len, tongue_w, tongue_h, mode=Mode.ADD)

        with BuildPart() as terminal_def:
            terminal_center_pos = connector.center(CenterOf.BOUNDING_BOX)
            terminal_center_pos = terminal_center_pos - Vector(1, 0, 0)
            with Locations(terminal_center_pos):
                Sphere(self._terminal_def_size)

        usb_a = Compound(children=[usb_a.part, terminal_def.part])
        self.color_and_label(usb_a, male=False)
        return usb_a.moved(Location(moved_to))

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
    show(USB_A(0).bd_geometry_male((0, 0, 0)))
