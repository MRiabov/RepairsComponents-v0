from dataclasses import dataclass

import torch
from repairs_components.logic.tools.tool import Tool, ToolsEnum, attachment_link_name
from pathlib import Path
from build123d import *  # noqa: F403


class Multimeter(Tool):
    # A tool that measures voltage, current, and resistance.
    # I think the best use of it would be to be able to send electricity through it and let it measure the resistance.
    # wherever the resistance is too high, it can find a broken piece of electronics.
    id: int = ToolsEnum.MULTIMETER.value

    @staticmethod
    def tool_grip_position():  # TODO rename to uppercase and make var.
        return torch.tensor([0, 0, 0.1])

    @staticmethod  # TODO rename to uppercase and make var.
    def fastener_connector_pos_relative_to_center():  # note: this is a correct value.
        return torch.tensor([0, 0, -(0.1 / 2 + 0.1)])

    def step(self, action: torch.Tensor, state: dict):
        raise NotImplementedError

    @staticmethod  # TODO rename to uppercase and make var.
    def dist_from_grip_link():
        return 5  # 5 meters. for debug.

    def bd_geometry(self, export: bool = True, base_dir: Path | None = None):
        # An automatic end effector for pick-and-place screwdriver.
        BODY_DIAMETER = 20  # mm
        BODY_HEIGHT = 100  # mm

        IRON_DIAMETER = 10  # mm
        IRON_LEN = 100  # mm

        with BuildPart() as multimeter:
            grip = Cylinder(BODY_DIAMETER / 2, BODY_HEIGHT)
            with Locations((0, 0, -(BODY_HEIGHT / 2 + IRON_LEN / 2))):
                iron = Cylinder(IRON_DIAMETER / 2, IRON_LEN)
            last_face = iron.faces().sort_by(Axis.Z).first  # lowest.
            chamfer(last_face.edges(), length=5, angle=60)

        # fastener_connector_pos_relative_to_center = aabb.max.z
        return multimeter.part
