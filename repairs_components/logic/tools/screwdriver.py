from dataclasses import dataclass

import torch
from repairs_components.logic.tools.tool import Tool, ToolsEnum, attachment_link_name
from pathlib import Path
from build123d import *  # noqa: F403


@dataclass
class Screwdriver(Tool):
    id: int = ToolsEnum.SCREWDRIVER.value
    picked_up_fastener_name: str | None = None
    has_picked_up_fastener: bool = False
    picked_up_fastener_tip_position: torch.Tensor | None = None

    @staticmethod
    def tool_grip_position():  # TODO rename to uppercase and make var.
        return torch.tensor([0, 0, 0.15])  # 0.3m?

    @staticmethod  # TODO rename to uppercase and make var.
    def fastener_connector_pos_relative_to_center():
        return torch.tensor([0, 0, -0.2])  # 0.2m?

    def step(self, action: torch.Tensor, state: dict):
        raise NotImplementedError

    @staticmethod  # TODO rename to uppercase and make var.
    def dist_from_grip_link():
        return 5  # 5 meters. for debug.

    def get_mjcf(self, base_dir: Path):
        # Get OBJ file
        obj_path = self.export_path(base_dir, "glb")
        return f"""
        <mujoco>
            <asset>
                <mesh name="tool_mesh" file="{obj_path}"/>
            </asset>
            <worldbody>
                <geom type="mesh" mesh="tool_mesh"/>
                <body name="attachment_link" pos="0 0 0.15">
                    <joint name="{attachment_link_name}" type="free"/>
                </body>
            </worldbody>
        </mujoco>
    """

    def bd_geometry(self, export: bool = True, base_dir: Path | None = None):
        # An automatic end effector for pick-and-place screwdriver.
        BODY_DIAMETER = 100
        BODY_HEIGHT = 150
        BOTTOM_HOLE_DIAMETER = 10
        BOTTOM_FILLET_RADIUS = 40

        END_EFFECTOR_ATTACHMENT_HOLE_DIAMETER = 50
        END_EFFECTOR_ATTACHMENT_HOLE_DEPTH = 30

        with BuildPart() as auto_screwdriver:
            body = Cylinder(BODY_DIAMETER / 2, BODY_HEIGHT)
            lower_face = body.faces().sort_by(Axis.Z)[0]
            fillet(lower_face.edges(), radius=BOTTOM_FILLET_RADIUS)
            with Locations(lower_face):
                Hole(BOTTOM_HOLE_DIAMETER / 2, 30)
            with Locations(body.faces().sort_by(Axis.Z)[-1]):
                Hole(
                    END_EFFECTOR_ATTACHMENT_HOLE_DIAMETER / 2,
                    END_EFFECTOR_ATTACHMENT_HOLE_DEPTH,
                )
        auto_screwdriver.part.label = "screwdriver@control"
        return auto_screwdriver.part

    def export_path(self, base_dir: Path, file_extension: str = "glb") -> Path:
        return base_dir / f"shared/tools/screwdriver.{file_extension}"

    def on_tool_release(self):
        self.has_picked_up_fastener = False
        self.picked_up_fastener_name = None
        self.picked_up_fastener_tip_position = None


def receive_screw_in_action(
    actions: torch.Tensor,
    screw_in_threshold: float = 0.75,
    screw_out_threshold: float = 0.25,
):
    assert actions.shape[1] == 10, (
        "Screwdriver action check expects that action has shape [batch, 9]"
    )
    assert actions.ndim == 2, (
        "Screwdriver action check expects that action has shape [batch, action_dim]"
    )
    screw_in_action = actions[:, 8]
    assert screw_in_action.ndim == 1, (
        "Screwdriver action check expects that action has shape [batch]"
    )
    screw_in_mask = screw_in_action > screw_in_threshold
    screw_out_mask = screw_in_action < screw_out_threshold
    return screw_in_mask, screw_out_mask


def receive_fastener_pickup_action(
    actions: torch.Tensor,
    pick_up_threshold: float = 0.75,
    release_threshold: float = 0.25,
):
    assert actions.shape[1] == 10, (
        "Screwdriver action check expects that action has shape [batch, 9]"
    )
    assert actions.ndim == 2, (
        "Screwdriver action check expects that action has shape [batch, action_dim]"
    )
    screw_in_action = actions[:, 7]  # note: maybe 7. Maybe not.
    assert screw_in_action.ndim == 1, (
        "Screwdriver action check expects that action has shape [batch]"
    )
    screw_in_mask = screw_in_action > pick_up_threshold
    screw_out_mask = screw_in_action < release_threshold
    return screw_in_mask, screw_out_mask
