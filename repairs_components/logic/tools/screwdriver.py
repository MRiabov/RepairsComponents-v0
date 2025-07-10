from dataclasses import dataclass

import torch
from repairs_components.geometry.b123d_utils import export_obj
from repairs_components.logic.tools.tool import Tool
from repairs_components.logic.tools.tool import attachment_link_name
from pathlib import Path
import trimesh
from build123d import *  # noqa: F403


@dataclass
class Screwdriver(Tool):
    picked_up_fastener_name: str | None = None
    picked_up_fastener: bool = False
    name: str = "screwdriver"  # deprecated

    def step(self, action: torch.Tensor, state: dict):
        raise NotImplementedError

    def dist_from_grip_link(self):
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
                <body name="attachment_link" pos="0 0 0.3">
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


def receive_screw_in_action(actions: torch.Tensor):
    assert actions.shape[1] == 10, (
        "Screwdriver action check expects that action has shape [batch, 9]"
    )
    assert actions.ndim == 2, (
        "Screwdriver action check expects that action has shape [batch, action_dim]"
    )
    screw_in = actions[:, 8]
    assert screw_in.ndim == 1, (
        "Screwdriver action check expects that action has shape [batch]"
    )
    return screw_in > 0.5
