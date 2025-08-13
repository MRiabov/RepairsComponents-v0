from dataclasses import field

import torch
from repairs_components.logic.tools.tool import Tool, ToolsEnum, attachment_link_name
from pathlib import Path
from build123d import *  # noqa: F403
from tensordict import TensorClass


class Screwdriver(Tool, TensorClass):
    @property
    def id(self):
        return ToolsEnum.SCREWDRIVER.value

    picked_up_fastener_tip_position: torch.Tensor = field(
        default_factory=lambda: torch.full((1, 3), float("nan"))
    )
    picked_up_fastener_quat: torch.Tensor = field(
        default_factory=lambda: torch.full((1, 4), float("nan"))
    )
    # Numeric id for picked-up fastener. Use -1 to denote "none".
    picked_up_fastener_id: torch.Tensor = field(
        default_factory=lambda: torch.tensor([-1], dtype=torch.long)
    )

    @property
    def has_picked_up_fastener(self):
        # True when an id is present (>= 0). Returns a boolean tensor per env.
        return self.picked_up_fastener_id >= 0

    # @property # oudated syntax.
    # def picked_up_fastener_name(self):
    #     "Just a wrapper to make code more readable."
    #     from repairs_components.geometry.fasteners import Fastener

    #     return [
    #         Fastener.fastener_name_in_simulation(fastener_id.item())
    #         if not torch.isnan(fastener_id)
    #         else None
    #         for fastener_id in self.picked_up_fastener_id
    #     ]

    def picked_up_fastener_name(self, env_ids: torch.Tensor):
        "Just a wrapper to make code more readable."
        from repairs_components.geometry.fasteners import Fastener

        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(torch.long)
        assert (self.picked_up_fastener_id[env_ids] >= 0).all(), (
            "picked_up_fastener_id should be non-negative when requesting a name"
        )
        names = [
            Fastener.fastener_name_in_simulation(fastener_id.item())
            for fastener_id in self.picked_up_fastener_id[env_ids]
        ]
        if env_ids.numel() == 1:
            return names[0]
        return names

    @property
    def tool_grip_position(self):  # TODO rename to uppercase and make var.
        return torch.tensor([0, 0, 0.15])  # 0.3m?

    @staticmethod  # TODO rename to uppercase and make var.
    def fastener_connector_pos_relative_to_center():
        return torch.tensor([0, 0, -0.2])  # 0.2m?

    def step(self, action: torch.Tensor, state: dict):
        raise NotImplementedError

    @property
    def dist_from_grip_link(self):
        return torch.tensor(5.0)  # meters, for debug.

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

    def on_tool_release(self, env_idx: torch.Tensor):
        self.picked_up_fastener_tip_position[env_idx] = torch.full((3,), float("nan"))
        self.picked_up_fastener_quat[env_idx] = torch.full((4,), float("nan"))
        self.picked_up_fastener_id[env_idx] = torch.tensor(-1, dtype=torch.long)


def receive_screw_in_action(
    actions: torch.Tensor,
    screw_in_threshold: float = 0.75,
    screw_out_threshold: float = 0.25,
):
    assert actions.shape[1] == 10, (
        "Screwdriver action check expects that action has shape [batch, 10]"
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
        "Screwdriver action check expects that action has shape [batch, 10]"
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
