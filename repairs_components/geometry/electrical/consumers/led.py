from genesis import gs
from repairs_components.geometry.electrical.electrical_component import (
    PhysicalElectricalComponent,
)
from typing_extensions import Any


class Led(PhysicalElectricalComponent):
    """A wrapper around a Genesis entity, representing a physical electrical component.
    Use it because of visualize_state"""

    def __init__(
        self,
        name: str,
        gs_scene: gs.Scene,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        max_brightness=1.0,  # set to control max brightness, just in case.
    ):
        self.name = name
        self.size = size
        self.pos = pos
        self.max_brightness = max_brightness
        self.gs_entity = gs_scene.add_entity(
            morph=self.create_geometry(),
            surface=gs.surfaces.Emission(
                # color=(0.8, 0.8, 0.8, 0.1),
                emissive=(1, 1, 1, 0),  # 0 emission.
            ),
        )

    def create_geometry(self):
        return gs.morphs.Box(size=self.size, pos=self.pos)

    def visualize_state(self, state: dict):
        assert state["type"] == "lightbulb"
        original_surface_color = self.gs_entity.surface.color
        self.gs_entity.surface = gs.surfaces.Emission(
            emissive=(
                original_surface_color,
                original_surface_color,
                original_surface_color,
                min(self.max_brightness, state["brightness"]),
            )
        )
        return self.gs_entity  # I hope it preserves reference... if not, this is vain.
