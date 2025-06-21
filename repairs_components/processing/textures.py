from repairs_components.processing.translation import PART_TYPE_COLORS
from genesis import surfaces
import random


def get_random_texture(part_type: str):
    color_surfaces = [
        surfaces.Plastic(color=tuple(PART_TYPE_COLORS[k][:3])) for k in PART_TYPE_COLORS
    ]
    metal_surfaces = [surfaces.Aluminium(), surfaces.Copper(), surfaces.Iron()]

    return random.choice(color_surfaces + metal_surfaces)


def get_color_by_type(part_type: str):
    return PART_TYPE_COLORS[part_type][:3]
