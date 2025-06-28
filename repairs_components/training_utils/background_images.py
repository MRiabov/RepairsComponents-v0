import random


def get_raytracer_renderer():
    return gs.renderers.RayTracer(  # type: ignore
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ImageTexture(
                image_path=select_background_image(),
            ),
        ),
        env_radius=15.0,
        env_euler=(0, 0, 180),
        lights=[
            {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
        ],
    )


def select_background_image():
    textures = ["textures/indoor_bright.png", "textures/indoor_bright.png"]
    return random.choice(textures)
