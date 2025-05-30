from build123d import *  # noqa: F403
import numpy as np
import torch
import torchvision
from PIL import Image

# noqa: F405
import os

# import ocp_vscode
import build123d

import genesis as gs

WORKING_SPACE_SIZE = (128, 128, 128)  # cm


# Note: here Y is depth, X is length and Z is height.
# Constants from tooling_geometry.md
TOOL_SLOT_COUNT = 4
TOOL_SLOT_SIZE = 12  # cm
TOOL_SLOT_DEPTH = 15  # cm
TOOL_SLOT_SPACING = 18  # cm, center-to-center

# fastener slots
FASTENER_HOLE_COUNT = 12
FASTENER_HOLE_DIAMETER = 1
FASTENER_HOLE_DEPTH = 3  # cm
FASTENER_HOLE_SPACING = 4  # cm, center-to-center

# robot arm slot.
ROBOT_ARM_BASE_WIDTH = 20


# stand plate
TOOLING_SPACE_WIDTH = 60 + (ROBOT_ARM_BASE_WIDTH)
FASTENER_SPACE_WIDTH = 20
STAND_PLATE_WIDTH = TOOLING_SPACE_WIDTH + FASTENER_SPACE_WIDTH  # cm
STAND_PLATE_DEPTH = 40  # cm
STAND_PLATE_HEIGHT = 20  # cm


# guard wall
GUARD_WALL_THICKNESS = 1.2  # cm
GUARD_WALL_HEIGHT = 12.0  # cm

MOUNT_HOLE_DIAMETER = 0.6  # cm
MAGNET_DIAMETER = 1.0  # cm
MAGNET_DEPTH = 0.3  # cm
ALIGNMENT_PIN_DIAMETER = 0.4  # cm
ALIGNMENT_PIN_HEIGHT = 0.8  # cm


def plate_env_bd_geometry():
    with BuildPart() as plate_env:
        # stand plate
        with Locations((0, 0, STAND_PLATE_HEIGHT / 2)):
            stand_plate = Box(STAND_PLATE_WIDTH, STAND_PLATE_DEPTH, STAND_PLATE_HEIGHT)
            # tool slots
            with BuildSketch(
                stand_plate.faces().sort_by(Axis.Z).last
            ) as tool_slot_sketch:
                with Locations(
                    (-STAND_PLATE_WIDTH / 2 + TOOLING_SPACE_WIDTH / 2 - 10, 0, 0)
                ):
                    with GridLocations(TOOL_SLOT_SPACING, TOOL_SLOT_SPACING, 3, 2):
                        slot = Rectangle(TOOL_SLOT_SIZE, TOOL_SLOT_SIZE)

            extrude(
                tool_slot_sketch.sketch,
                TOOL_SLOT_DEPTH,
                dir=(0, 0, -1),
                mode=Mode.SUBTRACT,
            )
            fillet(plate_env.edges(Select.LAST), radius=1)

        with Locations(
            stand_plate.faces().sort_by(Axis.Z).last
        ) as fastener_slot_sketch:
            # fastener slots
            with Locations(((STAND_PLATE_WIDTH - FASTENER_SPACE_WIDTH) / 2, 0, 0)):
                with GridLocations(FASTENER_HOLE_SPACING, FASTENER_HOLE_SPACING, 3, 4):
                    slot = Hole(FASTENER_HOLE_DIAMETER / 2, FASTENER_HOLE_DEPTH)

        hole_arcs = plate_env.edges(Select.LAST)

        with Locations(
            (
                0,
                WORKING_SPACE_SIZE[1] / 2 + STAND_PLATE_DEPTH / 2,
                STAND_PLATE_HEIGHT / 2,
            )
        ):
            Box(WORKING_SPACE_SIZE[0], WORKING_SPACE_SIZE[1], STAND_PLATE_HEIGHT)

    # guard rails removed for simplicity of AABB collision detecition..
    # # Left wall
    # with Locations(
    #     (
    #         -STAND_PLATE_WIDTH / 2 + GUARD_WALL_THICKNESS / 2,
    #         0,
    #         STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
    #     )
    # ):
    #     Box(GUARD_WALL_THICKNESS, STAND_PLATE_DEPTH, GUARD_WALL_HEIGHT)
    # Right wall
    # with Locations(
    #     (
    #         STAND_PLATE_WIDTH / 2 - GUARD_WALL_THICKNESS / 2,
    #         0,
    #         STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
    #     )
    # ):
    #     Box(GUARD_WALL_THICKNESS, STAND_PLATE_DEPTH, GUARD_WALL_HEIGHT)
    # # Back wall
    # with Locations(
    #     (
    #         0,
    #         STAND_PLATE_DEPTH / 2 - GUARD_WALL_THICKNESS / 2,
    #         STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
    #     )
    # ):
    #     Box(STAND_PLATE_WIDTH, GUARD_WALL_THICKNESS, GUARD_WALL_HEIGHT)

    # # on the top of the stand plate
    plate_env_export = scale(plate_env.part, 0.01)  # convert to mm.
    export_gltf(
        plate_env_export,
        "/workspace/RepairsComponents-v0/geom_exports/tooling_stands/tool_stand_plate.gltf",
        unit=Unit.M,
    )
    # ocp_vscode.show(plate_env.part)
    print("exported gltf")


def genesis_setup():
    # for debug
    import genesis as gs

    gs.init(theme="light")
    scene = gs.Scene(show_viewer=False)

    # Add plane
    # plane = scene.add_entity(gs.morphs.Plane())

    # Add mesh with proper scale and position
    # Add mesh with multiple lights and better camera position
    tooling_stand = scene.add_entity(
        gs.morphs.Mesh(
            file="/workspace/RepairsComponents-v0/geom_exports/tooling_stands/tool_stand_plate.gltf",
            scale=1,  # Use 1.0 scale since we're working in cm
            pos=(0.05, 0.15, 0.1),
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 0.7, 0.3, 1)),  # Add color material
    )

    # Add box for reference
    box = scene.add_entity(gs.morphs.Box(pos=(0.7, 0.1, 0.1), size=(0.1, 0.1, 0.1)))

    # Set up camera with proper position and lookat
    camera_1 = scene.add_camera(
        pos=(1, 2.5, 3.5),  # Position camera further away and above
        lookat=(0, 0, 0),  # Look at the center of the scene
        res=(1024, 1024),
    )

    scene.build()
    rgb, depth, _segmentation, normal = camera_1.render(
        rgb=True, depth=True, segmentation=False, normal=True
    )

    print(scene.entities)

    # save images
    # Create output directory if it doesn't exist
    os.makedirs("renders", exist_ok=True)

    # Save RGB image (convert from float [0,1] to uint8 [0,255])

    # Transpose from HWC to CHW
    rgb_uint8 = np.transpose(rgb, (2, 0, 1))
    # Save using PIL
    from PIL import Image

    Image.fromarray(np.transpose(rgb_uint8, (1, 2, 0))).save("renders/rgb.png")

    # Save depth (normalize to [0,1] for visualization)
    if depth is not None:
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        # Save using PIL
        Image.fromarray(depth_uint8).save("renders/depth.png")

    # Save normal map (convert from [-1,1] to [0,1])
    if normal is not None:
        normal_normalized = normal * 0.5 + 0.5
        normal_uint8 = (normal_normalized * 255).astype(np.uint8)
        # Transpose from HWC to CHW
        normal_uint8 = np.transpose(normal_uint8, (2, 0, 1))
        # Save using PIL
        Image.fromarray(np.transpose(normal_uint8, (1, 2, 0))).save(
            "renders/normal.png"
        )

    print("Renders saved to 'renders/' directory")


plate_env_bd_geometry()
render_genesis = True
if render_genesis:
    genesis_setup()
