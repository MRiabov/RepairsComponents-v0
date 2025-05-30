from build123d import *  # noqa: F403

# noqa: F405
import torch
import torchvision
import os
import ocp_vscode
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

    export_gltf(plate_env.part, "geom_exports/tooling_stands/tool_stand_plate.gltf")
    ocp_vscode.show(plate_env.part)


def genesis_setup():
    # for debug
    import genesis as gs

    gs.init(theme="light")
    scene = gs.Scene()
    plane = scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Mesh())

    camera_1 = scene.add_camera()
    # camera_2 = scene.add_camera()
    rgb, depth, _segmentation, normal = camera_1.render(
        rgb=True, depth=True, segmentation=False, normal=True
    )
    # save images
    # Create output directory if it doesn't exist
    os.makedirs("renders", exist_ok=True)

    # Save RGB image (convert from float [0,1] to uint8 [0,255])
    rgb_uint8 = (rgb * 255).byte().permute(2, 0, 1)  # HWC to CHW
    torchvision.io.write_png(rgb_uint8, "renders/rgb.png")

    # Save depth (normalize to [0,1] for visualization)
    if depth is not None:
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_uint8 = (depth_normalized * 255).byte()
        torchvision.io.write_png(
            depth_uint8.unsqueeze(0), "renders/depth.png"
        )  # Add channel dim

    # Save normal map (convert from [-1,1] to [0,1])
    if normal is not None:
        normal_uint8 = (
            ((normal * 0.5 + 0.5) * 255).byte().permute(2, 0, 1)
        )  # HWC to CHW
        torchvision.io.write_png(normal_uint8, "renders/normal.png")

    print("Renders saved to 'renders/' directory")


plate_env_bd_geometry()
render_genesis = True
if render_genesis:
    genesis_setup()
