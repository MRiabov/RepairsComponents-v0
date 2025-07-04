from build123d import *  # noqa: F403
from genesis.vis.camera import Camera
import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path

# noqa: F405
import os

from genesis.engine.entities import RigidEntity
import genesis as gs

WORKING_SPACE_SIZE = (64, 64, 64)  # cm


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

# for exporting the environment; moved as the center of the parts.
SCENE_CENTER = (0, 20 + STAND_PLATE_DEPTH / 2, STAND_PLATE_HEIGHT)


def plate_env_bd_geometry(export_geom_gltf: bool, base_dir: Path | None = None) -> Part:
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

    # # on the top of the stand plate
    plate_env_export = scale(plate_env.part, 0.01)  # convert to mm.

    if export_geom_gltf:
        assert base_dir is not None, (
            "base_dir must be provided if export_geom_gltf is True"
        )
        path = export_path(base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        successful_write = export_gltf(plate_env_export, str(path), unit=Unit.M)
        assert successful_write, "Failed to export gltf"
        # print("exported gltf")

    # ocp_vscode.show(plate_env.part)
    return plate_env.part


def genesis_setup(scene: gs.Scene, base_dir: Path):
    # NOTE: in genesis, the YZ is swapped compared to build123d, so define in XZY.

    # Add plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Add mesh with proper scale and position
    # Add mesh with multiple lights and better camera position
    tooling_stand: RigidEntity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(export_path(base_dir)),
            scale=1,  # Use 1.0 scale since we're working in cm
            pos=(0, 0, 0.1),
            euler=(90, 0, 0),  # Rotate 90 degrees around X axis
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 0.7, 0.3, 1)),  # Add color material
    )

    # Add box for reference

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.3, STAND_PLATE_DEPTH / 100 / 2, 0.20),
        ),
    )

    # Set up camera with proper position and lookat
    camera_1 = scene.add_camera(
        # pos=(1, 2.5, 3.5),
        pos=(1, 2.5, 3.5),  # Position camera further away and above
        lookat=(
            0.64 / 2,
            0.64 / 2 + STAND_PLATE_DEPTH / 100,
            0.3,
        ),  # Look at the center of the working pos
        res=(256, 256),  # (1024,1024) for debug
    )

    camera_2 = scene.add_camera(
        pos=(-2.5, 1.5, 1.5),  # second camera from the other side
        lookat=(
            0.64 / 2,
            0.64 / 2 + STAND_PLATE_DEPTH / 100,
            0.3,
        ),  # Look at the center of the working pos
        res=(256, 256),  # (1024,1024) for debug
    )
    entities = {
        "plane": plane,
        "franka_arm": franka,
        "tooling_stand": tooling_stand,
    }

    return (scene, [camera_1, camera_2], entities)


def render_and_save(scene: gs.Scene, camera_1: Camera, camera_2: Camera):
    "Util to debug"
    if not scene.is_built:
        scene.build()
    # update to match definitely match physics to image.
    scene.visualizer.update()

    rgb_1, depth_1, _segmentation, normal_1 = camera_1.render(
        rgb=True, depth=True, segmentation=False, normal=True
    )
    # Render from camera_2
    rgb_2, depth_2, _segmentation_2, normal_2 = camera_2.render(
        rgb=True, depth=True, segmentation=False, normal=True
    )

    # save images
    # Create output directory if it doesn't exist
    os.makedirs("renders", exist_ok=True)

    # Save RGB image (convert from float [0,1] to uint8 [0,255])

    # Function to save images for a camera

    def save_camera_renders(rgb, depth, normal, camera_name):
        # Save RGB image (convert from float [0,1] to uint8 [0,255])
        # Transpose from HWC to CHW
        rgb_uint8 = np.transpose(rgb, (2, 0, 1))
        # Save using PIL
        Image.fromarray(np.transpose(rgb_uint8, (1, 2, 0))).save(
            f"renders/rgb_{camera_name}.png"
        )

        # Save depth (normalize to [0,1] for visualization)
        if depth is not None:  # normalisation is desirable.
            depth_normalized = (depth - depth.min()) / (
                depth.max() - depth.min() + 1e-6
            )
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            # Save using PIL
            Image.fromarray(depth_uint8).save(f"renders/depth_{camera_name}.png")

        # Save normal map (convert from [-1,1] to [0,1])
        if normal is not None:
            normal_normalized = normal * 0.5 + 0.5
            normal_uint8 = (normal_normalized * 255).astype(np.uint8)
            # Transpose from HWC to CHW
            normal_uint8 = np.transpose(normal_uint8, (2, 0, 1))
            # Save using PIL
            Image.fromarray(np.transpose(normal_uint8, (1, 2, 0))).save(
                f"renders/normal_{camera_name}.png"
            )

    # Save renders from both cameras
    save_camera_renders(rgb_1, depth_1, normal_1, "camera1")
    save_camera_renders(rgb_2, depth_2, normal_2, "camera2")
    print("Saved camera outputs!")


def export_path(base_dir: Path):
    return base_dir / "meshes/tooling_stands/tool_stand_plate.gltf"


# plate_env_bd_geometry()
# render_genesis = True
# if render_genesis:
#     gs.init(theme="light")
#     genesis_setup()
