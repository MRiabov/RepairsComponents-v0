# noqa: F405
import ocp_vscode
from build123d import *  # noqa: F403

# Note: here Y is depth, X is length and Z is height.
# Constants from tooling_geometry.md
TOOL_SLOT_COUNT = 6
TOOL_SLOT_SIZE = 12  # cm
TOOL_SLOT_DEPTH = 15  # cm
TOOL_SLOT_SPACING = 18  # cm, center-to-center

# fastener slots
FASTENER_HOLE_COUNT = 12
FASTENER_HOLE_DIAMETER = 1
FASTENER_HOLE_DEPTH = 3  # cm
FASTENER_HOLE_SPACING = 4  # cm, center-to-center

# stand plate
TOOLING_SPACE_WIDTH = 60
FASTENER_SPACE_WIDTH = 20
STAND_PLATE_WIDTH = TOOLING_SPACE_WIDTH + FASTENER_SPACE_WIDTH  # cm
STAND_PLATE_DEPTH = 40  # cm
STAND_PLATE_HEIGHT = 20  # cm
# legs
STAND_LEG_HEIGHT = 100  # cm
STAND_LEG_WIDTH = 5  # cm
STAND_LEG_DEPTH = 5  # cm

# guard wall
GUARD_WALL_THICKNESS = 1.2  # cm
GUARD_WALL_HEIGHT = 12.0  # cm

MOUNT_HOLE_DIAMETER = 0.6  # cm
MAGNET_DIAMETER = 1.0  # cm
MAGNET_DEPTH = 0.3  # cm
ALIGNMENT_PIN_DIAMETER = 0.4  # cm
ALIGNMENT_PIN_HEIGHT = 0.8  # cm

with BuildPart() as tool_stand:
    # stand plate
    with Locations((0, 0, STAND_LEG_HEIGHT + STAND_PLATE_HEIGHT / 2)):
        stand_plate = Box(STAND_PLATE_WIDTH, STAND_PLATE_DEPTH, STAND_PLATE_HEIGHT)
        # tool slots
        with BuildSketch(stand_plate.faces().sort_by(Axis.Z).last) as tool_slot_sketch:
            with Locations((-STAND_PLATE_WIDTH / 2 + TOOLING_SPACE_WIDTH / 2, 0, 0)):
                with GridLocations(TOOL_SLOT_SPACING, TOOL_SLOT_SPACING, 3, 2):
                    slot = Rectangle(TOOL_SLOT_SIZE, TOOL_SLOT_SIZE)

        extrude(
            tool_slot_sketch.sketch, TOOL_SLOT_DEPTH, dir=(0, 0, -1), mode=Mode.SUBTRACT
        )
        fillet(tool_stand.edges(Select.LAST), radius=1)

    # Left wall
    with Locations(
        (
            -STAND_PLATE_WIDTH / 2 + GUARD_WALL_THICKNESS / 2,
            0,
            STAND_LEG_HEIGHT + STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
        )
    ):
        Box(GUARD_WALL_THICKNESS, STAND_PLATE_DEPTH, GUARD_WALL_HEIGHT)
    # Right wall
    with Locations(
        (
            STAND_PLATE_WIDTH / 2 - GUARD_WALL_THICKNESS / 2,
            0,
            STAND_LEG_HEIGHT + STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
        )
    ):
        Box(GUARD_WALL_THICKNESS, STAND_PLATE_DEPTH, GUARD_WALL_HEIGHT)
    # Back wall
    with Locations(
        (
            0,
            STAND_PLATE_DEPTH / 2 - GUARD_WALL_THICKNESS / 2,
            STAND_LEG_HEIGHT + STAND_PLATE_HEIGHT + GUARD_WALL_HEIGHT / 2,
        )
    ):
        Box(STAND_PLATE_WIDTH, GUARD_WALL_THICKNESS, GUARD_WALL_HEIGHT)

    # legs
    with Locations((0, 0, STAND_LEG_HEIGHT / 2)):
        with GridLocations(
            STAND_PLATE_WIDTH - STAND_LEG_WIDTH,
            STAND_PLATE_DEPTH - STAND_LEG_DEPTH,
            2,
            2,
        ):
            Box(STAND_LEG_WIDTH, STAND_LEG_DEPTH, STAND_LEG_HEIGHT)

    fillet(tool_stand.edges(), radius=0.5)

    # on the top of the stand plate
    with Locations(stand_plate.faces().sort_by(Axis.Z).last) as fastener_slot_sketch:
        # fastener slots
        with Locations(((STAND_PLATE_WIDTH - FASTENER_SPACE_WIDTH) / 2, 0, 0)):
            with GridLocations(FASTENER_HOLE_SPACING, FASTENER_HOLE_SPACING, 3, 4):
                slot = Hole(FASTENER_HOLE_DIAMETER / 2, FASTENER_HOLE_DEPTH)

    hole_arcs = tool_stand.edges(Select.LAST)

export_gltf(tool_stand.part, "geom_exports/tool_stand_big.gltf")
ocp_vscode.show(tool_stand.part)
