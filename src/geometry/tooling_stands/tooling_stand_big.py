from build123d import *  # noqa: F403 
# noqa: F405
import ocp_vscode
import build123d

#Note: here Y is depth, X is length and Z is height.
# Constants from tooling_geometry.md
TOOL_SLOT_COUNT = 6
TOOL_SLOT_SIZE = 12  # cm
TOOL_SLOT_DEPTH = 15  # cm
TOOL_SLOT_SPACING = 18  # cm, center-to-center
#stand plate
STAND_PLATE_WIDTH = 60  # cm
STAND_PLATE_DEPTH = 40  # cm
STAND_PLATE_HEIGHT = 20  # cm
#legs
STAND_LEG_HEIGHT = 100  # cm
STAND_LEG_WIDTH = 5  # cm
STAND_LEG_DEPTH = 5  # cm

MOUNT_HOLE_DIAMETER = 0.6  # cm
MAGNET_DIAMETER = 1.0  # cm
MAGNET_DEPTH = 0.3  # cm
ALIGNMENT_PIN_DIAMETER = 0.4  # cm
ALIGNMENT_PIN_HEIGHT = 0.8  # cm

with BuildPart() as tool_stand:
    #stand plate
    with Locations((0,0,STAND_LEG_HEIGHT+STAND_PLATE_HEIGHT/2)):
        stand_plate=Box(STAND_PLATE_WIDTH, STAND_PLATE_DEPTH, STAND_PLATE_HEIGHT)
        #tool slots
        with BuildSketch(stand_plate.faces().sort_by(Axis.Z).last) as tool_slot_sketch:
            with GridLocations(TOOL_SLOT_SPACING, TOOL_SLOT_SPACING, 3, 2):
                slot = Rectangle(TOOL_SLOT_SIZE, TOOL_SLOT_SIZE)
        
        extrude(tool_slot_sketch.sketch, TOOL_SLOT_DEPTH, dir = (0,0,-1),mode=Mode.SUBTRACT)
        
        fillet(tool_stand.edges(Select.LAST),radius=1)
        
        
    #legs
    with Locations((0,0,STAND_LEG_HEIGHT/2)):
        with GridLocations(STAND_PLATE_WIDTH-STAND_LEG_WIDTH, STAND_PLATE_DEPTH-STAND_LEG_DEPTH, 2, 2):
            Box(STAND_LEG_WIDTH, STAND_LEG_DEPTH, STAND_LEG_HEIGHT)
    fillet(tool_stand.edges(),radius=0.5)

    
export_gltf(tool_stand.part,"tool_stand_big.glb")
ocp_vscode.show(tool_stand.part)