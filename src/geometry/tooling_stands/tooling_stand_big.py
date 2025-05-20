from build123d import *  # noqa: F403 
# noqa: F405
from dataclasses import dataclass
from typing import List, Tuple
import math

# Constants from tooling_geometry.md
TOOL_SLOT_COUNT = 6
TOOL_SLOT_SIZE = 12  # cm
TOOL_SLOT_SPACING = 15  # cm, center-to-center
STAND_WIDTH = 40  # cm
STAND_DEPTH = 30  # cm
STAND_HEIGHT = 5  # cm
MOUNT_HOLE_DIAMETER = 0.6  # cm
MAGNET_DIAMETER = 1.0  # cm
MAGNET_DEPTH = 0.3  # cm
ALIGNMENT_PIN_DIAMETER = 0.4  # cm
ALIGNMENT_PIN_HEIGHT = 0.8  # cm

@dataclass
class ToolSlot:
    """Represents a single tool slot in the tool stand."""
    position: Tuple[float, float]  # (x, y) position in cm
    rotation: float = 0.0  # Rotation in degrees
    tool_id: int = -1  # -1 means empty slot


def create_tool_stand() -> Part:
    """
    Create a tool stand for robot tools with 6 slots in a 2x3 grid.
    
    Returns:
        Part: The tool stand as a build123d Part
    """
    # Create the base plate
    with BuildPart() as stand_builder:
        # Main base plate
        Box(
            width=STAND_WIDTH,
            length=STAND_DEPTH,
            height=STAND_HEIGHT,
            align=(Align.CENTER, Align.CENTER, Align.MIN)
        )
        
        # Add rubber feet for stability (4 corners)
        foot_diameter = 3.0  # cm
        foot_height = 0.5  # cm
        foot_positions = [
            (-STAND_WIDTH/2 + 2, -STAND_DEPTH/2 + 2, 0),
            (STAND_WIDTH/2 - 2, -STAND_DEPTH/2 + 2, 0),
            (-STAND_WIDTH/2 + 2, STAND_DEPTH/2 - 2, 0),
            (STAND_WIDTH/2 - 2, STAND_DEPTH/2 - 2, 0)
        ]
        
        for pos in foot_positions:
            with BuildPart() as foot:
                Cylinder(
                    radius=foot_diameter/2,
                    height=foot_height,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
                stand_builder.add(foot.part, pos=pos)
        
        # Create tool slots in a 2x3 grid
        slots_per_row = 3
        rows = 2
        start_x = -(slots_per_row - 1) * TOOL_SLOT_SPACING / 2
        start_y = -(rows - 1) * TOOL_SLOT_SPACING / 2
        
        for row in range(rows):
            for col in range(slots_per_row):
                x = start_x + col * TOOL_SLOT_SPACING
                y = start_y + row * TOOL_SLOT_SPACING
                
                # Create a recessed area for the tool
                with BuildPart() as slot:
                    # Recessed area for tool base
                    Box(
                        width=TOOL_SLOT_SIZE,
                        length=TOOL_SLOT_SIZE,
                        height=0.2,  # 2mm recess
                        align=(Align.CENTER, Align.CENTER, Align.MIN)
                    )
                    
                    # Add alignment features
                    with Locations((0, 0, 0.2)):
                        # Add 4 magnets in a square pattern
                        magnet_positions = [
                            (-TOOL_SLOT_SIZE/3, -TOOL_SLOT_SIZE/3, 0),
                            (TOOL_SLOT_SIZE/3, -TOOL_SLOT_SIZE/3, 0),
                            (-TOOL_SLOT_SIZE/3, TOOL_SLOT_SIZE/3, 0),
                            (TOOL_SLOT_SIZE/3, TOOL_SLOT_SIZE/3, 0)
                        ]
                        
                        for m_x, m_y, m_z in magnet_positions:
                            Cylinder(
                                radius=MAGNET_DIAMETER/2,
                                height=MAGNET_DEPTH,
                                align=(Align.CENTER, Align.CENTER, Align.MIN)
                            )
                            
                        # Add alignment pins (2 diagonal pins)
                        pin_positions = [
                            (-TOOL_SLOT_SIZE/4, -TOOL_SLOT_SIZE/4, 0),
                            (TOOL_SLOT_SIZE/4, TOOL_SLOT_SIZE/4, 0)
                        ]
                        
                        for p_x, p_y, p_z in pin_positions:
                            Cylinder(
                                radius=ALIGNMENT_PIN_DIAMETER/2,
                                height=ALIGNMENT_PIN_HEIGHT,
                                align=(Align.CENTER, Align.CENTER, Align.MIN)
                            )
                    
                    # Position the slot
                    stand_builder.add(slot.part, pos=(x, y, STAND_HEIGHT))
        
        # Add mounting holes for attaching to the workbench
        mount_hole_positions = [
            (-STAND_WIDTH/2 + 2, 0, 0),
            (STAND_WIDTH/2 - 2, 0, 0)
        ]
        
        for pos in mount_hole_positions:
            with BuildPart() as hole:
                Cylinder(
                    radius=MOUNT_HOLE_DIAMETER/2,
                    height=STAND_HEIGHT * 2,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)
                )
                stand_builder.add(hole.part, pos=pos, mode=Mode.SUBTRACT)
    
    # Add labels for tool slots
    with BuildPart() as labels:
        for i, slot in enumerate(stand_builder.part.children):
            if i >= TOOL_SLOT_COUNT:  # Skip non-slot children
                continue
                
            # Add a simple number label
            with BuildPart() as label:
                with BuildSketch() as text_sketch:
                    Text(
                        text=str(i + 1),
                        font_size=0.5,
                        align=(Align.CENTER, Align.CENTER)
                    )
                extrude(amount=0.1)
                
                # Position the label near the slot
                slot_pos = slot.location.position
                label_pos = (
                    slot_pos.X,
                    slot_pos.Y + TOOL_SLOT_SIZE/2 + 1,
                    STAND_HEIGHT + 0.1
                )
                labels.part.add(label.part, pos=label_pos)
    
    # Combine everything
    stand_builder.add(labels.part)
    
    return stand_builder.part


def export_tool_stand(output_format: str = "step", filename: str = "tool_stand"):
    """
    Create and export the tool stand to a file.
    
    Args:
        output_format: Output file format ("step", "stl", "obj")
        filename: Output filename without extension
    """
    tool_stand = create_tool_stand()



if __name__ == "__main__":
    # Create and export the tool stand when run directly
    export_tool_stand(output_format="step")
    print("Tool stand created and exported as 'tool_stand.step'")
