"""Module for exporting 3D parts to voxel grids with priority-based handling.

This module provides functionality to convert 3D parts into a voxel grid representation
with support for part type prioritization, where certain part types (like connectors)
can take precedence over others in overlapping regions.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import trimesh
from build123d import Shape, Compound, Color, export_stl, Mesher
import tempfile
import os

# Define part type to color mapping (RGBA format)
PART_TYPE_COLORS = {
    "connector": [1, 1, 0, 0.8],  # Yellow
    "solid": [0.5, 0.5, 0.5, 0.8],  # Grey
    "fluid": [0, 0, 1, 0.5],  # Semi-transparent blue
    "LED": [0, 1, 0, 0.8],  # Green
    "default": [1, 0, 0, 0.8],  # Red (for any other type)
}

# Define part type priority (higher number = higher priority)
PART_TYPE_PRIORITY = {
    "connector": 4,  # Highest priority
    "LED": 3,
    "fluid": 2,
    "solid": 1,
    "default": 0,  # Lowest priority
}


def export_voxel_grid(
    parts: List[Shape | Compound],
    voxel_size: float = 1.0,
    use_part_priority: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Export a list of 3D parts to a voxel grid with optional priority-based handling.

    Args:
        parts: List of build123d Shape or Compound objects to voxelize
        voxel_size: Size of each voxel in the output grid
        use_part_priority: If True, use priority-based handling for overlapping parts

    Returns:
        Tuple containing:
            - 3D numpy array with voxel values (0 = empty, 1+ = part labels)
            - Dictionary with metadata including part types and colors
    """
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export parts to STL and load with trimesh
        temp_files = []
        meshes = []

        for i, part in enumerate(parts):
            # Get part name or use index
            name = getattr(part, "label", f"part_{i}")
            part_type = _extract_part_type(name)

            # Export to STL
            temp_path = os.path.join(temp_dir, f"part_{i}.stl")
            export_stl(part, temp_path)
            temp_files.append(temp_path)

            # Load with trimesh
            mesh = trimesh.load(temp_path)
            if isinstance(mesh, trimesh.Scene):
                # If the part is a compound, we'll get a scene
                mesh = mesh.dump(concatenate=True)

            # Store mesh with metadata
            mesh.metadata = {
                "name": name,
                "part_type": part_type,
                "priority": _get_part_priority(part_type),
            }
            meshes.append(mesh)

    if not meshes:
        raise ValueError("No valid meshes were generated from the input parts")

    # Compute overall bounding box
    all_bounds = [mesh.bounds for mesh in meshes]
    mins = np.min([b[0] for b in all_bounds], axis=0)
    maxs = np.max([b[1] for b in all_bounds], axis=0)

    # Compute the shape of the common grid
    bbox_size = maxs - mins
    shape = np.ceil(bbox_size / voxel_size).astype(int)
    origin = mins

    # Sort meshes by priority (lowest priority first if using priority)
    if use_part_priority:
        meshes.sort(key=lambda m: m.metadata["priority"])

    # Initialize grids
    combined = np.zeros(shape, dtype=int)
    part_type_grid = np.full(shape, "", dtype="<U20")

    # Process each mesh
    for i, mesh in enumerate(meshes):
        print(f"Processing mesh {i + 1}/{len(meshes)}")
        try:
            # Voxelize the mesh
            voxelized = trimesh.voxel.creation.voxelize(
                mesh,
                pitch=voxel_size,
                method="subdivide",  # Faster than 'subdivide' for complex meshes
            )

            # Skip if voxelization failed
            if voxelized is None:
                continue

            # Get points and convert to grid indices
            points = voxelized.points
            indices = np.floor((points - origin) / voxel_size).astype(int)

            # Filter out-of-bounds indices
            valid = np.all((indices >= 0) & (indices < shape), axis=1)
            if not np.any(valid):
                continue

            indices = indices[valid]

            # Get part metadata
            part_type = mesh.metadata["part_type"]
            priority = mesh.metadata["priority"] if use_part_priority else 0

            # Create a mask for this part's voxels
            part_mask = np.zeros(shape, dtype=bool)
            part_mask[tuple(indices.T)] = True

            if use_part_priority:
                # Create a mask where this part has higher priority
                priority_mask = np.ones(shape, dtype=bool)
                # Only update where part exists and has higher priority
                update_mask = part_mask

                # Check against existing priorities
                if i > 0:  # Skip for first part
                    existing_priorities = np.vectorize(
                        lambda x: _get_part_priority(x if x != "" else "default")
                    )(part_type_grid)
                    update_mask = part_mask & (priority >= existing_priorities)

                # Update the grids
                combined[update_mask] = i + 1
                part_type_grid[update_mask] = part_type
            else:
                # No priority handling, just update all voxels
                combined[part_mask] = i + 1
                part_type_grid[part_mask] = part_type

        except Exception as e:
            print(f"Error processing {mesh.metadata.get('name', 'unknown')}: {str(e)}")
            continue

    # Replace empty strings with 'default' for unoccupied voxels
    part_type_grid[part_type_grid == ""] = "default"

    # Prepare metadata
    metadata = {
        "voxel_size": voxel_size,
        "origin": origin,
        "shape": shape,
        "part_types": {i + 1: m.metadata["part_type"] for i, m in enumerate(meshes)},
        "part_names": {i + 1: m.metadata["name"] for i, m in enumerate(meshes)},
        "part_priorities": {
            i + 1: m.metadata["priority"] for i, m in enumerate(meshes)
        },
        "colors": {
            part_type: color
            for part_type, color in PART_TYPE_COLORS.items()
            if part_type in part_type_grid
        },
        "type_to_label": {
            part_type: [
                i + 1
                for i, m in enumerate(meshes)
                if m.metadata["part_type"] == part_type
            ]
            for part_type in set(m.metadata["part_type"] for m in meshes)
        },
    }

    return combined, metadata


def _extract_part_type(name: str) -> str:
    """Extract part type from object name.

    Args:
        name: Object name, expected to contain part type after '@' symbol

    Returns:
        str: The part type, or 'default' if not found
    """
    if "@" in name:
        part_type = name.split("@")[-1].lower()
        if part_type in PART_TYPE_COLORS:
            return part_type
    return "default"


def _get_part_priority(part_type: str) -> int:
    """Get the priority of a part type.

    Args:
        part_type: The part type to get priority for

    Returns:
        int: The priority of the part type (higher = more important)
    """
    return PART_TYPE_PRIORITY.get(part_type, 0)
