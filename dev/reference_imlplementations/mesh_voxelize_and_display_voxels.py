"""
The purpose of testing_file.py is to export build123d parts to trimesh, then voxelize them there and display using matplotlib.
I want the logic to be as follows:
connector, solid, fluid, LED, and others are own type of labels. So if the part (the mesh) ends with "@connector" it should be assigned a yellow color, and if a solid, a grey color, and etc.

Most importantly, I should be able to get this logic from the voxelized array, not only displayed in matplotlib.
Additionally, certain parts are more priority to be displayed over others. For example, a connector should always take precedence over a solid part.

"""
# TODO:  What will happen in this code if two entities overlap? In my case, a connector willl overlap with a part. I want the connector part to always take precedence over all parts, and also it would be cool to have this order in general. How is it currently determined?

import build123d as bd
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List, Union
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection

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


def extract_part_type(name: str) -> str:
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


def get_part_priority(part_type: str) -> int:
    """Get the priority of a part type.

    Args:
        part_type: The part type to get priority for

    Returns:
        int: The priority of the part type (higher = more important)
    """
    return PART_TYPE_PRIORITY.get(part_type, 0)


with bd.BuildPart() as test:
    bd.Box(10, 10, 10)

with bd.BuildPart() as test2:
    with bd.Locations((15, 0, 0)):
        bd.Box(10, 10, 10)

test = test.solid()
test2 = test2.solid()
test.label = "1@connector"
test2.label = "2@solid"
test.color = bd.Color("blue")
test2.color = bd.Color("red")

test_comp = bd.Compound((test, test2))

# bd.export_stl(test_comp, "test.stl")
mesher = bd.Mesher()
mesher.add_shape((test, test2))
mesher.write("test.3mf")

trimesh_scene = trimesh.load("test.3mf")
print(f"Loaded type: {type(trimesh_scene)}")

# Define part type to color mapping (RGBA format)
PART_TYPE_COLORS = {
    "connector": [1, 1, 0, 0.8],  # Yellow
    "solid": [0.5, 0.5, 0.5, 0.8],  # Grey
    "fluid": [0, 0, 1, 0.5],  # Semi-transparent blue
    "LED": [0, 1, 0, 0.8],  # Green
    "default": [1, 0, 0, 0.8],  # Red (for any other type)
}

voxel_size = 2.0

# Compute the overall bounding box for all meshes
all_bounds = [mesh.bounds for mesh in trimesh_scene.geometry.values()]
mins = np.min([b[0] for b in all_bounds], axis=0)
maxs = np.max([b[1] for b in all_bounds], axis=0)

# Compute the shape of the common grid
bbox_size = maxs - mins
shape = np.ceil(bbox_size / voxel_size).astype(int)
origin = mins

# Sort meshes by priority (lowest priority first)
sorted_meshes = sorted(
    trimesh_scene.geometry.items(),
    key=lambda x: get_part_priority(extract_part_type(x[0])),
)

# Initialize grids
combined = np.zeros(shape, dtype=int)
part_type_grid = np.full(shape, "", dtype="<U20")  # Empty string means unoccupied
labels = {}

# Process each mesh in order of increasing priority
for i, (name, mesh) in enumerate(sorted_meshes):
    # Voxelize the mesh
    voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
    if voxelized is None:
        print(f"Warning: Could not voxelize {name}")
        continue

    # Store the points and assign a label
    points = voxelized.points
    label = i + 1
    labels[name] = label

    # Get part type
    part_type = extract_part_type(name)
    priority = get_part_priority(part_type)
    print(f"Processing {name} (type: {part_type}, priority: {priority})")

    # Convert points to grid indices
    indices = np.floor((points - origin) / voxel_size).astype(int)
    valid = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = indices[valid]

    # Update the voxel grid with priority handling
    for idx in indices:
        idx_tuple = tuple(idx)
        current_priority = get_part_priority(part_type_grid[idx_tuple] or "default")

        # Only update if this part type has higher or equal priority
        if priority >= current_priority:
            combined[idx_tuple] = label
            part_type_grid[idx_tuple] = part_type

# Replace empty strings with 'default' for unoccupied voxels
part_type_grid[part_type_grid == ""] = "default"


def create_part_type_mapping(labels: Dict[str, int], scene: Any) -> Dict[int, str]:
    """Create a mapping from label to part type.

    Args:
        labels: Dictionary mapping mesh names to their labels
        scene: The trimesh scene containing the meshes

    Returns:
        Dict[int, str]: Mapping from label to part type
    """
    label_to_part_type = {}
    for name, label in labels.items():
        part_type = extract_part_type(name)
        label_to_part_type[label] = part_type
        if part_type == "default" and "@" in name:
            print(f"Warning: Unknown part type in '{name}', using default color")
    return label_to_part_type


def create_color_mapping(label_to_part_type: Dict[int, str]) -> Dict[int, List[float]]:
    """Create a mapping from label to RGBA color.

    Args:
        label_to_part_type: Dictionary mapping labels to part types

    Returns:
        Dict[int, List[float]]: Mapping from label to RGBA color
    """
    return {
        label: PART_TYPE_COLORS.get(part_type, PART_TYPE_COLORS["default"])
        for label, part_type in label_to_part_type.items()
    }


# Create mappings
label_to_part_type = create_part_type_mapping(labels, trimesh_scene)

print(f"Combined labeled voxel grid shape: {combined.shape}")
print(f"Unique labels in grid: {np.unique(combined)}")

print("\nPart types in the scene (in order of priority):")
for part_type in sorted(
    PART_TYPE_PRIORITY.keys(), key=lambda x: -PART_TYPE_PRIORITY[x]
):
    print(f"  {part_type}: priority {PART_TYPE_PRIORITY[part_type]}")

print("\nAssigned labels and their part types:")
for name, label in sorted(labels.items(), key=lambda x: x[1]):
    part_type = label_to_part_type[label]
    print(
        f"  Label {label}: {name} (type: {part_type}, priority: {get_part_priority(part_type)})"
    )

import matplotlib.pyplot as plt

# 3D scatter visualization of integer labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Extract voxel coordinates and labels
filled = combined > 0
x, y, z = np.nonzero(filled)
labels_vals = combined[x, y, z]

# Use discrete colormap for labels
cmap = plt.get_cmap('tab10', int(combined.max() + 1))
sc = ax.scatter(x, y, z, c=labels_vals, cmap=cmap, marker='s', s=6, alpha=0.8)
plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label='Label')

# Axis styling
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')
ax.set_title('Voxelized Parts (Labels)', color='white')
plt.show()
