"""
The purpose of testing_file.py is to export build123d parts to trimesh, then voxelize them there and display using matplotlib.
I want the logic to be as follows:
connector, solid, fluid, LED, and others are own type of labels. So if the part (the mesh) ends with "@connector" it should be assigned a yellow color, and if a solid, a grey color, and etc.

Most importantly, I should be able to get this logic from the voxelized array, not only displayed in matplotlib.

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

# Prepare voxel grids and labels
voxel_grids = {}
labels = {}

# Compute the overall bounding box for all meshes
all_bounds = [mesh.bounds for mesh in trimesh_scene.geometry.values()]
mins = np.min([b[0] for b in all_bounds], axis=0)
maxs = np.max([b[1] for b in all_bounds], axis=0)

# Compute the shape of the common grid
bbox_size = maxs - mins
shape = np.ceil(bbox_size / voxel_size).astype(int)
origin = mins

# Assign unique integer labels and voxelize into the common grid
voxel_grids = {}
labels = {}
for i, (name, mesh) in enumerate(trimesh_scene.geometry.items()):
    voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
    # voxelized.points are the center coordinates of occupied voxels
    voxel_grids[name] = voxelized.points
    labels[name] = i + 1


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


# Initialize the combined grid and part type grid
combined = np.zeros(shape, dtype=int)
part_type_grid = np.full(shape, "default", dtype="<U20")  # Initialize with default type

# Create mappings
label_to_part_type = create_part_type_mapping(labels, trimesh_scene)

# Fill the grids
for name, points in voxel_grids.items():
    indices = np.floor((points - origin) / voxel_size).astype(int)
    valid = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = indices[valid]
    label = labels[name]
    combined[indices[:, 0], indices[:, 1], indices[:, 2]] = label
    part_type = label_to_part_type[label]
    part_type_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = part_type

print(f"Combined labeled voxel grid shape: {combined.shape}")
print(f"Unique labels in grid: {np.unique(combined)}")

# Create color mappings
label_colors = create_color_mapping(label_to_part_type)
label_colors[0] = [0, 0, 0, 0]  # background (transparent)
max_label = max(label_colors.keys()) if label_colors else 0
label_colors_list = [label_colors.get(i, [1, 1, 1, 0.8]) for i in range(max_label + 1)]


print("Part types in the scene:")
for name, part_type in label_to_part_type.items():
    print(f"  Label {name}: {part_type}")

# Create a 3D figure with a dark background for better contrast
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")


# Set axis colors to white for better visibility
ax.xaxis.pane.set_color("black")
ax.yaxis.pane.set_color("black")
ax.zaxis.pane.set_color("black")
ax.xaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 0.3)})
ax.yaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 0.3)})
ax.zaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 0.3)})


# Plot each label with a different color
for label in np.unique(combined):
    if label == 0:
        continue  # Skip background (transparent)

    # Get the indices for this label
    idx = np.argwhere(combined == label)
    if len(idx) == 0:
        continue

    # Get color for this label from our mapping
    color = label_colors.get(
        label, [1, 1, 1, 0.8]
    )  # default to white if label not found

    # Create a voxel grid for this label
    mask = np.zeros(combined.shape, dtype=bool)
    mask[tuple(idx.T)] = True

    # Plot the voxels
    ax.voxels(mask, facecolors=color, edgecolor="black", linewidth=0.3, alpha=0.8)

# Set labels and title with white text
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_title("Voxelized Parts", color="white")

# Set the aspect ratio to be equal
ax.set_box_aspect(combined.shape)

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)

# Save the figure with a transparent background
output_path = "voxel_visualization.png"
print("saved to ", output_path)
plt.tight_layout()
plt.savefig(
    output_path, dpi=150, bbox_inches="tight", transparent=True, facecolor="black"
)
plt.close()

print(f"Saved visualization to {output_path}")

# trimesh_mesh.export("test.obj")

# open3d_mesh = o3d_io.read_triangle_mesh("test.obj")


# # Voxelize the open3d mesh
# voxel_size = 2.0
# voxel_grid = o3d_geometry.VoxelGrid.create_from_triangle_mesh(
#     open3d_mesh, voxel_size=voxel_size
# )

# # Visualize the voxel grid with open3d
# # open3d.visualization.draw([voxel_grid])

# open3d.visualization.ren
# o3d_vis.create_window(visible=False)
# o3d_vis.add_geometry(voxel_grid)
# o3d_vis.poll_events()
# o3d_vis.update_renderer()

# # Save to image
# o3d_vis.capture_screen_image("output_voxel_grid.png")
# o3d_vis.destroy_window()


# ocp_vscode.show([test, test2])
