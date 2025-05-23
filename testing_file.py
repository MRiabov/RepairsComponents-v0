import build123d as bd
import ocp_vscode
import trimesh
import numpy as np
import open3d
import open3d.visualization as o3d_vis
from open3d import geometry as o3d_geometry, io as o3d_io


with bd.BuildPart() as test:
    bd.Box(10, 10, 10)

with bd.BuildPart() as test2:
    with bd.Locations((15, 0, 0)):
        bd.Box(10, 10, 10)

test = test.solid()
test2 = test2.solid()
test.label = "1@connector"
test2.label = "2@connector"
test.color = bd.Color("blue")
test2.color = bd.Color("red")

test_comp = bd.Compound((test, test2))

# bd.export_stl(test_comp, "test.stl")
mesher = bd.Mesher()
mesher.add_shape((test, test2))
mesher.write("test.3mf")

trimesh_scene = trimesh.load("test.3mf")
print(f"Loaded type: {type(trimesh_scene)}")

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

# Combine into a single labeled grid
combined = np.zeros(shape, dtype=int)

for name, points in voxel_grids.items():
    # Convert world coordinates to grid indices
    indices = np.floor((points - origin) / voxel_size).astype(int)
    # Filter indices within bounds
    valid = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = indices[valid]
    combined[indices[:, 0], indices[:, 1], indices[:, 2]] = labels[name]

print(f"Combined labeled voxel grid shape: {combined.shape}")
print(f"Unique labels in grid: {np.unique(combined)}")

# Visualize the labeled voxel grid using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Colormap for up to 10 labels (expand as needed)
label_colors = [
    [0, 0, 0, 0],  # background (transparent)
    [1, 0, 0, 0.8],  # label 1: red
    [0, 1, 0, 0.8],  # label 2: green
    [0, 0, 1, 0.8],  # label 3: blue
    [1, 1, 0, 0.8],  # label 4: yellow
    [1, 0, 1, 0.8],  # label 5: magenta
    [0, 1, 1, 0.8],  # label 6: cyan
    [1, 0.5, 0, 0.8],  # label 7: orange
    [0.5, 0, 0.5, 0.8],  # label 8: purple
    [0, 0.5, 0, 0.8],  # label 9: dark green
]

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

    # Get color for this label (with alpha=0.8)
    color = label_colors[label] if label < len(label_colors) else [1, 1, 1, 0.8]

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
