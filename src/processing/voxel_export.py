import numpy as np
import trimesh
import tempfile
import os
import build123d as bd

# Define part type to color mapping (for priority extraction)
PART_TYPE_COLORS = {
    "connector": [1, 1, 0, 0.8],
    "solid": [0.5, 0.5, 0.5, 0.8],
    "fluid": [0, 0, 1, 0.5],
    "LED": [0, 1, 0, 0.8],
    "default": [1, 0, 0, 0.8],
}

# Map part types to integer labels
PART_TYPE_LABEL = {
    "connector": 1,
    "solid": 2,
    "fluid": 3,
    "LED": 4,
    "default": 0,
}


def extract_part_type(name: str) -> str:
    if "@" in name:
        pt = name.split("@")[-1].lower()
        if pt in PART_TYPE_COLORS:
            return pt
    return "default"


def export_voxel_grid(parts, voxel_size: float, grid_size=(256, 256, 256)):
    """
    Voxelize an iterable of build123d Parts into a 3D int8 label grid.

    Args:
        parts: iterable of build123d Parts; each part should have .label with '@type'.
        voxel_size: size of each voxel.
        grid_size: target output grid size; result is zero-padded to this shape.

    Returns:
        padded: np.ndarray of shape grid_size, dtype int8; values are part type labels.
        label_to_part_type: Dict[int, str] mapping integer labels back to part types.
    """
    parts_list = list(parts)
    part_types = [extract_part_type(getattr(p, "label", "default")) for p in parts_list]
    meshes = []
    for part in parts_list:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        tmp_path = tmp_file.name
        tmp_file.close()
        bd.export_stl(part, tmp_path)
        mesh = trimesh.load(tmp_path)
        os.remove(tmp_path)
        meshes.append(mesh)

    if not meshes:
        padded = np.zeros(grid_size, dtype=np.int8)
        return padded, {v: k for k, v in PART_TYPE_LABEL.items()}

    bounds = np.stack([m.bounds for m in meshes])  # shape (n, 2, 3)
    mins = bounds[:, 0, :].min(axis=0)
    maxs = bounds[:, 1, :].max(axis=0)
    grid_dims = np.ceil((maxs - mins) / voxel_size).astype(int)

    combined = np.zeros(grid_dims, dtype=np.int8)
    label_to_part_type = {v: k for k, v in PART_TYPE_LABEL.items()}

    for mesh, part_type in zip(meshes, part_types):
        vox = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
        if vox is None:
            continue
        pts = vox.points
        idx = np.floor((pts - mins) / voxel_size).astype(int)
        mask = np.all((idx >= 0) & (idx < grid_dims), axis=1)
        idx = idx[mask]
        label = PART_TYPE_LABEL.get(part_type, 0)
        combined[idx[:, 0], idx[:, 1], idx[:, 2]] = label

    padded = np.zeros(grid_size, dtype=np.int8)
    x_dim, y_dim, z_dim = grid_dims
    padded[:x_dim, :y_dim, :z_dim] = combined[
        : grid_size[0], : grid_size[1], : grid_size[2]
    ]
    return padded, label_to_part_type
