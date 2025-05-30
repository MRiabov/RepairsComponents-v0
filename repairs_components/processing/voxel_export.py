import numpy as np
import trimesh
import tempfile
import os
import build123d as bd

# Define part type to color mapping (for priority extraction)
PART_TYPE_COLORS = {
    "connector": [1, 1, 0, 0.8],  # yellow
    "LED": [0, 1, 0, 0.8],  # green
    "switch": [1, 0.25, 0.25, 0.8],
    "fastener_markup": [0.8, 0, 0.8, 0.8],  # magenta-like.
    "button": [1, 0.2, 0.2, 0.8],
    "solid": [0.5, 0.5, 0.5, 0.8],  # grey
    "fastener": [0.58, 0.44, 0.86, 0.8],  # medium purple
    "fluid": [0, 0, 1, 0.5],  # blue
    "default": [1, 0, 0, 0.8],  # red
}

# Map part types to integer labels
PART_TYPE_LABEL = {
    "connector": 1,
    "LED": 2,
    "fastener": 3,
    "fastener_markup": 4,  # should be below fastener.
    "switch": 5,
    "button": 6,
    "solid": 7,
    "fluid": 8,
    "default": 0,
}
LABEL_TO_PART_TYPE = {v: k for k, v in PART_TYPE_LABEL.items()}


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
    """
    parts_list = list(parts)

    # flatten all compounds.
    updated_parts_list = []
    for part in parts_list:
        if isinstance(part, bd.Compound) and not isinstance(part, bd.Part):
            assert len(part.children) > 0, (
                "Compound passed to export voxel is empty. This can't be."
            )
            updated_parts_list.extend(
                [comp_part.move(bd.Pos(part.position)) for comp_part in part.children]
            )
        else:
            updated_parts_list.append(part)

    parts_list = updated_parts_list
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
    return padded
