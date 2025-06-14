import logging
import numpy as np
import trimesh

import tempfile
import os
import build123d as bd
from build123d import Compound, Part
from typing import Any

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


def extract_part_type(label: str) -> str:
    assert "@" in label, 'Expected "@" in the name.'
    pt = label.split("@")[-1].lower()
    if pt in PART_TYPE_COLORS:
        return pt
    logging.warning(f"Unknown part type for name: {label}")
    return "default"


def export_voxel_grid(
    compound: Compound,
    voxel_size: float,
    grid_size=(256, 256, 256),
    cached=False,
    cache: dict[str, dict[str, Any]] | None = None,
):
    """
    Voxelize an iterable of build123d Parts into a 3D int8 label grid using trimesh (CPU).

    Args:
        compound: iterable of build123d Parts; each part should have .label with '@type'.
        voxel_size: size of each voxel.
        grid_size: target output grid size; result is zero-padded to this shape.
        cached: if True, use and return a cache dict for mesh and voxel data.
        cache: dict to use for caching mesh and voxel data (optional, only used if cached=True).

    Returns:
        padded: np.ndarray of shape grid_size, dtype int8; values are part type labels.
        cache (optional): dict of mesh/voxel data if cached=True.
    """
    if cached:
        assert cache is not None, "Cache must be provided if cached=True."
    parts_list = compound.children

    # flatten all compounds.
    updated_parts_list = []
    for part in parts_list:
        if isinstance(part, Compound) and not isinstance(part, Part):
            assert len(part.children) > 0, (
                "Compound passed to export voxel is empty. This can't be."
            )
            updated_parts_list.extend(
                [comp_part.move(bd.Pos(part.position)) for comp_part in part.children]
            )
        else:
            updated_parts_list.append(part)

    parts_list = updated_parts_list
    part_types = [extract_part_type(p.label) for p in parts_list]

    # --- Caching ---
    if cached:
        if cache is None:
            cache = {}
    else:
        cache = None

    def mesh_hash(part):
        # Simple key based on part label and rounded volume to avoid precision errors
        volume = round(
            part.volume, 6
        )  # without this some caches are missed due to voxel errors. e.g. box@solid_15624.999999999998 != box@solid_15625.999999999999
        return f"{part.label}_{volume}"

    meshes = []
    mesh_keys = []
    for part in parts_list:
        key = mesh_hash(part)
        mesh_keys.append(key)
        if cached and cache is not None and key in cache:
            mesh = cache[key]["mesh"]
        else:
            # Export STL and load via trimesh
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
            tmp_path = tmp_file.name
            tmp_file.close()
            bd.export_stl(part, tmp_path)
            mesh = trimesh.load(tmp_path)
            os.remove(tmp_path)
            if cached and cache is not None:
                cache[key] = {"mesh": mesh}
        meshes.append(mesh)

    if not meshes:
        padded = np.zeros(grid_size, dtype=np.int8)
        return (padded, cache) if cached else padded

    # Compute bounds from all meshes
    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    mins = all_vertices.min(axis=0)
    maxs = all_vertices.max(axis=0)
    grid_dims = np.ceil((maxs - mins) / voxel_size).astype(int)

    combined = np.zeros(grid_dims, dtype=np.int8)
    # print("Exporting voxel grid (CPU, trimesh)...")

    for mesh, part_type, key in zip(meshes, part_types, mesh_keys):
        # Voxel cache: reuse if available
        if cached and cache is not None and key in cache and "voxel" in cache[key]:
            pts = cache[key]["voxel"]
        else:
            # Voxelize mesh using trimesh
            assert (mesh.bounding_box.extents < (voxel_size * 256)).all(), (
                "Mesh is too large for voxelization."
            )
            vox_obj = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
            if vox_obj is None:
                continue
            pts = vox_obj.points
            if cached and cache is not None:
                cache[key]["voxel"] = pts
        # Compute voxel indices from points
        idx = np.floor((pts - mins) / voxel_size).astype(int)
        # Mask for bounds
        mask = np.all((idx >= 0) & (idx < grid_dims), axis=1)
        idx = idx[mask]
        label = PART_TYPE_LABEL.get(part_type, 0)
        combined[idx[:, 0], idx[:, 1], idx[:, 2]] = label
    # print("Exporting voxel grid... done")
    padded = np.zeros(grid_size, dtype=np.int8)
    x_dim, y_dim, z_dim = grid_dims
    padded[:x_dim, :y_dim, :z_dim] = combined[
        : grid_size[0], : grid_size[1], : grid_size[2]
    ]
    return padded, cache  # if cached=False, return cache=None
