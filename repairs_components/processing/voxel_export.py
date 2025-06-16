import logging
import numpy as np
import trimesh

import tempfile
import os
import build123d as bd
from build123d import Compound, Part
from typing import Any
import torch
import torchsparse


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
    device: str = "cpu",
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

    # Compute bounds from all meshes
    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    mins = all_vertices.min(axis=0)
    maxs = all_vertices.max(axis=0)

    # Collect voxel coordinates and labels for sparse output
    all_sparse = []

    for mesh, part_type, key in zip(meshes, part_types, mesh_keys):
        # Get or build sparse tensor
        cache_hit = (
            cached and cache is not None and key in cache and "sparse" in cache[key]
        )
        if cache_hit:
            sparse = cache[key]["sparse"]
        else:
            # Voxelize mesh and collect coordinates
            assert (mesh.bounding_box.extents < (voxel_size * 256)).all(), (
                "Mesh is too large for voxelization."
            )
            vox = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
            assert vox is not None, "Voxelization has failed."
            # note: coords_np are a tensor of (points, 3). In one case it is 60002*3

            # Get coordinates in grid space # of this shape to scene voxel grid.
            coords_np = np.floor((vox.points - mins) / voxel_size).astype(np.int8)

            # Convert to tensor and get features
            coords = torch.from_numpy(coords_np).to(device)
            feats = torch.full(
                (coords.shape[0], 1),
                PART_TYPE_LABEL.get(part_type, 0),  # get type of the part.
                dtype=torch.uint8,
                device=device,
            )
            # note: ideally csr for storage, but whatever.
            sparse = to_sparse_coo(coords, feats, device)

            if cached and cache is not None:
                cache[key]["sparse"] = sparse
        #dev note: this must be below the loop!
        all_sparse.append(sparse)

    return (all_sparse, cache) if cached else all_sparse


def to_sparse_coo(coords, labels, device):
    # Transpose coords from [N, 3] to [3, N] for sparse_coo_tensor
    coords_t = coords.t().contiguous()
    # Squeeze labels to [N] if it's [N, 1]
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    return torch.sparse_coo_tensor(
        coords_t, labels, size=(256, 256, 256), device=device
    )


def sparse_coo_to_torchsparse(sparse_coo: torch.Tensor):
    return torchsparse.SparseTensor(
        feats=sparse_coo.values(), coords=sparse_coo.indices()
    )


def batch_sparse_coo_to_torchsparse(sparse_coos: list[torch.Tensor]):
    coo = batch_coo_tensors(sparse_coos)
    return torchsparse.SparseTensor(feats=coo.values(), coords=coo.indices())


def batch_coo_tensors(sparse_coos: list[torch.Tensor]):
    assert all(coo.is_sparse for coo in sparse_coos), "All tensors must be sparse."
    # create indices with bdim.
    batched_indices = []
    for i, coo in enumerate(sparse_coos):
        batch_dim = torch.full(
            (1, coo.indices().shape[1]), i, dtype=torch.uint8, device=coo.device
        )
        indices_with_batch = torch.cat([batch_dim, coo.indices()], dim=0)
        batched_indices.append(indices_with_batch)
    all_indices = torch.cat(batched_indices, dim=1)
    all_values = torch.cat([coo.values() for coo in sparse_coos])
    return torch.sparse_coo_tensor(
        all_indices, all_values, size=(len(sparse_coos),) + tuple(sparse_coos[0].shape)
    )  # merge them.


def concat_batched_sparse(sparse_coos: list[torch.Tensor]):
    # Concatenate a list of 3D batched sparse COO tensors along the batch dimension.
    assert len(sparse_coos) > 0, "Input list is empty."
    assert all(sparse_coo.indices().ndim == 3 for sparse_coo in sparse_coos), (
        "All tensors must be 3D."
    )

    all_indices = []
    all_values = []
    batch_size = 0
    for coo in sparse_coos:
        # coo shape: (B, X, Y, Z)
        idx = coo.indices()
        vals = coo.values()
        # If tensors have batch dimension, adjust batch indices
        idx = idx.clone()
        idx[0] += batch_size
        all_indices.append(idx)
        all_values.append(vals)
        batch_size += coo.shape[0]
    indices = torch.cat(all_indices, dim=0)
    values = torch.cat(all_values, dim=0)
    shape = list(sparse_coos[0].shape)
    shape[0] = batch_size
    return torch.sparse_coo_tensor(
        indices, values, size=tuple(shape), device=values.device
    )
