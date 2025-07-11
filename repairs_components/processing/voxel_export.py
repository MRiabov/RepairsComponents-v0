import logging
import numpy as np
import trimesh

import tempfile
import build123d as bd
from build123d import Compound, Part
from typing import Any
import torch
from pathlib import Path
import torch_scatter
from torch_scatter import scatter_min


# Define part type to color mapping (for priority extraction)
PART_TYPE_COLORS = {
    "part_to_replace": [1, 0.5, 0, 0.8],  # orange
    "connector_def": [1, 1, 0, 0.8],  # yellow
    "connector": [0.9, 0.9, 0, 0.8],  # yellow, but a little greyer.
    "LED": [0, 1, 0, 0.8],  # green
    "switch": [1, 0.25, 0.25, 0.8],
    "fastener_markup": [0.8, 0, 0.8, 0.8],  # magenta-like.
    "button": [1, 0.2, 0.2, 0.8],
    "solid": [0.5, 0.5, 0.5, 0.8],  # grey
    "fixed_solid": [0.4, 0.4, 0.4, 0.8],  # darker grey
    "fastener": [0.58, 0.44, 0.86, 0.8],  # medium purple
    "fluid": [0, 0, 1, 0.5],  # blue
    "default": [1, 0, 0, 0.8],  # red
}

# Map part types to integer labels
PART_TYPE_LABEL = {
    "part_to_replace": 1,  # this is the part that will be replaced in "ReplaceTask".
    "connector_def": 2,
    "connector": 3,
    "LED": 4,
    "fastener": 5,
    "fastener_markup": 6,  # should be below fastener.
    "switch": 7,
    "button": 8,
    "solid": 9,
    "fixed_solid": 10,
    "fluid": 11,
    "default": 0,
}
LABEL_TO_PART_TYPE = {v: k for k, v in PART_TYPE_LABEL.items()}


def extract_part_type(label: str) -> str:
    assert label is not None, "Label is None."
    assert "@" in label, f"Expected '@' in the name. Got: {label}"
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
    save: bool = False,
    scene_id: int | None = None,
    save_path: Path | None = None,
    scene_file_name: str | None = None,
    device: str = "cpu",
):
    """
    Voxelize an iterable of build123d Parts into a 3D int8 label grid using trimesh (CPU).

    Args:
        compound: iterable of build123d Parts; each part should have .label with '@type'.
        voxel_size: size of each voxel.
        grid_size: target output grid size; result is zero-padded to this shape.
        cached: if True, use and return a cache dict for mesh and voxel data.
        scene_id: id of the scene. If None, voxel grid will not be saved.
        cache: dict to use for caching mesh and voxel data (optional, only used if cached=True).

    Returns:
        padded: np.ndarray of shape grid_size, dtype int8; values are part type labels.
        cache (optional): dict of mesh/voxel data if cached=True.
    """
    if cached:
        assert cache is not None, "Cache must be provided if cached=True."
    if save:
        assert save_path is not None, "Save path must be provided if save=True."
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
            if save:
                mesh_save_path = save_path / f"{key}.stl"
                bd.export_stl(part, mesh_save_path)
                mesh = trimesh.load_mesh(mesh_save_path)
            else:
                tmp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".stl")
                tmp_path = tmp_file.name
                bd.export_stl(part, tmp_path)
                mesh = trimesh.load_mesh(tmp_path)
                tmp_file.close()
            # os.remove(tmp_path)
            if cached and cache is not None:
                cache[key] = {"mesh": mesh}
        meshes.append(mesh)

    # Compute bounds from all meshes
    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    mins = all_vertices.min(axis=0)
    maxs = all_vertices.max(axis=0)

    # Collect voxel coordinates and labels for sparse output
    all_sparse = []

    # mesh all parts in a single scene
    for mesh, part_type, key in zip(meshes, part_types, mesh_keys):
        # Get or build sparse tensor
        # --- Retrieve or build LOCAL (re-centered) sparse voxel grid ---
        cache_hit = (
            cached and cache is not None and key in cache and "sparse" in cache[key]
        )
        if cache_hit:
            sparse_local = cache[key]["sparse"]
        else:
            # Voxelize mesh and collect coordinates
            assert (mesh.bounding_box.extents < (voxel_size * 256)).all(), (
                "Mesh is too large for voxelization."
            )
            vox = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
            assert vox is not None, "Voxelization has failed."

            mesh_min = mesh.bounds[0]  # part origin in world units
            # Local voxel coordinates (starting at 0) – makes cached representation reusable
            coords_np_local = np.round(
                (vox.points - mesh_min + 1e-5) / voxel_size
            ).astype(np.uint8)
            assert coords_np_local.min() >= 0, (
                f"Some voxel points are negative (local coords). Got: {coords_np_local.min()}"
            )
            assert coords_np_local.max() < 256, (
                f"Local voxel coords exceed grid. Got: {coords_np_local.max()}"
            )

            coords_local = torch.from_numpy(coords_np_local).to(device)
            feats = torch.full(
                (coords_local.shape[0], 1),
                PART_TYPE_LABEL.get(part_type, 0),
                dtype=torch.uint8,
                device=device,
            )
            sparse_local = to_sparse_coo(coords_local, feats, device).coalesce()

            # Store LOCAL sparse grid in cache for future re-use
            if cached and cache is not None:
                cache[key]["sparse"] = sparse_local

        # --- Shift local sparse grid to SCENE coordinates ---
        mesh_min = mesh.bounds[0]
        offset_vox = np.round((mesh_min - mins) / voxel_size).astype(np.int16)  # (3,)
        if (offset_vox != 0).any():  # if any of the voxels are not at 0, shift them
            offset_t = torch.tensor(
                offset_vox, device=device, dtype=torch.long
            ).unsqueeze(1)  # [3,1]
            shifted_indices = sparse_local.indices() + offset_t
            # Clamp any indices that accidentally became 256 (rounding edge-case) back to 255
            if (shifted_indices >= 256).any():
                shifted_indices = torch.clamp(shifted_indices, max=255)
            sparse = torch.sparse_coo_tensor(
                shifted_indices,
                sparse_local.values(),
                size=(256, 256, 256),
                device=device,
            ).coalesce()
        else:
            sparse = sparse_local

        all_sparse.append(sparse)

    # Merge all sparse tensors into a single scene tensor
    sparse_scene_stack_values = torch.cat([s.values() for s in all_sparse], dim=0)
    sparse_scene_stack_indices = torch.cat([s.indices() for s in all_sparse], dim=1)

    # Transpose indices so that each row is a voxel coordinate [x, y, z]
    indices_t = sparse_scene_stack_indices.t().contiguous()  # [N, 3]

    # Find unique voxel coordinates and an inverse mapping back to each stacked voxel
    unique_indices_t, inverse = torch.unique(indices_t, dim=0, return_inverse=True)

    # For overlapping voxels keep the minimum label (lower label = higher priority)
    values, _ = scatter_min(
        sparse_scene_stack_values,
        inverse,
        dim=0,
        dim_size=unique_indices_t.size(0),
    )

    # Convert indices back to the [3, N] format expected by sparse_coo_tensor
    unique_indices = unique_indices_t.t().contiguous()

    sparse_scene = torch.sparse_coo_tensor(
        unique_indices,
        values,
        size=(256, 256, 256),
        device=device,
    ).coalesce()

    return (sparse_scene, cache) if cached else sparse_scene


def to_sparse_coo(coords, labels, device):
    # Transpose coords from [N, 3] to [3, N] for sparse_coo_tensor
    coords_t = coords.t().contiguous()
    # Squeeze labels to [N] if it's [N, 1]
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    return torch.sparse_coo_tensor(
        coords_t, labels, size=(256, 256, 256), device=device
    )


# sparse arrays utils:
def sparse_arr_remove(
    sparse_tensor: torch.Tensor, remove_idx: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """Remove all values at the specified dimension.

    Args:
        remove_idx: The index to remove values from
        dim: The dimension to remove values from

    Note:
        This only removes the values at the specified dimension but doesn't
        reduce the size of the buffer. The buffer size remains the same, but the
        removed positions become available for new additions.
    """
    # Ensure the batch dimension is within bounds
    assert ((0 <= remove_idx) & (remove_idx < sparse_tensor.size(dim))).all(), (
        f"Tried to remove at idx {remove_idx} is out of bounds [0, {sparse_tensor.size(dim)})"
    )

    # Remove all values at the specified batch dimension
    sparse_tensor = sparse_tensor.coalesce()
    dim_indices = sparse_tensor.indices()[dim]
    bool_mask = torch.isin(dim_indices, remove_idx, invert=True)
    # ^ are remove_idx in dim_indices.

    return torch.sparse_coo_tensor(
        sparse_tensor.indices()[:, bool_mask],
        sparse_tensor.values()[bool_mask],
        size=sparse_tensor.size(),
        device=sparse_tensor.device,
    )


def sparse_arr_put(
    dest_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Put values from src_tensor into dest_tensor at the specified dimension using the provided indices.

    Args:
        dest_tensor: The destination sparse tensor
        src_tensor: The source sparse tensor to copy values from
        index: 1D tensor of indices where each element in src_tensor should be placed along the specified dimension
        dim: The dimension along which to place the values

    Returns:
        A new sparse tensor with values from src_tensor placed at the specified positions
    """
    src_tensor = src_tensor.coalesce()
    dest_tensor = dest_tensor.coalesce()
    assert src_tensor.is_sparse, "Source tensor must be a sparse tensor"
    assert dest_tensor.is_sparse, "Destination tensor must be a sparse tensor"
    assert index.dim() == 1, "Index must be a 1D tensor"
    assert index.shape[0] == src_tensor.shape[0], (
        "Index must have the same length as the number of non-zero elements in src_tensor"
    )

    # Ensure the tensors are on the same device
    if src_tensor.device != dest_tensor.device:
        src_tensor = src_tensor.to(dest_tensor.device)
    if index.device != dest_tensor.device:
        index = index.to(dest_tensor.device)

    # Coalesce both tensors
    dest_tensor = dest_tensor.coalesce()
    src_tensor = src_tensor.coalesce()

    # Get the indices and values
    dest_indices = dest_tensor.indices()
    dest_values = dest_tensor.values()
    src_indices = src_tensor.indices()
    src_values = src_tensor.values()

    # Create new indices for the source tensor with the specified dim remapped
    new_indices = src_indices.clone()

    # # Remap the dimension of interest using a loop
    # unique_indices = torch.unique(src_indices[dim]) # note: doable without loop, but whatever.
    # for i in range(len(unique_indices)):
    #     src_idx = unique_indices[i]
    #     mask = src_indices[dim] == src_idx
    #     new_indices[dim, mask] = index[i]

    # Remap the dimension of interest using unique and inverse mapping
    _, inverse_indices = torch.unique(src_indices[dim], return_inverse=True)
    new_indices[dim] = index[inverse_indices]  # just a little optimization.

    # Combine the indices and values
    combined_indices = torch.cat([dest_indices, new_indices], dim=1)
    combined_values = torch.cat([dest_values, src_values])

    # Create the new sparse tensor
    return torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        size=dest_tensor.size(),
        device=dest_tensor.device,
    ).coalesce()
