import torch
from genesis.engine.entities import RigidEntity

from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.sim_state_global import PhysicalStateInfo


def get_links_idx_from_names(names: list[str], gs_entities: dict[str, RigidEntity]):
    return torch.tensor(
        [gs_entities[name].base_link.idx for name in names], dtype=torch.int32
    )


# hmmm, this should be cached.
def get_links_idx_from_fastener_ids(
    fastener_ids: torch.Tensor, gs_entities: dict[str, RigidEntity]
):
    "Util to get link indices from fastener ids."
    # Minimize lookups into gs_entities by querying only for unique fasteners,
    # then reconstruct the full index list via inverse indices.
    unique_ids, inverse = torch.unique(fastener_ids, return_inverse=True)
    inverse_cpu = inverse.cpu()

    unique_names = [
        Fastener.fastener_name_in_simulation(fid.item()) for fid in unique_ids
    ]
    unique_link_idxs = torch.tensor(
        [gs_entities[name].base_link.idx for name in unique_names],
        dtype=torch.int32,
    )
    return unique_link_idxs[inverse_cpu].to(fastener_ids.device)


def populate_base_link_indices(
    physical_state_info: PhysicalStateInfo,
    gs_entities: dict[str, RigidEntity],
    num_fasteners: int,
):
    """Populate base link index tensors on a PhysicalStateInfo-like object.

    Sets:
    - physical_state_info.body_base_link_idx: torch.int32 [num_bodies]
    - physical_state_info.fastener_base_link_idx: torch.int32 [num_fasteners]

    Args:
        physical_state_info: Object with fields body_indices, body_base_link_idx,
            and fastener_base_link_idx.
        gs_entities: Dict mapping body/fastener names to Genesis RigidEntity.
        num_fasteners: Number of registered fasteners (integer or scalar tensor).
    """
    # Bodies
    num_bodies = len(physical_state_info.body_indices)
    body_base = torch.empty((num_bodies,), dtype=torch.int32)
    for name, idx in physical_state_info.body_indices.items():
        body_base[idx] = gs_entities[name].base_link.idx
    physical_state_info.body_base_link_idx = body_base

    # Fasteners
    assert isinstance(num_fasteners, int)
    n_fast = num_fasteners
    if n_fast > 0:
        ids = torch.arange(n_fast, dtype=torch.long)
        physical_state_info.fastener_base_link_idx = get_links_idx_from_fastener_ids(
            ids, gs_entities
        ).to(torch.int32)


def is_weld_constraint_present(scene, rigid_body_a, rigid_body_b, env_idx=0):
    """
    Check whether a weld exists between two links/entities in the given scene/environment.

    Accepts RigidEntity or RigidLink objects for convenience.
    Uses scene.sim.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True).
    Handles both dict outputs (with 'link_a'/'link_b' or 'obj_a'/'obj_b') and
    legacy tensor outputs of shape [N, 3] where rows are [env_id, link_a, link_b].
    """
    # FIXME: this code is extremely bloated.
    # see version below:
    # assert welds.ndim == 2 and welds.shape[-1] == 3
    # combination_a = torch.tensor(
    #     [env_id, base_link_idx_a, base_link_idx_b],
    #     device=welds.device,
    #     dtype=welds.dtype,
    # )
    # combination_b = torch.tensor(
    #     [env_id, base_link_idx_b, base_link_idx_a],
    #     device=welds.device,
    #     dtype=welds.dtype,
    # )
    # return torch.any(torch.all(welds == combination_a, dim=-1)) or torch.any(
    #     torch.all(welds == combination_b, dim=-1)
    # )
    # similarly, find nonzero of rows where a and b are equal to a_idx and b_idx, and find if env_idx is in them. or use bool masks, anyway.

    def _extract_link_idx(obj):
        # RigidEntity has base_link.idx, RigidLink has idx
        if hasattr(obj, "base_link") and hasattr(obj.base_link, "idx"):
            return obj.base_link.idx
        if hasattr(obj, "idx"):
            return obj.idx
        raise AssertionError("Object must be a RigidEntity or RigidLink with an idx")

    a_idx = _extract_link_idx(rigid_body_a)
    b_idx = _extract_link_idx(rigid_body_b)

    welds = scene.sim.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)

    # Dict output: expected keys 'link_a'/'link_b' or 'obj_a'/'obj_b'
    if isinstance(welds, dict):

        def _to_tensor(val):
            # Values may be tuples/lists of tensors; take the first element
            if isinstance(val, (tuple, list)):
                val = val[0]
            return val

        if "link_a" in welds and "link_b" in welds:
            link_a = _to_tensor(welds["link_a"])  # [*]
            link_b = _to_tensor(welds["link_b"])  # [*]
        elif "obj_a" in welds and "obj_b" in welds:
            link_a = _to_tensor(welds["obj_a"])  # [*]
            link_b = _to_tensor(welds["obj_b"])  # [*]
        else:
            raise AssertionError("Unsupported weld constraints dict format")

        # Normalize to 1D
        if hasattr(link_a, "reshape"):
            link_a = link_a.reshape(-1)
        if hasattr(link_b, "reshape"):
            link_b = link_b.reshape(-1)

        assert link_a.shape[0] == link_b.shape[0]

        # Optional env filtering if available
        if "env" in welds:
            env = _to_tensor(welds["env"])
            if hasattr(env, "reshape"):
                env = env.reshape(-1)
            mask = env == int(env_idx)
            if mask.ndim != 0:
                link_a = link_a[mask]
                link_b = link_b[mask]

        present = torch.any((link_a == a_idx) & (link_b == b_idx)) or torch.any(
            (link_a == b_idx) & (link_b == a_idx)
        )
        return bool(present)

    # Tensor output: [N, 3] with [env, link_a, link_b]
    assert torch.is_tensor(welds), "Unexpected weld constraints format"
    assert welds.ndim == 2 and welds.shape[-1] == 3, (
        "Expected weld tensor of shape [N, 3]"
    )
    device = welds.device
    dtype = welds.dtype
    env_val = int(env_idx) if isinstance(env_idx, int) else int(env_idx.item())
    row_ab = torch.tensor([env_val, a_idx, b_idx], device=device, dtype=dtype)
    row_ba = torch.tensor([env_val, b_idx, a_idx], device=device, dtype=dtype)
    present = torch.any(torch.all(welds == row_ab, dim=-1)) or torch.any(
        torch.all(welds == row_ba, dim=-1)
    )
    return bool(present)
