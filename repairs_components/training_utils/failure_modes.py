import torch
from genesis.engine.entities import RigidEntity


def out_of_bounds(
    min: torch.Tensor, max: torch.Tensor, gs_entities: dict[str, RigidEntity]
):
    """Check that any genesis entity is out of bounds"""
    for entity in gs_entities.values():
        aabb = entity.get_AABB()  # batch_shape, 2, 3
        if (aabb[:, 0] < min).any() or (aabb[:, 1] > max).any():
            return True
    return False
