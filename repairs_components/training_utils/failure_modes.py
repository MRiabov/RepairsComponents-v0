import torch
from genesis.engine.entities import RigidEntity


def out_of_bounds(
    min: torch.Tensor, max: torch.Tensor, gs_entities: dict[str, RigidEntity]
):
    """Check that any genesis entity is out of bounds"""
    gs_entities_no_control = gs_entities.copy()
    del gs_entities_no_control["franka@control"]
    del gs_entities_no_control["screwdriver@control"]
    del gs_entities_no_control["screwdriver_grip@tool_grip"]
    aabb = torch.stack(
        [entity.get_AABB() for entity in gs_entities_no_control.values()], dim=1
    )  # ^ batch_shape, num_entities, 2, 3
    return ((aabb[:, :, 0] < min) | (aabb[:, :, 1] > max)).any(dim=-1).any(dim=-1)
