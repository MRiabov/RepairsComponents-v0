import torch
from genesis.engine.entities import RigidEntity


def out_of_bounds(
    min: torch.Tensor,
    max: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    control_entities_min: torch.Tensor | None = torch.tensor([-10, -10, -0.5]),
    control_entities_max: torch.Tensor | None = torch.tensor([10, 10, 10]),
):
    """Check that any genesis entity is out of bounds. Control entites can be further out."""
    gs_entities_no_control = gs_entities.copy()
    control_entities = ["franka@control", "screwdriver@control"]
    # "screwdriver_grip@tool_grip",

    for entity in control_entities:
        del gs_entities_no_control[entity]
    parts_aabb = torch.stack(
        [entity.get_AABB() for entity in gs_entities_no_control.values()], dim=1
    )  # ^ batch_shape, num_entities, 2, 3

    # allow control entities to be further out
    control_aabb = torch.stack(
        [  # gs_entities["screwdriver_grip@tool_grip"].get_AABB(),
            gs_entities["franka@control"].get_AABB(),
            gs_entities["screwdriver@control"].get_AABB(),
        ],
        dim=1,
    )  # ^ batch_shape, num_entities, 2, 3

    parts_out_of_bounds = (
        (parts_aabb[:, :, 0] < min) | (parts_aabb[:, :, 1] > max)
    ).any(dim=(-1, -2))
    control_out_of_bounds = (
        (control_aabb[:, :, 0] < control_entities_min)
        | (control_aabb[:, :, 1] > control_entities_max)
    ).any(dim=(-1, -2))
    any_out_of_bounds = parts_out_of_bounds | control_out_of_bounds
    if any_out_of_bounds.any():
        print(f"Parts out of bounds: {parts_out_of_bounds}")
        print(f"Control out of bounds: {control_out_of_bounds}")
    return any_out_of_bounds
