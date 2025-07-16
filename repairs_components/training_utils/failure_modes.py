import torch
from genesis.engine.entities import RigidEntity


def out_of_bounds(
    min: torch.Tensor,
    max: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    control_entities_min: torch.Tensor | None = torch.tensor([-2, -2, -0.2]),
    control_entities_max: torch.Tensor | None = torch.tensor([2, 2, 2]),
):
    """Check that any genesis entity is out of bounds. Control entites can be further out."""
    filtered_gs_entities = gs_entities.copy()
    control_entities = ["franka@control", "screwdriver@control"]
    # fasteners can be out of bounds too when carried by screwdriver.
    fastener_entities = [k for k in gs_entities.keys() if k.endswith("fastener")]
    # NOTE: would be good to have a check that the screwdriver carries them explicitly. Doable through tool state/constraints check.
    # FIXME: ^ the above is pretty bad (fasteners can roll out and be undetected or whatnot.)
    expanded_bounds_entities = control_entities + fastener_entities

    # "screwdriver_grip@tool_grip",

    for entity in expanded_bounds_entities:
        del filtered_gs_entities[entity]
    parts_aabb = torch.stack(
        [entity.get_AABB() for entity in filtered_gs_entities.values()], dim=1
    )  # ^ batch_shape, num_entities, 2, 3

    # allow control entities to be further out
    control_aabb = torch.stack(
        [gs_entities[entity].get_AABB() for entity in expanded_bounds_entities],
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
