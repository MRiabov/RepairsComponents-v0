import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from repairs_components.geometry.connectors.connectors import check_connections
from repairs_components.geometry.fasteners import (
    activate_hand_connection,
    check_fastener_possible_insertion,
)
from repairs_components.logic.tools.screwdriver import (
    Screwdriver,
    receive_screw_in_action,
)
from repairs_components.processing.translation import (
    translate_genesis_to_python,
)
from repairs_components.training_utils.sim_state_global import RepairsSimState


def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    sim_state: RepairsSimState,
    desired_state: RepairsSimState,
):
    "Step repairs sim."
    # get values (state) from genesis
    (
        sim_state,
        picked_up_fastener_tip_position,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    ) = translate_genesis_to_python(scene, gs_entities, sim_state)

    # batch electronics attachments: compute, clear, and apply if present
    if sim_state.has_electronics:
        electronics_attachments_batch = check_connections(
            male_connector_positions, female_connector_positions
        )
        # clear all previous connections
        for es in sim_state.electronics_state:
            es.clear_connections()
        # apply batch attachments [batch_idx, male_idx, female_idx]
        for b, m, f in electronics_attachments_batch.tolist():
            sim_state.electronics_state[b].connect(m, f)
    else:
        electronics_attachments_batch = torch.empty((0, 3), dtype=torch.long)

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = sim_state.diff(desired_state)

    success = total_diff_left == 0  # Tensor[batch] bool
    B = scene.n_envs
    # Fastener insertion batch
    # mask where action screw_in and tool is Screwdriver
    screw_mask = (
        torch.tensor(
            [isinstance(ts.current_tool, Screwdriver) for ts in sim_state.tool_state],
            dtype=torch.bool,
            device=actions.device,
        )
        & receive_screw_in_action(
            actions
        )  # note:if there are more tools, create an interface, I guess.
    )
    if screw_mask.any():
        insert_indices = check_fastener_possible_insertion(
            picked_up_fastener_tip_position, fastener_hole_positions
        )  # [B] int or -1
        valid_insert = screw_mask & (insert_indices >= 0)
        if valid_insert.any():
            hole_keys = list(fastener_hole_positions.keys())
            for scene_id in valid_insert.nonzero(as_tuple=False).squeeze(1).tolist():
                name = sim_state.tool_state[
                    scene_id
                ].current_tool.picked_up_fastener_name
                hole_name = hole_keys[insert_indices[scene_id].item()]
                activate_hand_connection(
                    scene, gs_entities[name], gs_entities["franka@control"]
                )
                sim_state.physical_state[scene_id].connect(name, hole_name, None)

    return success, total_diff_left, sim_state, diff
