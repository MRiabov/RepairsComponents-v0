import genesis as gs
from genesis.engine.entities.rigid_entity import RigidLink
import torch
from genesis.engine.entities import RigidEntity

from repairs_components.geometry.connectors.connectors import check_connections
from repairs_components.geometry.fasteners import (
    activate_hand_connection,
    check_fastener_possible_insertion,
)
from repairs_components.logic.tools import tool
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import (
    Screwdriver,
    receive_screw_in_action,
)
from repairs_components.processing.translation import (
    translate_genesis_to_python,
)
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.tool import attach_tool_to_arm, detach_tool_from_arm


def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    desired_state: RepairsSimState,
):
    "Step repairs sim."
    # get values (state) from genesis
    (
        current_sim_state,
        picked_up_fastener_tip_position,
        fastener_hole_positions,
        male_connector_positions,
        female_connector_positions,
    ) = translate_genesis_to_python(
        scene, gs_entities, current_sim_state, device=actions.device
    )

    # step electronics attachments: compute, clear, and apply if present
    if current_sim_state.has_electronics:
        electronics_attachments_batch = check_connections(
            male_connector_positions, female_connector_positions
        )
        # clear all previous connections
        for es in current_sim_state.electronics_state:
            es.clear_connections()
        # apply batch attachments [batch_idx, male_idx, female_idx]
        for b, m, f in electronics_attachments_batch.tolist():
            current_sim_state.electronics_state[b].connect(m, f)
    else:
        electronics_attachments_batch = torch.empty((0, 3), dtype=torch.long)

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = current_sim_state.diff(desired_state)

    success = total_diff_left == 0  # Tensor[batch] bool
    if (
        False
    ):  # note: currently stuck because of cuda error. Genesis devs could resolve it.
        threshold_pick_up_tool = 0.75
        threshold_release_tool = 0.25

        # pick up tool logic
        pick_up_tool_desired = (
            actions[:, 9] > threshold_pick_up_tool
        )  # 0.75 is a threshold for picking up tool
        release_tool_desired = actions[:, 9] < threshold_release_tool
        gripper_mask = torch.tensor(
            [
                isinstance(ts.current_tool, Gripper)
                for ts in current_sim_state.tool_state
            ],
            dtype=torch.bool,
            device=actions.device,
        )
        pick_up_tool_mask = gripper_mask & pick_up_tool_desired
        release_tool_mask = (~gripper_mask) & release_tool_desired

        screwdriver: RigidEntity = gs_entities["screwdriver@control"]
        franka_hand: RigidEntity = gs_entities["franka@control"]
        grip_pos = screwdriver.get_pos().to(actions.device)
        hand_pos = franka_hand.get_pos().to(actions.device)

        # pick up or drop tool
        if pick_up_tool_mask.any():
            # TODO logic for more tools.
            dist = torch.norm(hand_pos.squeeze(1) - grip_pos.squeeze(1), dim=-1)
            required_dist = torch.tensor(
                current_sim_state.tool_state[0]
                .all_tools["screwdriver"]
                .dist_from_grip_link(),  # note: singular float.
                dtype=torch.float,
                device=actions.device,
            )
            pick_up_tool_mask_and_close = pick_up_tool_mask & (dist < required_dist)

            if pick_up_tool_mask_and_close.any():
                env_idx = pick_up_tool_mask_and_close.nonzero().squeeze(1)
                print(f"picking up at env_idx: {env_idx.tolist()}")
                for env_id in env_idx.tolist():
                    current_sim_state.tool_state[env_id].current_tool = Screwdriver()
                    # attach the screwdriver to the hand
                    # device issues?
                attach_tool_to_arm(scene, screwdriver, franka_hand, env_idx)
        if release_tool_mask.any():
            env_idx = release_tool_mask.nonzero().squeeze(1)
            print(f"releasing at env_idx: {env_idx.tolist()}")

            for env_id in env_idx.tolist():
                current_sim_state.tool_state[env_id].current_tool = Gripper()
                # detach the screwdriver from the hand
            detach_tool_from_arm(scene, screwdriver, franka_hand, env_idx)

        B = scene.n_envs
        # Fastener insertion batch
        # mask where action screw_in and tool is Screwdriver
        screw_mask = (
            torch.tensor(
                [
                    isinstance(ts.current_tool, Screwdriver)
                    for ts in current_sim_state.tool_state
                ],
                dtype=torch.bool,
                device=actions.device,
            )
            & receive_screw_in_action(
                actions
            )  # note:if there are more tools, create an interface, I guess.
            & ~torch.isnan(picked_up_fastener_tip_position).all(dim=-1)
        )  # nans are expected

        # step screw insertion
        if screw_mask.any():
            # TODO probably not work over all indices...
            insert_indices = check_fastener_possible_insertion(
                picked_up_fastener_tip_position, fastener_hole_positions
            )  # [B] int or -1
            valid_insert = screw_mask & (insert_indices >= 0)
            if valid_insert.any():
                hole_keys = list(fastener_hole_positions.keys())
                for env_id in valid_insert.nonzero(as_tuple=False).squeeze(1).tolist():
                    name = current_sim_state.tool_state[
                        env_id
                    ].current_tool.picked_up_fastener_name
                    hole_name = hole_keys[insert_indices[env_id].item()]
                    activate_hand_connection(
                        scene,
                        fastener_entity=gs_entities[name],
                        franka_arm=gs_entities["franka@control"],
                    )
                    current_sim_state.physical_state[env_id].connect(
                        name, hole_name, None
                    )

    return success, total_diff_left, current_sim_state, diff
