import genesis as gs
from genesis.engine.entities.rigid_entity import RigidLink
import torch
from genesis.engine.entities import RigidEntity
from torch_geometric.data import Data

from repairs_components.geometry.connectors.connectors import check_connections
from repairs_components.geometry.fasteners import (
    Fastener,
    activate_fastener_to_hand_connection,
    activate_part_to_fastener_connection,
    deactivate_fastener_to_hand_connection,
    check_fastener_possible_insertion,
    deactivate_part_connection,
)
from repairs_components.logic.tools import tool
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import (
    Screwdriver,
    receive_fastener_pickup_action,
    receive_screw_in_action,
)
from repairs_components.processing.translation import (
    create_constraints_based_on_graph,
    get_connector_pos,
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
    starting_hole_positions: dict[str, torch.Tensor],
    starting_hole_quats: dict[str, torch.Tensor],
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

    current_sim_state = step_electronics(
        current_sim_state, male_connector_positions, female_connector_positions
    )

    # step update hole locs.
    current_sim_state = update_hole_locs(
        current_sim_state,
        starting_hole_positions,
        starting_hole_quats,
    )  # would be ideal if starting_hole_positions, hole_quats and hole_batch were batched already.

    # step release tool
    current_sim_state = step_pick_up_release_tool(
        scene, gs_entities, current_sim_state, actions
    )

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = current_sim_state.diff(desired_state)

    success = total_diff_left == 0  # Tensor[batch] bool

    # print(
    #     f"Scene step stats per scene: screw_mask {screw_mask}, pick_up_tool_mask {pick_up_tool_mask}, gripper_mask {gripper_mask}, release_tool_mask {release_tool_mask}, distance_to_grip_link {dist}"
    # )

    # step fastener pick up/release.
    current_sim_state = step_fastener_pick_up_release(
        scene, gs_entities, current_sim_state, actions
    )

    # step screw in or out
    current_sim_state = step_screw_in_or_out(
        scene,
        gs_entities,
        current_sim_state,
        actions,
        fastener_hole_positions,
        picked_up_fastener_tip_position,
    )
    create_constraints_based_on_graph(current_sim_state, gs_entities, scene)

    return success, total_diff_left, current_sim_state, diff


def step_electronics(
    current_sim_state: RepairsSimState,
    male_connector_positions,
    female_connector_positions,
):
    "Step electronics attachments: compute, clear, and apply if present"
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

    return current_sim_state


def step_pick_up_release_tool(
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    actions: torch.Tensor,
):
    "Step pick up or release tool."
    # `if False` was here.
    # note: currently stuck because of cuda error. Genesis devs could resolve it.
    threshold_pick_up_tool = 0.75
    threshold_release_tool = 0.25

    # pick up tool logic
    pick_up_tool_desired = (
        actions[:, 9] > threshold_pick_up_tool
    )  # 0.75 is a threshold for picking up tool
    release_tool_desired = actions[:, 9] < threshold_release_tool
    gripper_mask = torch.tensor(
        [isinstance(ts.current_tool, Gripper) for ts in current_sim_state.tool_state],
        dtype=torch.bool,
        device=actions.device,
    )
    pick_up_tool_mask = gripper_mask & pick_up_tool_desired
    release_tool_mask = (~gripper_mask) & release_tool_desired

    screwdriver: RigidEntity = gs_entities["screwdriver@control"]
    franka_hand: RigidEntity = gs_entities["franka@control"]

    grip_pos = get_connector_pos(
        screwdriver.get_pos(),
        screwdriver.get_quat(),
        Screwdriver.tool_connector_pos_relative_to_center(),
    )
    hand_pos = franka_hand.get_pos().to(actions.device)

    dist = torch.full((scene.n_envs,), fill_value=-1.0, device=actions.device)
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

    print(
        f"pick_up_tool_mask {pick_up_tool_mask}, release_tool_mask {release_tool_mask}, pick_up_tool_mask_and_close {pick_up_tool_mask_and_close}"
    )

    return current_sim_state


# untested
def step_screw_in_or_out(
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    actions: torch.Tensor,
    fastener_hole_positions: dict[str, torch.Tensor],
    picked_up_fastener_tip_position: torch.Tensor,
):
    "A method to update fastener attachments based on actions and proximity."
    tool_state = current_sim_state.tool_state
    physical_state = current_sim_state.physical_state
    # Fastener insertion batch
    # mask where `screw_in>threshold` and the picked up tool is Screwdriver
    # note:if there are more tools, create an interface, I guess.

    has_picked_up_fastener_mask = torch.tensor(
        [
            isinstance(ts.current_tool, Screwdriver)
            and ts.current_tool.has_picked_up_fastener
            for ts in tool_state
        ],
        dtype=torch.bool,
        device=actions.device,
    )
    nan_mask = torch.isnan(picked_up_fastener_tip_position).all(dim=-1)
    screw_in_mask, screw_out_mask = receive_screw_in_action(actions)

    screw_in_mask = has_picked_up_fastener_mask & screw_in_mask & ~nan_mask
    screw_out_mask = has_picked_up_fastener_mask & screw_out_mask & ~nan_mask

    # step screw insertion
    if screw_in_mask.any():
        # TODO probably not work over all indices...
        insert_indices = check_fastener_possible_insertion(
            picked_up_fastener_tip_position[screw_in_mask],
            fastener_hole_positions,
        )  # [B] int or -1
        valid_insert = insert_indices >= 0
        if valid_insert.any():
            part_names = list(fastener_hole_positions.keys())
            for env_id in valid_insert.nonzero(as_tuple=False).squeeze(1).tolist():
                fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
                part_name = part_names[insert_indices[env_id].item()]
                # TODO ideally check whether the fastener is already connected to the hole (WIP).
                if not (
                    physical_state[env_id].graph.fasteners_attached_to[
                        fastener_name.split("@")[0]
                    ]
                    == physical_state[env_id].body_indices[part_name]
                ).any():
                    # FIXME: body_idx should be gettable from fastener_hole_positions
                    activate_part_to_fastener_connection(
                        scene,
                        fastener_entity=gs_entities[fastener_name],
                        part_entity=gs_entities[part_name],
                        hole_link_name=part_name,
                        envs_idx=torch.tensor([env_id], device=actions.device),
                    )
                    # future: assert (prevent) more than two connections.

                ##Design choice: don't remove connection after screw in to allow ML to constrain two parts more easily.

                physical_state[env_id].connect(fastener_name, part_name, None)

    if screw_out_mask.any():
        for env_id in screw_out_mask.nonzero(as_tuple=False).squeeze(1).tolist():
            assert tool_state[env_id].current_tool.picked_up_fastener_name is not None
            fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
            # check if there is one or two parts (or none)
            # for every non -1, check delete weld constraints from body_indices
            fastener_id = fastener_name.split("@")[
                0
            ]  # fasteners have naming as "{id}@fastener"
            fastener_body_indices = physical_state[env_id].graph.fasteners_attached_to[
                fastener_id
            ]
            for body_idx in fastener_body_indices:
                if body_idx != -1:
                    body_name = physical_state[env_id].body_names[body_idx]
                    deactivate_part_connection(
                        scene,
                        fastener_entity=gs_entities[fastener_name],
                        part_entity=gs_entities[body_name],
                        hole_link_name=part_name,  # FIXME: no hole name!!!
                        envs_idx=torch.tensor([env_id], device=actions.device),
                    )

            screwdriver_grip_pos = get_connector_pos(
                gs_entities["screwdriver@control"].get_pos(env_id),
                gs_entities["screwdriver@control"].get_quat(env_id),
                Screwdriver.fastener_connector_pos_relative_to_center(),
            )

            activate_fastener_to_hand_connection(  # reattach the fastener to hand.
                scene,
                fastener_entity=gs_entities[fastener_name],
                franka_arm=gs_entities["franka@control"],
                reposition_to_xyz=screwdriver_grip_pos,
                rotate_to_quat=gs_entities["screwdriver@control"].get_quat(env_id),
                envs_idx=torch.tensor([env_id], device=actions.device),
                tool_state_to_update=tool_state,
            )

    print(f"screw_mask {screw_in_mask}")

    return current_sim_state


def step_fastener_pick_up_release(
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    actions: torch.Tensor,
    max_pick_up_threshold: float = 0.1,  # 10cm (!)
):
    "Step fastener pick up/release."
    tool_state = current_sim_state.tool_state
    physical_state = current_sim_state.physical_state

    screwdriver_mask = torch.tensor(
        [isinstance(ts.current_tool, Screwdriver) for ts in tool_state],
        dtype=torch.bool,
        device=actions.device,
    )
    pick_up_mask, release_mask = receive_fastener_pickup_action(
        actions[screwdriver_mask]
    )

    if pick_up_mask.any():
        desired_pick_up_indices = pick_up_mask.nonzero().squeeze(1)

        # update screwdriver gripper position
        screwdriver_pos = gs_entities["screwdriver@control"].get_pos()
        screwdriver_quat = gs_entities["screwdriver@control"].get_quat()
        screwdriver_gripper_pos = get_connector_pos(
            screwdriver_pos,
            screwdriver_quat,
            Screwdriver.fastener_connector_pos_relative_to_center(),
        )
        # calculate positions of fasteners close to screwdriver_gripper_pos
        fastener_positions = torch.stack(
            [
                gs_entities[fastener_name].get_pos()
                for fastener_name in physical_state.fastener_names
            ]
        )
        fastener_distances = torch.norm(
            fastener_positions[desired_pick_up_indices] - screwdriver_gripper_pos,
            dim=-1,
        )
        fastener_distances = fastener_distances.min(dim=-1)  # [B]
        close_enough = fastener_distances < max_pick_up_threshold
        closest_fastener_id = fastener_distances[close_enough].argmin(dim=-1)  # [B]

        for i, env_id in enumerate(desired_pick_up_indices[close_enough].tolist()):
            fastener_name = f"{closest_fastener_id[i]}@fastener"
            activate_fastener_to_hand_connection(
                scene,
                fastener_entity=gs_entities[fastener_name],
                franka_arm=gs_entities["franka@control"],
                reposition_to_xyz=screwdriver_gripper_pos[i],
                rotate_to_quat=screwdriver_quat[i],
                envs_idx=torch.tensor([env_id], device=actions.device),
                tool_state_to_update=tool_state,
            )  # right?

    if release_mask.any():
        desired_release_indices = release_mask.nonzero().squeeze(1)
        for i, env_id in enumerate(desired_release_indices.tolist()):
            fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
            deactivate_fastener_to_hand_connection(
                scene,
                fastener_entity=gs_entities[fastener_name],
                franka_arm=gs_entities["franka@control"],
                tool_state_to_update=tool_state,
                envs_idx=torch.tensor([env_id], device=actions.device),
            )

    return current_sim_state


def update_hole_locs(
    current_sim_state: RepairsSimState,
    starting_hole_positions: dict[str, torch.Tensor],
    starting_hole_quats: dict[str, torch.Tensor],
):
    """Update hole locs.

    Args:
        current_sim_state (RepairsSimState): The current sim state.
        starting_hole_positions (dict[str, torch.Tensor]): The starting hole positions.
        starting_hole_quats (dict[str, torch.Tensor]): The starting hole quats.

    Process:
    1. Stack the starting hole positions and quats into one [B, sum(num_holes_per_part), 3] and [B, sum(num_holes_per_part), 4]
    2. Repeat the part positions and quats for each hole
    3. Calculate the hole positions and quats
    4. Set the hole positions and quats
    """
    all_starting_hole_positions = torch.stack(list(starting_hole_positions.values()))
    all_starting_hole_quats = torch.stack(list(starting_hole_quats.values()))
    num_holes_per_part = [v.shape[0] for v in starting_hole_positions.values()]
    hole_batch = torch.repeat_interleave(
        torch.arange(
            len(num_holes_per_part), device=all_starting_hole_positions.device
        ),
        num_holes_per_part,
    )

    # will remove when (if) batch RepairsSimStep.
    part_pos = torch.stack(
        [phys_state.graph.position for phys_state in current_sim_state.physical_state]
    )
    part_quat = torch.stack(
        [phys_state.graph.quat for phys_state in current_sim_state.physical_state]
    )

    part_pos_batch = torch.repeat_interleave(part_pos, num_holes_per_part, dim=0)
    part_quat_batch = torch.repeat_interleave(part_quat, num_holes_per_part, dim=0)

    # note: not sure get_connector_pos will be usable with batches.
    hole_pos = get_connector_pos(
        part_pos_batch, part_quat_batch, all_starting_hole_positions
    )
    # no custom function for quat, calculate directly (sum quats)
    hole_quat = torch.einsum(
        "bhw,bhw->bh", part_quat_batch, all_starting_hole_quats
    )  # i hope correct
    for i, phys_state in enumerate(current_sim_state.physical_state):
        phys_state.hole_positions = hole_pos[i]
        phys_state.hole_quats = hole_quat[i]
    return current_sim_state
