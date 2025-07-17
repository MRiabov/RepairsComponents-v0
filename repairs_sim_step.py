import genesis as gs
from genesis.engine.entities.rigid_entity import RigidLink
import torch
from genesis.engine.entities import RigidEntity
from torch_geometric.data import Data

from repairs_components.geometry.connectors.connectors import check_connections
from repairs_components.geometry.fasteners import (
    Fastener,
    attach_fastener_to_screwdriver,
    attach_fastener_to_part,
    detach_fastener_from_screwdriver,
    check_fastener_possible_insertion,
    detach_fastener_from_part,
)
from repairs_components.logic.tools import tool
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import (
    Screwdriver,
    receive_fastener_pickup_action,
    receive_screw_in_action,
)
from repairs_components.processing.translation import (
    get_connector_pos,
    translate_genesis_to_python,
)
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.tool import (
    ToolsEnum,
    attach_tool_to_arm,
    detach_tool_from_arm,
)


def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    desired_state: RepairsSimState,
    starting_hole_positions: dict[str, torch.Tensor],
    starting_hole_quats: dict[str, torch.Tensor],
):
    """Step repairs sim.

    Abstraction note: anything that uses `actions` goes into step_repairs, anything that
    merely collects observations goes into translate"""
    # get values (state) from genesis
    # minor todo: male_connector_positions, female_connector_positions should be set into current_sim_state (because all the other logic does)
    current_sim_state, male_connector_positions, female_connector_positions = (
        translate_genesis_to_python(
            scene,
            gs_entities,
            current_sim_state,
            device=actions.device,
            starting_hole_positions=starting_hole_positions,
            starting_hole_quats=starting_hole_quats,
        )
    )

    current_sim_state = step_electronics(
        current_sim_state, male_connector_positions, female_connector_positions
    )

    # step pick up or release tool
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
    )
    # create_constraints_based_on_graph(current_sim_state, gs_entities, scene)

    print(
        f"Tools: {[ToolsEnum(ts.current_tool_id).name.lower() for ts in current_sim_state.tool_state]}",
        f"Pick up fastener tip position: {[ts.current_tool.picked_up_fastener_tip_position for ts in current_sim_state.tool_state if isinstance(ts.current_tool, Screwdriver)]}",
    )

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
        Screwdriver.tool_grip_position().unsqueeze(0),
    )
    hand_pos = franka_hand.get_pos().to(actions.device)

    dist = torch.full((scene.n_envs,), fill_value=-1.0, device=actions.device)
    # pick up or drop tool
    if pick_up_tool_mask.any():
        # TODO logic for more tools.
        dist = torch.norm(hand_pos.squeeze(1) - grip_pos.squeeze(1), dim=-1)
        # TODO: not looking for closest tool, but should look for closest only.
        required_dist = torch.tensor(
            Screwdriver.dist_from_grip_link(),  # note: singular float.
            dtype=torch.float,
            device=actions.device,
        )
        pick_up_tool_mask_and_close = pick_up_tool_mask & (dist < required_dist)

        if pick_up_tool_mask_and_close.any():
            env_idx = pick_up_tool_mask_and_close.nonzero().squeeze(1)
            print(f"picking up tool at env_idx: {env_idx.tolist()}")
            for env_id in env_idx.tolist():
                current_sim_state.tool_state[env_id].current_tool = Screwdriver()
                # attach the screwdriver to the hand
                # device issues?
            attach_tool_to_arm(scene, screwdriver, franka_hand, Screwdriver(), env_idx)
    if release_tool_mask.any():
        env_idx = release_tool_mask.nonzero().squeeze(1)
        print(f"releasing tool at env_idx: {env_idx.tolist()}")

        current_tools = [
            current_sim_state.tool_state[env_id].current_tool
            for env_id in env_idx.tolist()
        ]
        for env_id in env_idx.tolist():
            current_sim_state.tool_state[env_id].current_tool = Gripper()
            # detach the screwdriver from the hand
        detach_tool_from_arm(
            scene, screwdriver, franka_hand, gs_entities, env_idx, current_tools
        )

    return current_sim_state


# untested
def step_screw_in_or_out(
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    actions: torch.Tensor,
):
    "A method to update fastener attachments based on actions and proximity."
    tool_state = current_sim_state.tool_state
    physical_state = current_sim_state.physical_state
    # Fastener insertion batch
    # mask where `screw_in>threshold` and the picked up tool is Screwdriver
    # note:if there are more tools, create an interface, I guess.

    fastener_connected_to = torch.stack(
        [phys_state.graph.fasteners_attached_to for phys_state in physical_state],
    ).to(actions.device)
    picked_up_fastener_ids = torch.tensor(
        [
            int(ts.current_tool.picked_up_fastener_name.split("@")[0])
            if isinstance(ts.current_tool, Screwdriver)
            and ts.current_tool.has_picked_up_fastener
            else -1
            for ts in tool_state
        ],
        device=actions.device,
    )
    has_picked_up_fastener_mask = picked_up_fastener_ids != -1

    screw_in_desired_mask, screw_out_desired_mask = receive_screw_in_action(
        actions
    )  # [B]

    # check if fastener is already connected to two parts
    fastener_disconnected_mask = fastener_connected_to == -1
    fastener_fully_connected_mask = ~fastener_disconnected_mask[
        torch.arange(scene.n_envs), picked_up_fastener_ids
    ].any(dim=-1)  # [B]  # neither of connections is -1
    fastener_partially_connected_mask = fastener_disconnected_mask[
        torch.arange(scene.n_envs), picked_up_fastener_ids
    ].any(dim=-1)  # [B]  # any of connections is -1
    # ^ this is incorrect logic.

    # where they actually happen
    has_fastener_and_desired_in = (
        has_picked_up_fastener_mask  # must have picked up fastener
        & screw_in_desired_mask
        & ~fastener_fully_connected_mask  # can screw in only when fastener has <2 connections.
    )  # note: fastener_fully_connected_mask can take -1, but it is cancelled out by has_picked_up_fastener_mask
    # ^ beware when debugging.
    has_fastener_and_desired_out = (
        has_picked_up_fastener_mask
        & screw_out_desired_mask
        & fastener_partially_connected_mask  # can't screw out where it's not connected anywhere
    )

    # step screw insertion
    if has_fastener_and_desired_in.any():
        # FIXME: should not be able to insert fastener with two connections already!

        env_ids = has_fastener_and_desired_in.nonzero().squeeze(1)
        ignore_part_idx = (  # fastener_already_in_idx
            fastener_connected_to[env_ids, picked_up_fastener_ids[env_ids]]
            .max(dim=-1)
            .values
        )  # because ignore_hole_idx is a tensor of shape [2], with always values of either [-1, -1], [-1, value], [value, -1],
        # we can take max to get the value or -1.

        part_idx, hole_idx = check_fastener_possible_insertion(
            active_fastener_tip_position=torch.stack(
                [
                    tool_state[env_id].current_tool.picked_up_fastener_tip_position
                    for env_id in env_ids
                ]
            ),
            part_hole_positions=torch.stack(
                [physical_state[env_id].hole_positions for env_id in env_ids]
            ),
            part_hole_quats=torch.stack(
                [physical_state[env_id].hole_quats for env_id in env_ids]
            ),
            part_hole_batch=torch.stack(
                [physical_state[env_id].hole_indices_batch for env_id in env_ids]
            ),
            # as fastener quat is equal to screwdriver quat, we can pass it here
            active_fastener_quat=gs_entities["screwdriver@control"].get_quat(env_ids),
            ignore_part_idx=ignore_part_idx,
            # note: ignore hole idx could be moved out to get -1s separately but idgaf.
        )  # [B] int or -1
        valid_insert = part_idx >= 0
        # if valid_insert.any():
        #     # FIXME: this never executes! (i've never seen it, at least.)
        #     # ^ note: just seen it execute.
        #     # but in some cases it threw cuda error?
        for env_id in env_ids[valid_insert].tolist():
            hole_pos = physical_state[env_id].hole_positions[hole_idx[env_id].item()]
            hole_quat = physical_state[env_id].hole_quats[hole_idx[env_id].item()]
            fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
            part_id = part_idx[env_id].item()
            part_name = physical_state[env_id].inverse_body_indices[part_id]
            # note: fasteners that are already connected are ignored in check_fastener_possible_insertion
            # FIXME: body_idx should be gettable from fastener_hole_positions
            attach_fastener_to_part(
                scene,
                fastener_entity=gs_entities[fastener_name],
                hole_pos=hole_pos,
                hole_quat=hole_quat,
                part_entity=gs_entities[part_name],
                envs_idx=torch.tensor([env_id], device=actions.device),
            )
            # future: assert (prevent) more than two connections.

            ##Design choice: don't remove connection after screw in to allow ML to constrain two parts more easily.

            physical_state[env_id].connect(fastener_name, part_name, None)

    if has_fastener_and_desired_out.any():
        for env_id in (
            has_fastener_and_desired_out.nonzero(as_tuple=False).squeeze(1).tolist()
        ):
            assert tool_state[env_id].current_tool.picked_up_fastener_name is not None
            fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
            # check if there is one or two parts (or none)
            # for every non -1, check delete weld constraints from body_indices
            fastener_id = int(
                fastener_name.split("@")[0]
            )  # fasteners have naming as "{id}@fastener"
            fastener_body_indices = physical_state[env_id].graph.fasteners_attached_to[
                fastener_id
            ]
            for body_idx in fastener_body_indices:
                if body_idx != -1:
                    body_name = physical_state[env_id].inverse_indices[body_idx.item()]
                    detach_fastener_from_part(
                        scene,
                        fastener_entity=gs_entities[fastener_name],
                        part_entity=gs_entities[body_name],
                        envs_idx=torch.tensor([env_id], device=actions.device),
                    )

            screwdriver_grip_pos = get_connector_pos(
                gs_entities["screwdriver@control"].get_pos(env_id),
                gs_entities["screwdriver@control"].get_quat(env_id),
                Screwdriver.fastener_connector_pos_relative_to_center().unsqueeze(0),
            )

            attach_fastener_to_screwdriver(  # reattach the fastener to hand.
                scene,
                fastener_entity=gs_entities[fastener_name],
                screwdriver_entity=gs_entities["franka@control"],
                screwdriver_grip_xyz=screwdriver_grip_pos,
                env_id=env_id,
                tool_state_to_update=tool_state[env_id].current_tool,
                fastener_id=fastener_id,
            )

    print(f"has_fastener_and_desired_in {has_fastener_and_desired_in}")
    print(f"has_fastener_and_desired_out {has_fastener_and_desired_out}")

    return current_sim_state


def step_fastener_pick_up_release(
    scene: gs.Scene,
    gs_entities: dict[str, RigidEntity],
    current_sim_state: RepairsSimState,
    actions: torch.Tensor,
    max_pick_up_threshold: float = 2,  # 2m, should be 10cm (!)
):
    "Step fastener pick up/release."
    tool_state = current_sim_state.tool_state

    screwdriver_picked_up = torch.tensor(
        [isinstance(ts.current_tool, Screwdriver) for ts in tool_state],
        dtype=torch.bool,
        device=actions.device,
    )
    screwdriver_with_fastener_mask = torch.tensor(
        [
            w_screwdriver and (ts.current_tool.picked_up_fastener_name is not None)
            for w_screwdriver, ts in zip(screwdriver_picked_up.tolist(), tool_state)
        ],
        dtype=torch.bool,
        device=actions.device,
    )
    pick_up_desired_mask, release_desired_mask = receive_fastener_pickup_action(actions)
    # can pick up when screwdriver but empty, can release when has screwdriver with fastener
    pick_up_mask = (
        pick_up_desired_mask & screwdriver_picked_up & ~screwdriver_with_fastener_mask
    )
    release_mask = release_desired_mask & screwdriver_with_fastener_mask
    assert not (pick_up_mask & release_mask).any(), (
        "pick up and release can not happen at the same time"
    )

    if pick_up_mask.any():
        desired_pick_up_indices = pick_up_mask.nonzero().squeeze(1)

        # update screwdriver gripper position
        screwdriver_pos = gs_entities["screwdriver@control"].get_pos(
            desired_pick_up_indices
        )
        screwdriver_quat = gs_entities["screwdriver@control"].get_quat(
            desired_pick_up_indices
        )
        screwdriver_gripper_pos = get_connector_pos(
            screwdriver_pos,
            screwdriver_quat,
            Screwdriver.fastener_connector_pos_relative_to_center().unsqueeze(0),
        ).unsqueeze(1)  # unsqueeze so it can be broadcasted to fastener pos
        # calculate positions of fasteners close to screwdriver_gripper_pos
        fastener_positions = (
            torch.stack(
                [
                    current_sim_state.physical_state[i].graph.fasteners_loc
                    for i in desired_pick_up_indices
                ]
            )
            .to_dense()
            .to(actions.device)
        )
        fastener_distances = torch.norm(
            fastener_positions - screwdriver_gripper_pos, dim=-1
        )  # made changes, not sure it works.
        fastener_distances = fastener_distances.min(dim=-1)  # [B]
        closest_fastener_id = fastener_distances.indices
        close_enough = fastener_distances.values < max_pick_up_threshold

        print("fastener pick up mask", pick_up_mask, "Close enough:", close_enough)

        for i, env_id in enumerate(desired_pick_up_indices[close_enough].tolist()):
            fastener_name = Fastener.fastener_name_in_simulation(
                closest_fastener_id[i].item()
            )
            attach_fastener_to_screwdriver(
                scene,
                fastener_entity=gs_entities[fastener_name],
                screwdriver_entity=gs_entities["franka@control"],
                env_id=env_id,
                tool_state_to_update=tool_state[env_id].current_tool,
                fastener_id=closest_fastener_id[i].item(),
            )

    ##debug
    assert all(
        tool_state[i].current_tool.picked_up_fastener_name is not None
        for i in screwdriver_with_fastener_mask.nonzero().squeeze(1).tolist()
    ), (
        "screwdriver_with_fastener_mask should be true only for those with fastener picked up"
    )
    # // # somehow this fails.

    if release_mask.any():
        desired_release_indices = release_mask.nonzero().squeeze(1)
        for env_id in desired_release_indices.tolist():
            assert tool_state[env_id].current_tool.picked_up_fastener_name is not None
            fastener_name = tool_state[env_id].current_tool.picked_up_fastener_name
            detach_fastener_from_screwdriver(
                scene,
                fastener_entity=gs_entities[fastener_name],
                screwdriver_entity=gs_entities["franka@control"],
                tool_state_to_update=tool_state[env_id].current_tool,
                env_id=env_id,
            )
        print("fastener release mask", release_mask)

    return current_sim_state
