import genesis as gs
import torch

from repairs_components.geometry.fasteners import (
    attach_picked_up_fastener_to_part,
    attach_fastener_to_screwdriver,
    check_fastener_possible_insertion,
    detach_fastener_from_part,
    detach_fastener_from_screwdriver,
)
from repairs_components.logic.electronics.mna import solve_dc_once
from repairs_components.logic.physical_state import connect_fastener_to_one_body
from repairs_components.logic.tools.gripper import Gripper
from repairs_components.logic.tools.screwdriver import (
    Screwdriver,
    receive_fastener_pickup_action,
    receive_screw_in_action,
)
from repairs_components.logic.tools.tool import (
    ToolsEnum,
)
from repairs_components.processing.translation import (
    get_connector_pos,
    translate_genesis_to_python,
)
from repairs_components.training_utils.sim_state_global import (
    RepairsSimInfo,
    RepairsSimState,
)


def step_repairs(
    scene: gs.Scene,
    actions: torch.Tensor,
    sim_state: RepairsSimState,
    desired_state: RepairsSimState,
    sim_info: RepairsSimInfo,
):
    """Step repairs sim.

    Abstraction note: anything that uses `actions` goes into step_repairs, anything that
    merely collects observations goes into translate"""
    # get values (state) from genesis
    sim_state = translate_genesis_to_python(scene, sim_state, sim_info=sim_info)

    sim_state, terminated, burned_component_indices = step_electronics(
        sim_state, sim_info
    )

    # step pick up or release tool
    sim_state = step_pick_up_release_tool(scene, sim_info, sim_state, actions)

    # sim-to-real assumes a small buffer between implementation and action, so let us just make reward compute first and action second which is equal to getting the reward and stepping the action.
    diff, total_diff_left = sim_state.diff(desired_state, sim_info)

    success = (total_diff_left == 0) & (~terminated)  # Tensor[batch] bool

    # print(
    #     f"Scene step stats per scene: screw_mask {screw_mask}, pick_up_tool_mask {pick_up_tool_mask}, gripper_mask {gripper_mask}, release_tool_mask {release_tool_mask}, distance_to_grip_link {dist}"
    # )

    # step fastener pick up/release.
    sim_state = step_fastener_pick_up_release(scene, sim_state, sim_info, actions)

    # step screw in or out
    sim_state = step_screw_in_or_out(scene, sim_state, sim_info, actions)
    # create_constraints_based_on_graph(sim_state, gs_entities, scene)

    return success, total_diff_left, sim_state, diff, burned_component_indices


def step_electronics(sim_state: RepairsSimState, sim_info: RepairsSimInfo):
    """Step electronics by running the DC MNA solver once and writing outputs.

    Connectivity is established during translation; here we solve the circuit and
    update dynamic outputs (e.g., LED luminosity and motor speed percentages).
    """
    if sim_info.component_info.has_electronics:
        # Batched DC solve; writes back to sim_state.electronics_state
        solve_result = solve_dc_once(
            sim_info.component_info, sim_state.electronics_state
        )
        sim_state.electronics_state = solve_result.state
        terminated = solve_result.terminated
        burned_component_indices = solve_result.burned_component_indices
        return sim_state, terminated, burned_component_indices

    # No electronics registered: return sane defaults matching batch size
    bs = sim_state.tool_state.tool_ids.shape[0]
    device = sim_state.tool_state.tool_ids.device
    terminated = torch.zeros((bs,), dtype=torch.bool, device=device)
    burned_component_indices = torch.empty((0,), dtype=torch.long, device=device)
    return sim_state, terminated, burned_component_indices


def step_pick_up_release_tool(
    scene: gs.Scene,
    sim_info: RepairsSimInfo,
    sim_state: RepairsSimState,
    actions: torch.Tensor,
):
    "Step pick up or release tool."
    assert sim_state.tool_state.tool_ids.ndim == 1, "tool_ids should be 1D"
    # note: currently stuck because of cuda error. Genesis devs could resolve it.
    threshold_pick_up_tool = 0.75
    threshold_release_tool = 0.25

    # pick up tool logic
    pick_up_tool_desired = (
        actions[:, 9] > threshold_pick_up_tool
    )  # 0.75 is a threshold for picking up tool
    release_tool_desired = actions[:, 9] < threshold_release_tool
    gripper_mask = sim_state.tool_state.tool_ids == Gripper().id
    pick_up_tool_mask = gripper_mask & pick_up_tool_desired
    release_tool_mask = (~gripper_mask) & release_tool_desired

    screwdriver_link_idx = sim_info.tool_info.tool_base_link_idx[
        ToolsEnum.SCREWDRIVER.value
    ]
    gripper_link_idx = sim_info.tool_info.tool_base_link_idx[ToolsEnum.GRIPPER.value]

    grip_pos = get_connector_pos(
        scene.rigid_solver.get_links_pos(screwdriver_link_idx),
        scene.rigid_solver.get_links_quat(screwdriver_link_idx),
        sim_info.tool_info.TOOLS_GRIPPER_POS[ToolsEnum.SCREWDRIVER.value].unsqueeze(0),
    )
    hand_pos = scene.rigid_solver.get_links_pos(gripper_link_idx).to(actions.device)

    dist = torch.full((scene.n_envs,), fill_value=-1.0, device=actions.device)
    # pick up or drop tool
    if pick_up_tool_mask.any():
        # TODO logic for more tools.
        dist = torch.norm(hand_pos.squeeze(1) - grip_pos.squeeze(1), dim=-1)
        # TODO: not looking for closest tool, but should look for closest only.
        required_dist = sim_info.tool_info.TOOLS_DIST_FROM_GRIP_LINK[
            ToolsEnum.SCREWDRIVER.value
        ].to(actions.device)
        pick_up_tool_within_reach_mask = pick_up_tool_mask & (dist < required_dist)

        if pick_up_tool_within_reach_mask.any():
            env_idx = pick_up_tool_within_reach_mask.nonzero().squeeze(1)
            sim_state.tool_state.tool_ids[env_idx] = ToolsEnum.SCREWDRIVER.value
            # attach the screwdriver to the hand
            # device issues?
            from repairs_components.logic.tools.tool import attach_tool_to_arm

            attach_tool_to_arm(scene, sim_state, sim_info, env_idx)
    if release_tool_mask.any():
        env_idx = release_tool_mask.nonzero().squeeze(1)
        sim_state.tool_state.tool_ids[env_idx] = Gripper().id
        # Only update tool_ids here; if constraints need removal, handle it in tool utilities.

    return sim_state


def step_screw_in_or_out(
    scene: gs.Scene,
    sim_state: RepairsSimState,
    sim_info: RepairsSimInfo,
    actions: torch.Tensor,  # [B, 10]
):
    "A method to update fastener attachments based on actions and proximity."
    tool_state = sim_state.tool_state
    physical_state = sim_state.physical_state
    # Fastener insertion batch
    # mask where `screw_in>threshold` and the picked up tool is Screwdriver
    # note:if there are more tools, create an interface, I guess.

    picked_up_fastener_ids = tool_state.screwdriver_tc.picked_up_fastener_id
    has_picked_up_fastener_mask = tool_state.screwdriver_tc.has_picked_up_fastener

    screw_in_desired_mask, screw_out_desired_mask = receive_screw_in_action(
        actions
    )  # [B]

    # connection status masks for the picked-up fastener in each env
    fastener_disconnected_mask = (
        physical_state.fasteners_attached_to_body == -1
    )  # [B, F, 2]
    picked_mask = fastener_disconnected_mask[
        torch.arange(scene.n_envs), picked_up_fastener_ids
    ]  # [B, 2]
    # fully connected => neither slot is -1
    fastener_fully_connected_mask = ~picked_mask.any(dim=-1)  # [B]
    # connected to at least one body (covers both partial and full)
    fastener_connected_to_any_mask = ~picked_mask.all(dim=-1)  # [B]

    # check if the fastener is already in a blind hole (non-through), vectorized and -1 safe
    if sim_info.physical_info.hole_is_through.numel() == 0:
        fastener_in_blind_hole_mask = torch.zeros(
            scene.n_envs, dtype=torch.bool, device=actions.device
        )
    else:
        holes_for_fastener = physical_state.fasteners_attached_to_hole[
            torch.arange(scene.n_envs), picked_up_fastener_ids
        ]  # [B,2]
        valid_slots = holes_for_fastener != -1
        holes_safe = holes_for_fastener.clone()
        holes_safe[~valid_slots] = 0
        through_flags = sim_info.physical_info.hole_is_through[holes_safe]
        through_flags[~valid_slots] = True
        fastener_in_blind_hole_mask = (~through_flags).any(dim=-1)
    # TODO: untested; expected that this will be equal to screw_in_desired_mask.shape
    # ### debug
    # assert fastener_in_blind_hole_mask.shape == screw_in_desired_mask.shape, (
    #     f"fastener_in_blind_hole_mask.shape != screw_in_desired_mask.shape: {fastener_in_blind_hole_mask.shape} != {screw_in_desired_mask.shape}"
    # )
    # ###/debug

    # where they actually happen
    has_fastener_and_desired_in = (
        has_picked_up_fastener_mask  # must have picked up fastener
        & screw_in_desired_mask
        & ~fastener_fully_connected_mask  # can screw in only when fastener has <2 connections.
        & ~fastener_in_blind_hole_mask
    )  # note: fastener_fully_connected_mask can take -1, but it is cancelled out by has_picked_up_fastener_mask
    # ^ beware when debugging.
    has_fastener_and_desired_out = (
        has_picked_up_fastener_mask
        & screw_out_desired_mask
        & fastener_connected_to_any_mask  # can screw out when connected to at least one body
    )

    # step screw insertion
    if has_fastener_and_desired_in.any():
        assert sim_info.physical_info.part_hole_batch.numel() > 0, (
            "part_hole_batch metadata was not set."
        )
        env_ids = has_fastener_and_desired_in.nonzero(as_tuple=False).squeeze(1)

        fids = picked_up_fastener_ids[env_ids]
        # Determine part to ignore per env (if already connected in this env)
        fastener_body_indices = physical_state.fasteners_attached_to_body[
            env_ids, fids
        ]  # [N,2]
        ignore_part_idx = fastener_body_indices.max(dim=1).values  # [N]

        # Batched query for possible insertion
        part_idx_tensor, hole_idx_tensor = check_fastener_possible_insertion(
            scene,
            sim_state,
            sim_info,
            env_ids=env_ids,
            ignore_part_idx=ignore_part_idx,
        )  # expected [N], [N]

        insertion_selection_mask = (part_idx_tensor >= 0) & (hole_idx_tensor >= 0)
        if insertion_selection_mask.any():
            env_ids_insert = env_ids[insertion_selection_mask]
            fastener_ids_insert = fids[insertion_selection_mask]
            hole_ids_insert = hole_idx_tensor[insertion_selection_mask]

            # already inserted hole id per env/fastener (or -1)
            already_inserted_hole_ids = (
                physical_state.fasteners_attached_to_hole[env_ids_insert, fastener_ids_insert]
                .max(dim=1)
                .values
            )

            # Attach batched
            attach_picked_up_fastener_to_part(
                scene,
                sim_state,
                sim_info,
                hole_ids_insert,
                already_inserted_hole_ids,
                env_ids_insert,
            )

            # Update graph state (batched)
            sim_state.physical_state = connect_fastener_to_one_body(
                sim_state.physical_state,
                sim_info.physical_info,
                fastener_ids_insert,
                hole_ids_insert,
                env_ids_insert,
            )

    if has_fastener_and_desired_out.any():
        env_out_ids = has_fastener_and_desired_out.nonzero(as_tuple=False).squeeze(1)
        fids_out = tool_state.screwdriver_tc.picked_up_fastener_id[env_out_ids]
        assert (fids_out >= 0).all(), "fastener ids must be valid for screw-out"

        body_idx = physical_state.fasteners_attached_to_body[
            env_out_ids, fids_out
        ]  # [N,2]
        connected_mask = body_idx != -1  # [N,2]
        num_connected_slots_per_env = connected_mask.sum(dim=1)
        if num_connected_slots_per_env.any():
            part_ids_flat = body_idx[connected_mask]
            envs_idx_flat = env_out_ids.repeat_interleave(num_connected_slots_per_env)
            sim_state = detach_fastener_from_part(
                scene, sim_state, sim_info, part_ids_flat, envs_idx_flat
            )

        # Graph-state: set to -1 for all slots in these envs for these fasteners
        sim_state.physical_state.fasteners_attached_to_body[
            env_out_ids, fids_out, :
        ] = -1
        sim_state.physical_state.fasteners_attached_to_hole[
            env_out_ids, fids_out, :
        ] = -1

        # Reattach to screwdriver (batched)
        attach_fastener_to_screwdriver(
            scene, sim_state, sim_info, fids_out, env_out_ids
        )

    return sim_state


def step_fastener_pick_up_release(
    scene: gs.Scene,
    sim_state: RepairsSimState,
    sim_info: RepairsSimInfo,
    actions: torch.Tensor,
    max_pick_up_threshold: float = 2,  # 2m, should be 10cm (!)
):
    "Step fastener pick up/release."
    tool_state = sim_state.tool_state
    assert tool_state.tool_ids.ndim == 1, "tool_ids should be 1D"

    screwdriver_picked_up = tool_state.tool_ids == ToolsEnum.SCREWDRIVER.value
    screwdriver_with_fastener_mask = tool_state.screwdriver_tc.has_picked_up_fastener

    pick_up_desired_mask, release_desired_mask = receive_fastener_pickup_action(actions)
    # can pick up when screwdriver but empty, can release when has screwdriver with fastener
    pick_up_mask = (
        pick_up_desired_mask & screwdriver_picked_up & ~screwdriver_with_fastener_mask
    )  # expected [B]
    release_mask = release_desired_mask & screwdriver_with_fastener_mask  # expected [B]
    assert not (pick_up_mask & release_mask).any(), (
        "pick up and release can not happen at the same time"
    )
    # sanity check:
    assert pick_up_mask.ndim == release_mask.ndim == 1

    if pick_up_mask.any():
        env_ids_pick = pick_up_mask.nonzero(as_tuple=False).squeeze(1)

        # update screwdriver gripper position
        screwdriver_pos = tool_state.picked_up_pos[env_ids_pick]
        screwdriver_quat = tool_state.picked_up_quat[env_ids_pick]
        rel_connector_pos = (
            Screwdriver.fastener_connector_pos_relative_to_center().expand(
                screwdriver_pos.shape[0], -1
            )
        )  # [N,3]
        screwdriver_gripper_pos = get_connector_pos(
            screwdriver_pos, screwdriver_quat, rel_connector_pos
        ).unsqueeze(1)  # [N,1,3]
        # calculate positions of fasteners close to screwdriver_gripper_pos
        fastener_positions = sim_state.physical_state.fasteners_pos[env_ids_pick]
        # Distances per env, per fastener -> [N, F]
        dists = torch.norm(fastener_positions - screwdriver_gripper_pos, dim=-1)
        # Reduce over fasteners -> values [N], indices [N]
        per_env_min_vals, per_env_min_ids = dists.min(dim=1)
        within_pickup_range_mask = per_env_min_vals < max_pick_up_threshold

        if within_pickup_range_mask.any():
            env_ids_pickup = env_ids_pick[within_pickup_range_mask]
            fastener_ids_pickup = per_env_min_ids[within_pickup_range_mask]
            attach_fastener_to_screwdriver(
                scene, sim_state, sim_info, fastener_ids_pickup, env_ids_pickup
            )

    assert (
        tool_state.screwdriver_tc.has_picked_up_fastener[screwdriver_with_fastener_mask]
    ).all(), (
        "screwdriver_with_fastener_mask should be true only for those with fastener picked up"
    )

    if release_mask.any():
        env_ids_rel = release_mask.nonzero(as_tuple=False).squeeze(1)
        fids_rel = tool_state.screwdriver_tc.picked_up_fastener_id[env_ids_rel]
        valid_release_env_mask = fids_rel >= 0
        if valid_release_env_mask.any():
            detach_fastener_from_screwdriver(
                scene,
                sim_state,
                sim_info,
                env_ids_rel[valid_release_env_mask],
            )

    return sim_state
