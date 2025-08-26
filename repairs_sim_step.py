import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink

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
    detach_tool_from_arm,
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

    print(
        f"Tools: {[ToolsEnum(tool_id.item()).name.lower() for tool_id in sim_state.tool_state.tool_ids]}",
        f"Pick up fastener tip position: {sim_state.tool_state.screwdriver_tc.picked_up_fastener_tip_position[sim_state.tool_state.screwdriver_tc.has_picked_up_fastener]}",
    )

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

    screwdriver_link: RigidEntity = sim_info.tool_info.tool_base_links_idx[
        ToolsEnum.SCREWDRIVER.value
    ]
    # hand, not arm.
    franka_hand_link: RigidLink = sim_info.tool_info.tool_base_links_idx[
        ToolsEnum.GRIPPER.value
    ]

    from repairs_components.logic.tools.tool import ToolsEnum
    from repairs_components.logic.tools.tools_state import ToolInfo

    grip_pos = get_connector_pos(
        scene.rigid_solver.get_base_links_pos(screwdriver_link),
        scene.rigid_solver.get_base_links_quat(screwdriver_link),
        ToolInfo().TOOLS_GRIPPER_POS[ToolsEnum.SCREWDRIVER.value].unsqueeze(0),
    )
    hand_pos = franka_hand_link.get_pos().to(actions.device)

    dist = torch.full((scene.n_envs,), fill_value=-1.0, device=actions.device)
    # pick up or drop tool
    if pick_up_tool_mask.any():
        # TODO logic for more tools.
        dist = torch.norm(hand_pos.squeeze(1) - grip_pos.squeeze(1), dim=-1)
        # TODO: not looking for closest tool, but should look for closest only.
        # Use ToolInfo property-based API for grip distance
        from repairs_components.logic.tools.tool import ToolsEnum
        from repairs_components.logic.tools.tools_state import ToolInfo

        required_dist = (
            ToolInfo()
            .TOOLS_DIST_FROM_GRIP_LINK[ToolsEnum.SCREWDRIVER.value]
            .to(actions.device)
        )
        pick_up_tool_mask_and_close = pick_up_tool_mask & (dist < required_dist)

        if pick_up_tool_mask_and_close.any():
            env_idx = pick_up_tool_mask_and_close.nonzero().squeeze(1)
            print(f"picking up tool at env_idx: {env_idx.tolist()}")
            sim_state.tool_state.tool_ids[env_idx] = ToolsEnum.SCREWDRIVER.value
            # attach the screwdriver to the hand
            # device issues?
            from repairs_components.logic.tools.tool import attach_tool_to_arm
            from repairs_components.logic.tools.tools_state import ToolInfo

            attach_tool_to_arm(
                scene,
                screwdriver,
                franka_hand_link,
                sim_state.tool_state,
                ToolInfo(),
                env_idx,
            )
    if release_tool_mask.any():
        env_idx = release_tool_mask.nonzero().squeeze(1)
        print(f"releasing tool at env_idx: {env_idx.tolist()}")
        sim_state.tool_state.tool_ids[env_idx] = Gripper().id
        # detach the screwdriver from the hand
        detach_tool_from_arm(
            scene,
            screwdriver,
            franka_hand_link,
            sim_state.tool_state,
            env_idx,
        )

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
    # indices must be integer for tensor indexing
    picked_up_fastener_ids_long = picked_up_fastener_ids.to(torch.long)
    has_picked_up_fastener_mask = tool_state.screwdriver_tc.has_picked_up_fastener

    screw_in_desired_mask, screw_out_desired_mask = receive_screw_in_action(
        actions
    )  # [B]

    # connection status masks for the picked-up fastener in each env
    fastener_disconnected_mask = (
        physical_state.fasteners_attached_to_body == -1
    )  # [B, F, 2]
    picked_mask = fastener_disconnected_mask[
        torch.arange(scene.n_envs), picked_up_fastener_ids_long
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
            torch.arange(scene.n_envs), picked_up_fastener_ids_long
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
        # if no holes metadata exists, cannot insert; skip safely
        assert sim_info.physical_info.part_hole_batch.numel() > 0, (
            "part_hole_batch metadata was not set."
        )
        env_mask = has_fastener_and_desired_in.reshape(-1)
        env_ids = env_mask.nonzero(as_tuple=False).squeeze(1)

        # Process each env independently since check_fastener_possible_insertion currently takes a single fastener_id
        for env_id in env_ids.tolist():
            fid = picked_up_fastener_ids_long[env_id]
            # Determine part to ignore (if already connected in this env)
            fastener_body_indices = physical_state.fasteners_attached_to_body[
                env_id, fid
            ]
            ignore_part_idx_env = fastener_body_indices.max().view(1)

            part_idx_tensor, hole_idx_tensor = check_fastener_possible_insertion(
                scene,
                sim_state,
                sim_info,
                env_ids=torch.tensor([env_id], dtype=torch.long, device=actions.device),
                ignore_part_idx=ignore_part_idx_env.to(actions.device),
            )
            hole_id = hole_idx_tensor[0]
            part_id = part_idx_tensor[0]
            if part_id < 0 or hole_id < 0:
                continue  # nothing to insert for this env

            # Already-inserted hole id (or -1) for this env/fastener
            already_inserted_hole_id_env = physical_state.fasteners_attached_to_hole[
                env_id, fid
            ].max()

            # Attach using ID-based API (tensors of shape [1])
            attach_picked_up_fastener_to_part(
                scene,
                sim_state,
                sim_info,
                hole_id,
                already_inserted_hole_id_env,
                env_id,
            )

            # Mark attachment in graph state: write body index into first free slot
            sim_state.physical_state = connect_fastener_to_one_body(
                sim_state.physical_state,
                sim_info.physical_info,
                fid,
                hole_id,
                torch.tensor([env_id], device=actions.device),
            )

    if has_fastener_and_desired_out.any():
        env_out_ids = (
            has_fastener_and_desired_out.reshape(-1)
            .nonzero(as_tuple=False)
            .squeeze(1)
            .tolist()
        )
        for env_id in env_out_ids:
            # problem is that this isn't easily batchable as I need to get different fastener entities on different batches.
            fastener_id = tool_state.screwdriver_tc.picked_up_fastener_id[env_id]
            assert fastener_id.item() >= 0
            # for every non -1, delete weld constraints by IDs
            fastener_body_indices = physical_state.fasteners_attached_to_body[
                env_id, fastener_id
            ]
            connected_mask = fastener_body_indices != -1
            if connected_mask.any():
                part_ids = fastener_body_indices[connected_mask]
                envs_idx = torch.full(
                    (part_ids.shape[0],),
                    env_id,
                    dtype=torch.long,
                    device=actions.device,
                )
                sim_state = detach_fastener_from_part(
                    scene, sim_state, sim_info, part_ids, envs_idx
                )

            # update graph-state to reflect detachment from all bodies/holes in this env
            sim_state.physical_state.fasteners_attached_to_body[
                env_id, fastener_id, :
            ] = -1
            sim_state.physical_state.fasteners_attached_to_hole[
                env_id, fastener_id, :
            ] = -1

            attach_fastener_to_screwdriver(  # reattach the fastener to hand.
                scene, sim_state, sim_info, fastener_id, env_id
            )

    print(f"has_fastener_and_desired_in {has_fastener_and_desired_in}")
    print(f"has_fastener_and_desired_out {has_fastener_and_desired_out}")

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
        desired_pick_up_indices = torch.nonzero(pick_up_mask).squeeze(1)

        # update screwdriver gripper position
        screwdriver_pos = tool_state.picked_up_pos[desired_pick_up_indices]
        screwdriver_quat = tool_state.picked_up_quat[desired_pick_up_indices]
        rel_connector_pos = (
            Screwdriver.fastener_connector_pos_relative_to_center().expand(
                screwdriver_pos.shape[0], -1
            )
        )  # [B,3]
        screwdriver_gripper_pos = get_connector_pos(
            screwdriver_pos, screwdriver_quat, rel_connector_pos
        ).unsqueeze(1)  # [B,1,3] so it can be broadcasted to fastener pos [B,F,3]
        # calculate positions of fasteners close to screwdriver_gripper_pos
        fastener_positions = sim_state.physical_state.fasteners_pos[
            desired_pick_up_indices
        ]
        # Distances per env, per fastener -> [B, F]
        dists = torch.norm(fastener_positions - screwdriver_gripper_pos, dim=-1)
        # Reduce over fasteners -> values [B], indices [B]
        per_env_min_vals, per_env_min_ids = dists.min(dim=1)
        closest_fastener_id = per_env_min_ids
        close_enough = per_env_min_vals < max_pick_up_threshold

        print("fastener pick up mask", pick_up_mask, "Close enough:", close_enough)

        valid_mask = close_enough
        for i, env_id in enumerate(desired_pick_up_indices[valid_mask].tolist()):
            fid = closest_fastener_id[valid_mask][i]
            attach_fastener_to_screwdriver(
                scene,
                sim_state,
                sim_info,
                torch.tensor([fid.item()], dtype=torch.long, device=actions.device),
                torch.tensor([env_id], dtype=torch.long, device=actions.device),
            )

    ##debug
    assert (
        tool_state.screwdriver_tc.has_picked_up_fastener[
            screwdriver_with_fastener_mask.nonzero().squeeze(1).tolist()
        ]
    ).all(), (
        "screwdriver_with_fastener_mask should be true only for those with fastener picked up"
    )
    # // # somehow this fails.

    if release_mask.any():
        desired_release_indices = torch.nonzero(release_mask).squeeze(1)
        for env_id in desired_release_indices.tolist():
            _fid = tool_state.screwdriver_tc.picked_up_fastener_id[env_id]
            fastener_id = int(_fid.item()) if torch.is_tensor(_fid) else int(_fid)
            if fastener_id < 0:
                # FIXME: this should never happen, and yet somehow it does. I've tried debugging many times at this point.
                continue
            detach_fastener_from_screwdriver(
                scene,
                sim_state,
                sim_info,
                torch.tensor([env_id], dtype=torch.long, device=actions.device),
            )
        print("fastener release mask", release_mask)

    return sim_state
