import pytest
import torch
from repairs_components.geometry.fasteners import Fastener
from repairs_components.processing.translation import update_hole_locs
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.electronics.electronics_state import ElectronicsState

import repairs_sim_step as step_mod
from repairs_sim_step import (
    step_repairs,
    step_screw_in_or_out,
    step_pick_up_release_tool,
    step_fastener_pick_up_release,
)
import genesis as gs
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import attach_tool_to_arm
from repairs_components.logic.tools.tool import ToolsEnum, detach_tool_from_arm
from repairs_components.logic.tools.gripper import Gripper
from genesis.engine.entities import RigidEntity
from genesis.engine.entities.rigid_entity import RigidLink
import numpy as np
from global_test_config import init_gs


@pytest.fixture
def fastener():
    return Fastener(
        initial_hole_id_a=0,
        initial_hole_id_b=2,
        length=15.0,
        diameter=5.0,
        b_depth=5.0,
        head_diameter=7.5,
        head_height=3.0,
        thread_pitch=0.5,
    )


@pytest.fixture
def bd_geometry(fastener):
    return fastener.bd_geometry()


@pytest.fixture(scope="module")
def _built_scene_with_fastener_screwdriver_and_two_parts(init_gs):
    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    screwdriver_cube_pos = (0.2, 0.0, 0.02)
    fastener_cube_pos = (0.0, 0.2, 0.02)
    part_with_holes_1_pos = (-0.2, 0.0, 0.02)
    part_with_holes_2_pos = (0.0, -0.2, 0.02)
    # "tool cube" and "fastener cube" as stubs for real geometry. Functionally the same.
    screwdriver_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=screwdriver_cube_pos,
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),  # red
    )
    fastener_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=fastener_cube_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),  # green
    )
    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    part_with_holes_1: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=part_with_holes_1_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)),  # blue
    )
    part_with_holes_2: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=part_with_holes_2_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 1)),  # cyan
    )

    camera = scene.add_camera(
        pos=(1.0, 2.5, 3.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
    )
    scene.build(n_envs=1)
    camera.start_recording()

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    entities = {
        "screwdriver@control": screwdriver_cube,
        "0@fastener": fastener_cube,
        "franka@control": franka,
        "part_with_holes_1@solid": part_with_holes_1,
        "part_with_holes_2@solid": part_with_holes_2,
    }
    return scene, entities


@pytest.fixture
def holes_for_two_parts():
    # holes on top of the cubes.
    num_holes_first_part = 2
    num_holes_second_part = 2
    total_holes = num_holes_first_part + num_holes_second_part
    starting_hole_positions = torch.tensor(
        [[0.2, 0.0, 0.06], [0.2, 0.0, 0.0], [0, 0.2, 0.06], [0.2, 0.0, 0.0]]
    )  # [H, 3]
    starting_hole_quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )  # [H, 4]
    hole_indices_batch = torch.tensor(
        [0] * num_holes_first_part + [1] * num_holes_second_part
    )  # [H]
    assert (
        total_holes
        == len(hole_indices_batch)
        == starting_hole_quats.shape[0]
        == starting_hole_positions.shape[0]
    ), "Test setup failure - hole count mismatch"
    return starting_hole_positions, starting_hole_quats, hole_indices_batch


@pytest.fixture(scope="function")
def fresh_scene_with_fastener_screwdriver_and_two_parts(
    _built_scene_with_fastener_screwdriver_and_two_parts,
    holes_for_two_parts,
):
    "Reset the scene."
    scene, entities = _built_scene_with_fastener_screwdriver_and_two_parts
    scene.reset()
    hole_positions, hole_quats, hole_indices_batch = holes_for_two_parts

    fastener_data = Fastener(initial_hole_id_a=None)
    # populate current sim state
    repairs_sim_state = RepairsSimState(1)
    physical_state = repairs_sim_state.physical_state[0]

    part_with_holes_1_pos = entities["part_with_holes_1@solid"].get_pos(0).squeeze(0)
    part_with_holes_2_pos = entities["part_with_holes_2@solid"].get_pos(0).squeeze(0)

    physical_state.register_fastener(fastener_data)
    physical_state.register_body(
        "part_with_holes_1@solid",
        position=part_with_holes_1_pos,
        rotation=entities["part_with_holes_1@solid"].get_quat(0).squeeze(0),
        rot_as_quat=True,
        _expect_unnormalized_coordinates=False,
    )
    physical_state.register_body(
        "part_with_holes_2@solid",
        position=part_with_holes_2_pos,
        rotation=entities["part_with_holes_2@solid"].get_quat(0).squeeze(0),
        rot_as_quat=True,
        _expect_unnormalized_coordinates=False,
    )
    # expected shape
    physical_state.hole_positions = hole_positions
    physical_state.hole_quats = hole_quats
    physical_state.part_hole_batch = hole_indices_batch

    # populate desired state
    desired_sim_state = RepairsSimState(1)
    desired_physical_state = desired_sim_state.physical_state[0]
    desired_part_with_holes_1_pos = part_with_holes_1_pos + torch.tensor(
        [0.0, 0.0, 0.5]
    )  # elevate 0.5m higher
    desired_part_with_holes_2_pos = part_with_holes_2_pos + torch.tensor(
        [0.0, 0.0, 0.5]
    )
    desired_physical_state.register_fastener(fastener_data)
    desired_physical_state.register_body(
        "part_with_holes_1@solid",
        position=desired_part_with_holes_1_pos,
        rotation=(0, 0, 0, 1),
        rot_as_quat=True,
        _expect_unnormalized_coordinates=False,
    )
    desired_physical_state.register_body(
        "part_with_holes_2@solid",
        position=desired_part_with_holes_2_pos,
        rotation=(0, 0, 0, 1),
        rot_as_quat=True,
        _expect_unnormalized_coordinates=False,
    )

    return scene, entities, repairs_sim_state, desired_sim_state


# -----------------------
# === screw_in_or_out ===
# -----------------------


def test_step_screw_in_or_out_screws_in_and_unscrews_from_one_part(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
):
    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )
    fastener_entity = gs_entities["0@fastener"]
    physical_state = repairs_sim_state.physical_state[0]
    graph_device = physical_state.fasteners_attached_to.device

    screwdriver = Screwdriver(
        picked_up_fastener_name="0@fastener",
        picked_up_fastener_tip_position=physical_state.hole_positions[0],
    )  # mark as moved to closest hole.
    repairs_sim_state.tool_state[0].current_tool = screwdriver
    connected_part_id = physical_state.body_indices["part_with_holes_1@solid"]

    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.99  # I beleive 8 was for screw in
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        physical_state.fasteners_attached_to
        == torch.tensor(
            [[connected_part_id, -1]],
            device=graph_device,
        )
    ).all(), "Fastener is expected to be marked as attached to a part"
    assert (
        fastener_entity.get_pos(0)
        == repairs_sim_state.physical_state[0].hole_positions[0]
    ).all(), "Fastener is expected to move to hole position"
    assert (
        fastener_entity.get_quat(0) == repairs_sim_state.physical_state[0].hole_quats[0]
    ).all(), "Fastener is expected to move to hole quat"
    # now, unscrew it. To make sim cheaper, simply assert that it runs.
    assert screwdriver.has_picked_up_fastener, (
        "Screwdriver is expected to not have released a fastener after screw in."
    )
    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.01
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        physical_state.fasteners_attached_to
        == torch.tensor(
            [[-1, -1]],
            device=graph_device,
        )
    ).all(), "Fastener is expected to be marked as detached from a part"
    # TODO: when genesis implements constraints checks, assert that fastener is not attached to a part
    assert screwdriver.has_picked_up_fastener, (
        "Screwdriver is expected to not have released a fastener after screw out."
    )


def test_step_screw_in_or_out_screws_in_and_unscrews_from_two_parts(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
):
    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )
    fastener_entity = gs_entities["0@fastener"]
    physical_state = repairs_sim_state.physical_state[0]
    graph_device = physical_state.fasteners_attached_to.device
    screwdriver = Screwdriver(
        picked_up_fastener_tip_position=physical_state.hole_positions[0],
        picked_up_fastener_name="0@fastener",
    )
    repairs_sim_state.tool_state[0].current_tool = screwdriver
    connected_part_1_id = physical_state.body_indices["part_with_holes_1@solid"]
    # Now attach the fastener to the second part as well (screw through both parts)
    connected_part_2_id = physical_state.body_indices["part_with_holes_2@solid"]

    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.99  # I beleive 8 was for screw in
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        physical_state.fasteners_attached_to
        == torch.tensor([[connected_part_1_id, -1]], device=graph_device)
    ).all(), "Fastener is expected to be marked as attached to a part"
    assert (fastener_entity.get_pos(0) == physical_state.hole_positions[0]).all(), (
        "Fastener is expected to move to hole position"
    )
    assert (fastener_entity.get_quat(0) == physical_state.hole_quats[0]).all(), (
        "Fastener is expected to move to hole quat"
    )
    assert (
        repairs_sim_state.tool_state[0].current_tool.picked_up_fastener_name
        == "0@fastener"
    ), "Fastener is not expected to be released after screwing in"

    # mark as moved to closest marked hole on second part
    hole_pos_part_2 = physical_state.hole_positions[2]
    screwdriver.picked_up_fastener_tip_position = hole_pos_part_2
    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.99  # screw in to attach second part
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        repairs_sim_state.physical_state[0].fasteners_attached_to
        == torch.tensor(
            [[connected_part_1_id, connected_part_2_id]], device=graph_device
        )
    ).all(), "Fastener is expected to be marked as attached to both parts"
    assert (fastener_entity.get_pos(0) == hole_pos_part_2).all(), (
        "Fastener is expected to move to hole position on second part"
    )
    assert (fastener_entity.get_quat(0) == physical_state.hole_quats[2]).all(), (
        "Fastener is expected to move to hole quat on second part"
    )
    # now, unscrew it to detach from both parts. To make sim cheaper, simply assert that it runs.
    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.01
    # screw in from both parts #note: it is desirable to be able to unscrew from only one part.
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        repairs_sim_state.physical_state[0].fasteners_attached_to
        == torch.tensor([[-1, -1]], device=graph_device)
    ).all(), "Fastener is expected to be marked as detached from both parts"
    # TODO: when genesis implements constraints checks, assert that fastener is not attached to a part


def test_step_screw_in_or_out_does_not_screws_in_at_one_part_inserted_and_large_angle(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
):
    """When one part is already connected, and the angle is too large, it should not screw in."""
    from repairs_components.processing.geom_utils import (
        quat_multiply,
        euler_deg_to_quat_wxyz,
    )

    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )
    fastener_entity = gs_entities["0@fastener"]
    physical_state = repairs_sim_state.physical_state[0]
    graph_device = physical_state.fasteners_attached_to.device
    repairs_sim_state.tool_state[0].current_tool = Screwdriver(
        picked_up_fastener_name="0@fastener",
        picked_up_fastener_tip_position=physical_state.hole_positions[0],
    )

    connected_part_1_id = physical_state.body_indices["part_with_holes_1@solid"]

    # First, attach fastener to the first part
    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.99  # screw in
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )
    assert (
        repairs_sim_state.physical_state[0].fasteners_attached_to
        == torch.tensor([[connected_part_1_id, -1]], device=graph_device)
    ).all(), "Fastener is expected to be marked as attached to first part"

    # Now try to attach to second part with a large angle difference (90 degrees rotation)

    # Create a quaternion with 90-degree rotation around Z-axis from the original
    original_quat = physical_state.hole_quats[2]
    # Create 90-degree rotation around Z-axis
    large_angle_rotation = euler_deg_to_quat_wxyz(torch.tensor([0, 0, 90]))
    # Convert to [w,x,y,z] format
    large_angle_rotation = torch.tensor(
        [
            large_angle_rotation[3],
            large_angle_rotation[0],
            large_angle_rotation[1],
            large_angle_rotation[2],
        ]
    )

    # Apply the large angle rotation to the original quaternion
    large_angle_quat = quat_multiply(
        original_quat.unsqueeze(0), large_angle_rotation.unsqueeze(0)
    ).squeeze(0)

    repairs_sim_state.tool_state[
        0
    ].current_tool.picked_up_fastener_tip_position = physical_state.hole_positions[
        2
    ]  # move it to closest marked hole on second part
    repairs_sim_state.tool_state[
        0
    ].current_tool.picked_up_fastener_tip_quat = large_angle_quat

    # Store the state before attempting to screw in
    fasteners_attached_before = physical_state.fasteners_attached_to.clone()
    fastener_pos_before = fastener_entity.get_pos(0).clone()
    fastener_quat_before = fastener_entity.get_quat(0).clone()

    actions = torch.zeros((1, 10))
    actions[:, 8] = 0.99  # attempt to screw in to attach second part
    repairs_sim_state = step_screw_in_or_out(
        scene, gs_entities, repairs_sim_state, actions
    )

    # Assert that the fastener attachment state did not change (still only attached to first part)
    assert torch.equal(
        repairs_sim_state.physical_state[0].fasteners_attached_to,
        fasteners_attached_before,
    ), "Fastener attachment should not change when angle is too large"

    # Assert that fastener position and orientation did not change
    assert torch.allclose(fastener_entity.get_pos(0), fastener_pos_before, atol=1e-6), (
        "Fastener position should not change when angle is too large"
    )
    assert torch.allclose(
        fastener_entity.get_quat(0), fastener_quat_before, atol=1e-6
    ), "Fastener orientation should not change when angle is too large"


# ---------------------------------
# === step_pick_up_release_tool ===
# ---------------------------------


def test_step_pick_up_release_tool_picks_up_or_releases_tool_when_in_proximity(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
):
    """
    Test that step_pick_up_release_tool picks up and releases tool.
    Test:
    1. When in proximity (set to 0.75), set pick up tool action to 1.0 and test that tool is picked up.
    2. When in proximity, set release tool action to 0.0 and test that tool is released.
    """
    from repairs_components.processing.translation import get_connector_pos

    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )

    screwdriver_entity: RigidEntity = gs_entities["screwdriver@control"]
    franka_entity: RigidEntity = gs_entities["franka@control"]
    franka_hand: RigidLink = franka_entity.get_link("hand")

    # Initially, tool state should have Gripper (no tool picked up)
    assert isinstance(repairs_sim_state.tool_state[0].current_tool, Gripper), (
        f"Initially should have Gripper (no tool), got {repairs_sim_state.tool_state[0].current_tool}"
    )

    # Position the franka hand close to the screwdriver grip position
    # Get the screwdriver grip position
    grip_pos = get_connector_pos(
        screwdriver_entity.get_pos(),
        screwdriver_entity.get_quat(),
        Screwdriver.tool_grip_position().unsqueeze(0),
    )

    # Move franka hand to be within the required distance from grip
    required_dist = Screwdriver.dist_from_grip_link()
    # Position franka hand slightly closer than required distance
    close_distance = max(
        required_dist * 0.8, 0.3
    )  # 80% of required distance. If the dist is too large (for debug), set it to 0.3

    # Set franka position to be close to grip position
    target_hand_pos = grip_pos.squeeze(1) + torch.tensor([[close_distance, 0.0, 0.0]])
    ik_point = franka_entity.inverse_kinematics(
        franka_hand, pos=target_hand_pos, quat=torch.tensor([[0, 1, 0, 0]])
    )
    franka_entity.set_dofs_position(ik_point)

    # Store initial screwdriver position before pickup
    initial_screwdriver_pos = screwdriver_entity.get_pos(0).clone()
    initial_screwdriver_quat = screwdriver_entity.get_quat(0).clone()

    # Test 1: Pick up tool when in proximity
    actions = torch.zeros((1, 10))  # 9 actions total
    actions[:, 9] = 1.0  # Set pick up tool action to 1.0 (> 0.75 threshold)

    repairs_sim_state = step_pick_up_release_tool(
        scene, gs_entities, repairs_sim_state, actions
    )

    # Assert that tool was picked up (should now have Screwdriver)
    assert isinstance(repairs_sim_state.tool_state[0].current_tool, Screwdriver), (
        "Tool should be picked up when in proximity and action > 0.75"
    )

    # Assert that screwdriver position changed (it should be snapped to franka arm)
    current_screwdriver_pos = screwdriver_entity.get_pos(0)
    current_screwdriver_quat = screwdriver_entity.get_quat(0)

    # The screwdriver should now be positioned at the calculated grip position relative to franka hand
    franka_hand = franka_entity.get_link("hand")
    expected_tool_pos = get_connector_pos(
        franka_hand.get_pos(torch.tensor([0])).squeeze(1),
        franka_hand.get_quat(torch.tensor([0])).squeeze(1),
        -Screwdriver()
        .tool_grip_position()
        .unsqueeze(0),  # minus because from arm to tool
    )

    assert torch.allclose(current_screwdriver_pos, expected_tool_pos, atol=1e-3), (
        "Screwdriver should be positioned at the correct grip position relative to franka hand"
    )

    # Test 2: Release tool when in proximity
    actions = torch.zeros((1, 10))
    actions[:, 9] = 0.0  # Set release tool action to 0.0 (< 0.25 threshold)

    repairs_sim_state = step_pick_up_release_tool(
        scene, gs_entities, repairs_sim_state, actions
    )

    # Assert that tool was released (should now have Gripper again)
    assert isinstance(repairs_sim_state.tool_state[0].current_tool, Gripper), (
        "Tool should be released when action < 0.25"
    )


def test_step_pick_up_release_tool_does_not_pick_up_or_release_tool_when_not_in_proximity():
    """
    Test that step_pick_up_release_tool does not pick up or release tool.
    Test:
    1. When not in proximity (set proximity to 0.01), set pick up tool action to 1.0 and test that tool is not picked up.
    2. When not in proximity, set release tool action to 0.0 and test that tool is not released.
    """


# ---------------------------------
# === step_fastener_pick_up_release ===
# ---------------------------------
def test_step_fastener_pick_up_release_picks_up_and_releases_fastener(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
):
    """
    Test that step_fastener_pick_up_release picks up and releases fastener.
    Test:
    1. When in proximity (set proximity to 0.75), set pick up fastener action to 1.0 and test that fastener is picked up:
     a. Test that pos of the fastener is equal to tool grip pos and quat is equal to tool quat.
     b. Tool has has_picked_up_fastener set to True.
     c. Tool has picked_up_fastener_name set to fastener name.
     d. Tool has picked_up_fastener_tip_position set to fastener tip position.
    2. When in proximity, set release fastener action to 0.0 and test that fastener is released:
     a. Tool has has_picked_up_fastener set to False.
     b. Tool has picked_up_fastener_name set to None.
     c. Tool has picked_up_fastener_tip_position set to None.
    """
    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )
    fastener_entity = gs_entities["0@fastener"]
    screwdriver_entity = gs_entities["screwdriver@control"]

    # Set up screwdriver as current tool
    repairs_sim_state.tool_state[0].current_tool = Screwdriver()

    # Position fastener close to screwdriver (within proximity threshold of 0.75)
    fastener_initial_pos = torch.tensor([0.2, 0.0, 0.02])  # Close to screwdriver
    fastener_entity.set_pos(fastener_initial_pos.unsqueeze(0))
    fastener_initial_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    fastener_entity.set_quat(fastener_initial_quat.unsqueeze(0))

    # Update fastener position in sim state
    repairs_sim_state.physical_state[0].fasteners_pos = torch.tensor(
        [[0.2, 0.0, 0.02]]
    )

    # Test 1: Pick up fastener (action[7] = 1.0 for pick up)
    actions = torch.zeros((1, 10))
    actions[:, 7] = 1.0  # Pick up action

    # Call step_fastener_pick_up_release with proximity threshold of 0.75
    updated_sim_state = step_fastener_pick_up_release(
        scene, gs_entities, repairs_sim_state, actions, max_pick_up_threshold=0.75
    )

    # Test 1a: Check that fastener is moved to tool grip position
    screwdriver_pos = screwdriver_entity.get_pos(torch.tensor([0]))
    screwdriver_quat = screwdriver_entity.get_quat(torch.tensor([0]))

    # Test 1b: Tool has has_picked_up_fastener set to True
    assert (
        updated_sim_state.tool_state[0].current_tool.has_picked_up_fastener == True
    ), "Tool should have has_picked_up_fastener set to True after picking up"

    # Test 1c: Tool has picked_up_fastener_name set to fastener name
    assert (
        updated_sim_state.tool_state[0].current_tool.picked_up_fastener_name
        == "0@fastener"
    ), "Tool should have picked_up_fastener_name set to '0@fastener'"

    # Test 1d: Tool has picked_up_fastener_tip_position set to fastener tip position
    assert (
        updated_sim_state.tool_state[0].current_tool.picked_up_fastener_tip_position
        is not None
    ), "Tool should have picked_up_fastener_tip_position set"

    # Test 2: Release fastener (action[7] = 0.0 for release)
    actions = torch.zeros((1, 10))
    actions[:, 7] = 0.0  # Release action

    updated_sim_state = step_fastener_pick_up_release(
        scene, gs_entities, updated_sim_state, actions, max_pick_up_threshold=0.75
    )

    # Test 2a: Tool has has_picked_up_fastener set to False
    assert (
        updated_sim_state.tool_state[0].current_tool.has_picked_up_fastener == False
    ), "Tool should have has_picked_up_fastener set to False after releasing"

    # Test 2b: Tool has picked_up_fastener_name set to None
    assert (
        updated_sim_state.tool_state[0].current_tool.picked_up_fastener_name is None
    ), "Tool should have picked_up_fastener_name set to None after releasing"

    # Test 2c: Tool has picked_up_fastener_tip_position set to None
    assert (
        updated_sim_state.tool_state[0].current_tool.picked_up_fastener_tip_position
        is None
    ), "Tool should have picked_up_fastener_tip_position set to None after releasing"


def test_step_fastener_pick_up_release_does_not_pick_up_fastener_when_not_in_proximity():
    """
    Test that step_fastener_pick_up_release does not pick up fastener when not in proximity.
    Test:
    1. When not in proximity (set proximity to 0.01), set pick up fastener action to 1.0 and test that fastener is not picked up.
     a. No attributes on screwdriver should be changed
     b. pos and quat of fastener should remain equal.
    """


def test_step_fastener_pick_up_release_does_not_pick_up_fastener_when_in_one_hand():
    """
    Test that step_fastener_pick_up_release does not pick up fastener when there is already a fastener attached to a screwdriver.
    Test:
    1. When there is already a fastener attached to a screwdriver, set pick up fastener action to 1.0 and test that fastener is not moved and the initial fastener is still attached to the screwdriver.
    """


# ---------------
# === success ===
# ---------------


def test_all_bodies_moved_to_desired_pos_results_in_success(
    fresh_scene_with_fastener_screwdriver_and_two_parts,
    holes_for_two_parts,
):
    """
    Test that all bodies moved to desired position results in success.
    Test:
    1. Move all bodies to desired position and test that success is returned.
    """
    scene, gs_entities, repairs_sim_state, desired_sim_state = (
        fresh_scene_with_fastener_screwdriver_and_two_parts
    )
    untranslated_hole_positions, untranslated_hole_quats, hole_indices_batch = (
        holes_for_two_parts
    )
    for body_name, body_idx in desired_sim_state.physical_state[0].body_indices.items():
        gs_entities[body_name].set_pos(
            desired_sim_state.physical_state[0].position[body_idx].unsqueeze(0)
        )
        gs_entities[body_name].set_quat(
            desired_sim_state.physical_state[0].quat[body_idx].unsqueeze(0)
        )

    success, total_diff_left, _, _ = step_repairs(
        scene=scene,
        actions=torch.zeros((1, 10)),
        gs_entities=gs_entities,
        current_sim_state=repairs_sim_state,
        desired_state=desired_sim_state,
        starting_hole_positions=untranslated_hole_positions,
        starting_hole_quats=untranslated_hole_quats,
        hole_depth=hole_depths,
        part_hole_batch=hole_indices_batch,
    )
    assert success == True, (
        "All bodies moved to desired position should result in success"
    )
    assert total_diff_left == 0, (
        "All bodies moved to desired position should result in total_diff_left == 0"
    )
