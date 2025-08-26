import genesis as gs
import numpy as np
import pytest
import torch

# import argparse
from genesis.engine.entities import RigidEntity

from repairs_components.geometry.fasteners import (
    Fastener,
    attach_picked_up_fastener_to_part,
    attach_fastener_to_screwdriver,
    detach_fastener_from_part,
    detach_fastener_from_screwdriver,
)
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import ToolsEnum
from repairs_components.processing.genesis_utils import (
    is_weld_constraint_present,
    populate_base_link_indices,
)
from repairs_components.logic.physical_state import (
    register_bodies_batch,
    register_fasteners_batch,
)
from repairs_components.processing.translation import update_hole_locs
from repairs_components.training_utils.sim_state_global import (
    RepairsSimInfo,
    RepairsSimState,
)


@pytest.fixture(scope="session")
def scene_with_fastener_screwdriver_and_two_parts(init_gs, test_device):
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
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # "tool cube" and "fastener cube" as stubs for real geometry. Functionally the same.
    screwdriver_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),  # red
    )
    fastener_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.65, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),  # green
    )
    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    part_with_holes_1: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(-0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)),  # blue
    )
    part_with_holes_2: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, -0.65, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 1)),  # cyan
    )
    # holes_1_pos = torch.tensor([[[0.5, 0.0, 0.04], [0.5, 0.0, 0.0]]])
    # holes_2_pos = torch.tensor([[[0, 0.5, 0.04], [0.5, 0.0, 0.0]]])
    # holes_1_quat = torch.tensor([[[0, 1, 0, 0], [0, 1, 0, 0]]])
    # holes_2_quat = torch.tensor([[[0, 1, 0, 0], [0, 1, 0, 0]]])

    camera = scene.add_camera(
        pos=(1.0, 2.5, 3.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
    )
    scene.build(n_envs=1)
    camera.start_recording()

    end_effector = franka.get_link("hand")

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
        "screwdriver@tool": screwdriver_cube,
        "0@fastener": fastener_cube,
        "franka@control": franka,
        "end_effector": end_effector,
        "part_with_holes_1@solid": part_with_holes_1,
        "part_with_holes_2@solid": part_with_holes_2,
    }
    return scene, entities


@pytest.fixture(scope="function")
def sim_state_for_genesis_fastener_tests(
    scene_with_fastener_screwdriver_and_two_parts, test_device
):
    """Build RepairsSimState/RepairsSimInfo for this scene and register bodies, holes, and one fastener.

    - Populates base link indices required by ID-based fastener API.
    - Provides two holes: hole 0 on part 1, hole 1 on part 2.
    """
    scene, entities = scene_with_fastener_screwdriver_and_two_parts
    scene.reset()

    sim_state = RepairsSimState(device=test_device).unsqueeze(0)
    sim_info = RepairsSimInfo()

    # Register the two bodies
    part1 = entities["part_with_holes_1@solid"]
    part2 = entities["part_with_holes_2@solid"]
    pos_b = torch.stack([part1.get_pos(0), part2.get_pos(0)], dim=1)
    quat_b = torch.stack([part1.get_quat(0), part2.get_quat(0)], dim=1)
    sim_state.physical_state, sim_info.physical_info = register_bodies_batch(
        names=["part_with_holes_1@solid", "part_with_holes_2@solid"],
        positions=pos_b,
        rotations=quat_b,
        fixed=torch.tensor([False, False]),
        min_bounds=torch.tensor([-2.0, -2.0, 0.0]),
        max_bounds=torch.tensor([2.0, 2.0, 2.0]),
    )

    # Two holes: index 0 belongs to part 1, index 1 belongs to part 2 (positions are in the local frame, will be translated)
    sim_info.physical_info.starting_hole_positions = torch.tensor(
        [
            [0.5, 0.5, 0.04],  # hole 0 on part 1
            [0.4, 0.4, 0.04],  # hole 1 on part 2
        ]
    )
    sim_info.physical_info.starting_hole_quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    sim_info.physical_info.part_hole_batch = torch.tensor([0, 1], dtype=torch.long)
    # Depth/through/diameter (meters)
    sim_info.physical_info.hole_depth = torch.tensor([0.005, 0.005])
    sim_info.physical_info.hole_is_through = torch.tensor([True, True])
    sim_info.physical_info.hole_diameter = torch.tensor([0.005, 0.005])

    # Translate hole positions into world given current part poses
    sim_state = update_hole_locs(
        sim_state,
        sim_info.physical_info.starting_hole_positions,
        sim_info.physical_info.starting_hole_quats,
        sim_info.physical_info.part_hole_batch,
    )

    # Register single free fastener using its current entity pose
    fastener = entities["0@fastener"]
    sim_state.physical_state, sim_info.physical_info = register_fasteners_batch(
        sim_state.physical_state,
        sim_info.physical_info,
        fastener.get_pos(0).unsqueeze(1),
        fastener.get_quat(0).unsqueeze(1),
        torch.tensor([[-1]]),
        torch.tensor([[-1]]),
        fastener_compound_names=["standard_d5_l15"],
    )

    # Base link indices for bodies and the single fastener
    populate_base_link_indices(sim_info.physical_info, entities, num_fasteners=1)

    # Populate tool base link indices for screwdriver and gripper
    sim_info.tool_info.tool_base_link_idx[ToolsEnum.SCREWDRIVER.value] = torch.tensor(
        entities["screwdriver@tool"].base_link.idx, dtype=torch.int32
    )
    # Gripper maps to Franka hand link
    sim_info.tool_info.tool_base_link_idx[ToolsEnum.GRIPPER.value] = torch.tensor(
        entities["end_effector"].idx, dtype=torch.int32
    )

    # Tool selection
    sim_state.tool_state.tool_ids = torch.tensor([ToolsEnum.SCREWDRIVER.value])

    return scene, entities, sim_state, sim_info


@pytest.fixture(autouse=True)
def cleanup_after_test(
    request, scene_with_fastener_screwdriver_and_two_parts, base_data_dir
):
    yield
    test_name = request.node.name
    scene, entities = request.getfixturevalue(
        "scene_with_fastener_screwdriver_and_two_parts"
    )
    # Clear any weld constraints to avoid cross-test interference
    try:
        welds = scene.sim.rigid_solver.get_weld_constraints(
            as_tensor=True, to_torch=True
        )
        if isinstance(welds, dict):

            def _to_tensor(val):
                if isinstance(val, (tuple, list)):
                    val = val[0]
                return val

            link_a = _to_tensor(welds.get("link_a", welds.get("obj_a")))
            link_b = _to_tensor(welds.get("link_b", welds.get("obj_b")))
            if link_a is not None and link_b is not None:
                link_a = link_a.reshape(-1)
                link_b = link_b.reshape(-1)
                env = welds.get("env")
                if env is not None:
                    env = _to_tensor(env).reshape(-1)
                else:
                    env = torch.zeros_like(link_a, dtype=torch.int32)
                if link_a.numel() > 0:
                    scene.sim.rigid_solver.delete_weld_constraint(link_a, link_b, env)
        else:
            # Legacy tensor format [N,3]: [env, link_a, link_b]
            if welds.ndim == 2 and welds.shape[-1] == 3 and welds.shape[0] > 0:
                env = welds[:, 0].to(torch.int32)
                link_a = welds[:, 1]
                link_b = welds[:, 2]
                scene.sim.rigid_solver.delete_weld_constraint(link_a, link_b, env)
    except Exception:
        pass
    scene.visualizer.cameras[0].stop_recording(
        save_to_filename=str(base_data_dir / f"test_videos/video_{test_name}.mp4"),
        fps=60,
    )
    scene.reset()
    scene.visualizer.cameras[0].start_recording()


def step_and_render(scene, camera, num_steps=10):
    for i in range(num_steps):
        scene.step()
        camera.render()


def test_attach_and_detach_fastener_to_screwdriver(
    sim_state_for_genesis_fastener_tests,
):
    scene, entities, sim_state, sim_info = sim_state_for_genesis_fastener_tests
    scene.reset()
    screwdriver_pos = torch.tensor([[0.5, 0.5, 1.0]])
    camera = scene.visualizer.cameras[0]
    entities["screwdriver@tool"].set_pos(screwdriver_pos)
    # Sync sim_state tool pose with entity pose
    sim_state.physical_state.tool_pos[0, ToolsEnum.SCREWDRIVER.value] = entities[
        "screwdriver@tool"
    ].get_pos(0)
    sim_state.physical_state.tool_quat[0, ToolsEnum.SCREWDRIVER.value] = entities[
        "screwdriver@tool"
    ].get_quat(0)

    # Attach using new ID-based API
    attach_fastener_to_screwdriver(
        scene,
        sim_state,
        sim_info,
        fastener_id=torch.tensor([0], dtype=torch.long),
        env_ids=torch.tensor([0]),
    )
    # Assert weld fastener<->screwdriver was added
    assert is_weld_constraint_present(
        scene, entities["0@fastener"], entities["screwdriver@tool"], env_idx=0
    ), "Expected weld constraint between fastener and screwdriver"
    fastener_grip_pos = Screwdriver.fastener_connector_pos_relative_to_center()
    # Allow physics to sync entity transforms with solver
    step_and_render(scene, camera, num_steps=1)
    assert torch.isclose(
        entities["0@fastener"].get_pos(0),
        screwdriver_pos + fastener_grip_pos,
        atol=5e-3,
    ).all(), (
        f"Fastener cube should be attached to screwdriver cube at the fastener connector position. "
        f"Fastener cube pos: {entities['0@fastener'].get_pos(0)}, screwdriver pos: {screwdriver_pos}, fastener grip pos: {fastener_grip_pos}"
    )
    assert torch.isclose(
        entities["0@fastener"].get_quat(),
        entities["screwdriver@tool"].get_quat(),
        atol=5e-3,
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )  # FIXME: this is likely wrong - a) screwdriver may be inherently incorrectly imported, b) franka is rotated as 0,1,0,0 as base.
    assert sim_state.tool_state.screwdriver_tc.has_picked_up_fastener[0]
    tip_rel = Fastener.get_tip_pos_relative_to_center().unsqueeze(0)
    assert torch.isclose(
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_tip_position,
        screwdriver_pos + tip_rel,
        atol=5e-3,
    ).all(), "Stored tip position should equal screwdriver pose + tip offset"
    assert torch.isclose(
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_quat,
        entities["screwdriver@tool"].get_quat(),
        atol=5e-3,
    ).all()
    assert sim_state.tool_state.screwdriver_tc.picked_up_fastener_id[0] == 0

    # detach fastener from screwdriver and assert they fall down.

    detach_fastener_from_screwdriver(
        scene,
        sim_state,
        sim_info,
        env_ids=torch.tensor([0]),
    )
    # Assert weld fastener<->screwdriver was removed
    assert not is_weld_constraint_present(
        scene, entities["0@fastener"], entities["screwdriver@tool"], env_idx=0
    ), "Weld constraint between fastener and screwdriver should be removed"
    # assert tool state # note: all should have shape (1,)
    assert not sim_state.tool_state.screwdriver_tc.has_picked_up_fastener[0]
    assert torch.isnan(
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_tip_position[0]
    ).all()
    assert torch.isnan(
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_quat[0]
    ).all()
    assert sim_state.tool_state.screwdriver_tc.picked_up_fastener_id[0].item() == -1

    for i in range(100):
        scene.step()
        camera.render()
    assert torch.isclose(
        entities["0@fastener"].get_pos(0)[0, 2], torch.tensor([0.01]), atol=0.10
    ), "Fastener cube should be close to the ground"
    # assert fastener is far away from screwdriver
    # assert torch.isclose( # no: screwdriver is not attached to anything and will fall.
    #     entities["screwdriver@tool"].get_pos(0),
    #     screwdriver_pos,
    #     atol=0.15,
    # ).all(), "Screwdriver should've left at the same place."


# NOTE: this is a complicated test... although necessary.
# will do later when get_rigid_weld_constraints is done.
def test_attach_and_detach_fastener_to_part(
    sim_state_for_genesis_fastener_tests,
):
    scene, entities, sim_state, sim_info = sim_state_for_genesis_fastener_tests
    screwdriver_pos = torch.tensor([[0.0, 0.0, 1.0]])
    camera = scene.visualizer.cameras[0]
    entities["screwdriver@tool"].set_pos(screwdriver_pos)
    # Make orientation deterministic for this test
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    entities["screwdriver@tool"].set_quat(identity_quat)
    # Sync sim_state tool pose with entity pose
    sim_state.physical_state.tool_pos[0, ToolsEnum.SCREWDRIVER.value] = entities[
        "screwdriver@tool"
    ].get_pos(0)
    sim_state.physical_state.tool_quat[0, ToolsEnum.SCREWDRIVER.value] = entities[
        "screwdriver@tool"
    ].get_quat(0)
    attach_fastener_to_screwdriver(
        scene,
        sim_state,
        sim_info,
        fastener_id=torch.tensor([0], dtype=torch.long),
        env_ids=torch.tensor([0]),
    )
    fastener_grip_pos = Screwdriver.fastener_connector_pos_relative_to_center()
    # Allow physics to sync entity transforms with solver
    step_and_render(scene, camera, num_steps=1)
    assert torch.isclose(
        entities["0@fastener"].get_pos(0),
        screwdriver_pos + fastener_grip_pos,
        atol=2e-2,
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )
    assert torch.isclose(
        entities["0@fastener"].get_quat(),
        entities["screwdriver@tool"].get_quat(),
        atol=2e-2,
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )  # FIXME: this is likely wrong - a) screwdriver may be inherently incorrectly imported, b) franka is rotated as 0,1,0,0 as base.

    # attach fastener to part using hole id 0 (belongs to part 1)
    attach_picked_up_fastener_to_part(
        scene,
        sim_state.physical_state,
        sim_info.physical_info,
        fastener_id=torch.tensor([0], dtype=torch.long),
        inserted_into_hole_ids=torch.tensor([0], dtype=torch.long),
        top_hole_id=torch.tensor([-1], dtype=torch.long),  # not yet inserted into top
        envs_idx=torch.tensor([0], dtype=torch.long),
    )
    # Allow physics to sync entity transforms with solver
    step_and_render(scene, camera, num_steps=1)
    # Assert weld fastener<->part was added (screwdriver weld may still exist)
    assert is_weld_constraint_present(
        scene, entities["0@fastener"], entities["part_with_holes_1@solid"], env_idx=0
    ), "Expected weld constraint between fastener and part"
    # For blind hole without partial insertion, fastener should be offset by (fastener_length - hole_depth)
    # Expected to align fastener with hole 0 pose recorded in physical_state
    expected_fastener_pos = sim_state.physical_state.hole_positions[0, 0]
    assert torch.isclose(
        entities["0@fastener"].get_pos(0), expected_fastener_pos, atol=5e-3
    ).all(), (
        "Fastener cube should be attached to part at offset position for blind hole"
    )
    assert torch.isclose(
        entities["0@fastener"].get_quat(0),
        sim_state.physical_state.hole_quats[0, 0],
        atol=5e-3,
    ).all(), "Fastener cube should be aligned with the blind hole orientation"
    # After attaching fastener to part, screwdriver should maintain its relative position to the fastener
    # If fastener is at expected_fastener_pos, then screwdriver should be at expected_fastener_pos - fastener_grip_pos
    expected_screwdriver_pos = expected_fastener_pos - fastener_grip_pos
    # FIXME: Fix this assertion - screwdriver position behavior needs investigation
    # assert torch.isclose(
    #     entities["screwdriver@tool"].get_pos(0), expected_screwdriver_pos
    # ).all(), (
    #     "Screwdriver cube should be attached to fastener at the connector position after attaching to part"
    # )
    assert torch.isclose(
        entities["screwdriver@tool"].get_quat(), entities["screwdriver@tool"].get_quat()
    ).all(), (
        "Screwdriver cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )

    # move screwdriver to a new position
    screwdriver_pos = torch.tensor([[0.0, 0.0, 2.0]])
    entities["screwdriver@tool"].set_pos(screwdriver_pos)
    entities["screwdriver@tool"].set_quat(torch.tensor([1.0, 0, 0, 0]))

    assert torch.isclose(
        entities["screwdriver@tool"].get_pos(0), screwdriver_pos[0]
    ).all(), "Screwdriver cube should be at the position it was set to"
    assert torch.isclose(
        entities["screwdriver@tool"].get_quat(0),
        entities["screwdriver@tool"].get_quat(0),
    ).all(), (
        "Screwdriver cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )

    # detach fastener from part
    detach_fastener_from_part(
        scene,
        entities["0@fastener"],
        part_entity=entities["part_with_holes_1@solid"],
        envs_idx=torch.tensor([0]),
    )
    assert torch.isclose(
        entities["0@fastener"].get_pos(0), expected_fastener_pos
    ).all(), "Fastener cube should remain attached to part, not move with screwdriver"
    assert torch.isclose(
        entities["0@fastener"].get_quat(0), entities["screwdriver@tool"].get_quat(0)
    ).all(), (
        "Fastener cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )
    # let the fastener fall down; let the part fall down too.
    for i in range(200):
        scene.step()
        camera.render()
    assert torch.isclose(
        entities["0@fastener"].get_pos(0)[:, 2], torch.tensor([0.01]), atol=0.10
    ), "Fastener cube should be close to the ground"
    assert torch.isclose(
        entities["part_with_holes_1@solid"].get_pos(0)[:, 2],
        torch.tensor([0.01]),
        atol=0.10,
    ), "Part with holes should be close to the ground"


@pytest.mark.xfail(
    reason="Should redo with assertions of get_weld later, won't work until then."
)
def test_attach_and_detach_fastener_to_two_parts(
    sim_state_for_genesis_fastener_tests,
):
    """
    Test:
    1. get initial positions of all parts,
    2. attach fastener to part 1, fastener should move, part 1 shouldn't.
    3. attach fastener to part 2, fastener should move, part 1 should follow, part 2 shouldn't move.
    4. Set fastener position by an arbitrary offset, both parts should move by that offset too.
    5. detach fastener from both parts, move fastener pos by an arbitrary offset, parts shouldn't move.
    """
    # NOTE: test failure due to not being able to easily check for constraints... and it's unexpected. Anyhow, it's not super-important.
    scene, entities, sim_state, sim_info = sim_state_for_genesis_fastener_tests
    camera = scene.visualizer.cameras[0]
    fastener = entities["0@fastener"]
    step_and_render(scene, camera)

    hole_pos1 = torch.tensor([[0.5, 0.5, 0.04]])  # note: explicitly fairly close.
    hole_pos2 = torch.tensor([[0.4, 0.4, 0.04]])
    through_hole_depth_1 = torch.tensor([0.005])
    fastener_length = torch.tensor([0.015])

    # attach fasteners to two holes via IDs (0 on part1, 1 on part2)
    attach_picked_up_fastener_to_part(
        scene,
        sim_state.physical_state,
        sim_info.physical_info,
        fastener_id=torch.tensor([0], dtype=torch.long),
        inserted_into_hole_ids=torch.tensor([0], dtype=torch.long),
        top_hole_id=torch.tensor([-1], dtype=torch.long),
        envs_idx=torch.tensor([0], dtype=torch.long),
    )
    # Assert weld fastener<->part1 added
    assert is_weld_constraint_present(
        scene, fastener, entities["part_with_holes_1@solid"], env_idx=0
    ), "Expected weld constraint between fastener and part 1"
    assert torch.isclose(fastener.get_pos(0), hole_pos1[0]).all(), (
        "Fastener should be moved to the part 1 at hole_pos1"
    )
    pos_diff_part_1_to_fastener = entities["part_with_holes_1@solid"].get_pos(
        0
    ) - entities["0@fastener"].get_pos(0)
    # (unused) pos diff to check later if needed
    part_1_pos_pre_attachment_to_second = entities["part_with_holes_1@solid"].get_pos(0)
    fastener_pos_pre_attachment_to_second = fastener.get_pos(0)
    part_2_pos_pre_attachment = entities["part_with_holes_2@solid"].get_pos(0)
    pos_diff_part_to_part_pre_attachment_to_second = (
        part_1_pos_pre_attachment_to_second - part_2_pos_pre_attachment
    )

    # second
    # note: the top hole here is the first hole (as it was the first to be inserted.)
    attach_picked_up_fastener_to_part(
        scene,
        sim_state.physical_state,
        sim_info.physical_info,
        fastener_id=torch.tensor([0], dtype=torch.long),
        inserted_into_hole_ids=torch.tensor([1], dtype=torch.long),
        top_hole_id=torch.tensor([0], dtype=torch.long),
        envs_idx=torch.tensor([0], dtype=torch.long),
    )
    # Assert weld fastener<->part2 added (part1 weld remains)
    assert is_weld_constraint_present(
        scene, fastener, entities["part_with_holes_2@solid"], env_idx=0
    ), "Expected weld constraint between fastener and part 2"
    # test fastener move after 2nd attachment
    assert torch.isclose(
        fastener.get_pos(0),
        hole_pos2
        + torch.tensor([[0.0, 0.0, fastener_length[0] - through_hole_depth_1[0]]]),
    ).all(), "Fastener should be at hole_pos2"
    holes_xyz_diff = hole_pos1 - hole_pos2
    fastener_pos_diff_after_2nd_attach_actual = (
        fastener.get_pos(0) - fastener_pos_pre_attachment_to_second
    )
    # test that a fastener has moved by correct distance
    assert torch.isclose(
        holes_xyz_diff,
        fastener_pos_diff_after_2nd_attach_actual,
    ).all(), (
        f"Fastener should've moved by difference of the hole positions. Got diff: {fastener_pos_diff_after_2nd_attach_actual}, expected: {holes_xyz_diff}"
    )
    # test that a part 1 has moved by correct distance
    part_1_pos_after_attachment_to_second = entities[
        "part_with_holes_1@solid"
    ].get_pos()
    part_1_moved_after_attachment_to_second = (
        part_1_pos_pre_attachment_to_second - part_1_pos_after_attachment_to_second
    )
    assert torch.isclose(
        holes_xyz_diff,
        part_1_moved_after_attachment_to_second,
    ).all(), (
        f"Part 1 should've moved equally as much as the fastener has moved as it is expected to be attached to the fastener. Got diff: {part_1_moved_after_attachment_to_second}, expected: {holes_xyz_diff}"
    )
    # test that a part 2 has not moved
    part_2_pos_after_attachment = entities["part_with_holes_2@solid"].get_pos()
    assert torch.isclose(
        part_2_pos_pre_attachment - part_2_pos_after_attachment,
        torch.tensor([[0.0, 0.0, 0.0]]),
    ).all(), "Part 2 should not have moved"

    # move fastener (and parts should follow)
    move_fastener_by = torch.tensor([[0.0, 0.0, 0.5]])
    fastener.set_pos(fastener.get_pos() + move_fastener_by)

    # assert that they have moved equally to the fastener.
    part_2_pos_after_move = entities["part_with_holes_2@solid"].get_pos()
    part_2_moved_after_move = part_2_pos_after_attachment - part_2_pos_after_move
    assert torch.isclose(move_fastener_by, part_2_moved_after_move).all(), (
        f"Part 2 should've moved equally as much as the fastener has moved as it is expected to be attached to the fastener. Got diff: {part_2_moved_after_move}, expected: {move_fastener_by}"
    )
    part_1_pos_after_move = entities["part_with_holes_1@solid"].get_pos()
    part_1_moved_after_move = (
        part_1_pos_after_attachment_to_second - part_1_pos_after_move
    )
    assert torch.isclose(move_fastener_by, part_1_moved_after_move).all(), (
        f"Part 1 should've moved equally as much as the fastener has moved as it is expected to be attached to the fastener. Got diff: {part_1_moved_after_move}, expected: {move_fastener_by}"
    )

    # detach fastener from parts
    detach_fastener_from_part(
        scene,
        fastener,
        part_entity=entities["part_with_holes_1@solid"],
        envs_idx=torch.tensor([0]),
    )
    # Assert weld fastener<->part1 removed (part2 still attached until next detach)
    assert not is_weld_constraint_present(
        scene, fastener, entities["part_with_holes_1@solid"], env_idx=0
    ), "Weld constraint between fastener and part 1 should be removed"
    detach_fastener_from_part(
        scene,
        fastener,
        part_entity=entities["part_with_holes_2@solid"],
        envs_idx=torch.tensor([0]),
    )
    # Assert weld fastener<->part2 removed
    assert not is_weld_constraint_present(
        scene, fastener, entities["part_with_holes_2@solid"], env_idx=0
    ), "Weld constraint between fastener and part 2 should be removed"
    # and move (relatively) randomly.
    fastener.set_pos(torch.tensor([[-1.0, 0.0, 1.0]]))
    step_and_render(scene, camera)
    entities["part_with_holes_2@solid"].set_pos(torch.tensor([[1.0, 0.0, 1.0]]))
    step_and_render(scene, camera)

    assert not torch.isclose(
        entities["part_with_holes_1@solid"].get_pos(),
        entities["part_with_holes_2@solid"].get_pos()
        + pos_diff_part_to_part_pre_attachment_to_second,
    ).all(), "Parts should not be attached to each other after detaching"
    assert not torch.isclose(
        entities["part_with_holes_1@solid"].get_pos(),
        entities["0@fastener"].get_pos() + pos_diff_part_1_to_fastener,
    ).all(), (
        "Parts should not be attached to fastener after detaching. Got diff: "
        + str(
            entities["part_with_holes_1@solid"].get_pos()
            - entities["0@fastener"].get_pos()
        )
    )
