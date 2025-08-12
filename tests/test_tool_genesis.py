import genesis as gs
import numpy as np

# import argparse
from genesis.engine.entities import RigidEntity
import torch
from PIL import Image
import pytest

from repairs_components.geometry.fasteners import (
    Fastener,
    attach_fastener_to_screwdriver,
)
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.logic.tools.tool import (
    ToolsEnum,
    attach_tool_to_arm,
    detach_tool_from_arm,
)
from repairs_components.logic.tools.tools_state import ToolState
from repairs_components.processing.geom_utils import get_connector_pos
from tests.global_test_config import init_gs, base_data_dir


@pytest.fixture(scope="module")
def scene_franka_and_two_cubes(init_gs):
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
    plane = scene.add_entity(gs.morphs.Plane())
    # "tool cube" and "fastener cube" as stubs for real geometry. Functionally the same.
    tool_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    fastener_cube: RigidEntity = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.0, 0.65, 0.02),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka: RigidEntity = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
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
        "tool_cube": tool_cube,
        "0@fastener": fastener_cube,
        "franka@control": franka,
        "end_effector": end_effector,
    }
    return scene, entities


@pytest.fixture
def motors_dof():
    return np.arange(7)


@pytest.fixture
def fingers_dof():
    return np.arange(7, 9)


@pytest.fixture(autouse=True)
def cleanup_after_test(request, scene_franka_and_two_cubes, base_data_dir):
    yield
    test_name = request.node.name
    scene, entities = scene_franka_and_two_cubes
    scene.visualizer.cameras[0].stop_recording(
        save_to_filename=str(base_data_dir / f"test_videos/video_{test_name}.mp4"),
        fps=60,
    )
    scene.reset()
    scene.visualizer.cameras[0].start_recording()


def move_franka_to_pos(scene, franka, end_effector, pos, camera, fingers_dof):
    "A util for IK."
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=pos,
        quat=torch.tensor([[0, 1, 0, 0]]),  # downward facing orientation.
    )
    # gripper open pos
    qpos[:, -2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=100)
    # 1s duration
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        franka.control_dofs_force(torch.Tensor([[0.5, 0.5]]), fingers_dof)
        scene.step()
        camera.render()
    for i in range(50):
        scene.step()


def test_attach_tool_to_arm(scene_franka_and_two_cubes, fingers_dof):
    scene, entities = scene_franka_and_two_cubes
    franka = entities["franka@control"]
    tool_cube = entities["tool_cube"]
    end_effector = entities["end_effector"]
    camera = scene.visualizer.cameras[0]
    tool_state = ToolState(batch_size=1)
    tool_state.screwdriver_tc = Screwdriver()
    tool_state.tool_ids = torch.tensor([ToolsEnum.SCREWDRIVER.value])
    # move to pre-grasp pose
    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.tensor([[0.65, 0.0, 0.5]]),
        camera,
        fingers_dof,
    )

    print("end_effector.get_pos(): ", end_effector.get_pos(0))
    print("end_effector.get_quat(): ", end_effector.get_quat(0))

    # tested func.
    attach_tool_to_arm(
        scene,
        tool_cube,
        end_effector,
        tool_state,
        torch.tensor([0]),
    )
    rgb, _, _, _ = camera.render()
    Image.fromarray(rgb).save("cube_tool.png")

    expected_tool_pos = get_connector_pos(
        end_effector.get_pos(0).squeeze(1),
        end_effector.get_quat(0).squeeze(1),
        screwdriver.tool_grip_position().unsqueeze(0),
    )
    assert torch.isclose(tool_cube.get_pos(0), expected_tool_pos, atol=0.05).all(), (
        f"Tool pos should align with hand grip offset, got {tool_cube.get_pos(0)} vs {expected_tool_pos}"
    )
    assert (tool_cube.get_AABB()[0, 0, 2] < end_effector.get_AABB()[0, 0, 2]).all(), (
        "expected min of tool AABB to be lower than hand pos"
    )

    # move to
    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.tensor([[0.0, 0.65, 0.5]]),
        camera,
        fingers_dof,
    )

    expected_tool_pos = get_connector_pos(
        end_effector.get_pos(0).squeeze(1),
        end_effector.get_quat(0).squeeze(1),
        screwdriver.tool_grip_position().unsqueeze(0),
    )
    assert torch.isclose(tool_cube.get_pos(), expected_tool_pos, atol=0.05).all(), (
        f"Tool pos should align with hand grip offset, got {tool_cube.get_pos()} vs {expected_tool_pos}"
    )
    assert (tool_cube.get_AABB()[0, 0, 2] < end_effector.get_AABB()[0, 0, 2]).all(), (
        "expected min of tool AABB to be lower than hand pos"
    )


def test_detach_tool_from_arm(scene_franka_and_two_cubes, fingers_dof):
    scene, entities = scene_franka_and_two_cubes
    franka = entities["franka@control"]
    tool_cube = entities["tool_cube"]
    end_effector = entities["end_effector"]
    camera = scene.visualizer.cameras[0]
    tool_state = ToolState(
        batch_size=1,
        screwdriver_tc=Screwdriver(batch_size=1),
        tool_ids=torch.tensor([ToolsEnum.SCREWDRIVER.value]),
    )

    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.Tensor([[0.65, 0.0, 0.5]]),
        camera,
        fingers_dof,
    )

    print("end_effector.get_pos(): ", end_effector.get_pos())
    print("end_effector.get_quat(): ", end_effector.get_quat())

    # attach tool to arm
    attach_tool_to_arm(
        scene,
        tool_cube,
        end_effector,
        tool_state,
        torch.tensor([0]),
    )
    rgb, _, _, _ = camera.render()
    Image.fromarray(rgb).save("cube_tool.png")

    assert torch.isclose(
        tool_cube.get_pos(0),
        torch.tensor([[0.65, 0.0, 0.5 - screwdriver.tool_grip_position()[2]]]),
        atol=0.05,
    ).all(), (
        f"Cube pos expected to be [0.65, 0.0, 0.5 - screwdriver.tool_grip_position()[2]], got {tool_cube.get_pos(0)}"
    )

    # move to
    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.Tensor([[0.0, 0.65, 0.5]]),
        camera,
        fingers_dof,
    )

    expected_tool_pos = get_connector_pos(
        end_effector.get_pos(0).squeeze(1),
        end_effector.get_quat(0).squeeze(1),
        screwdriver.tool_grip_position().unsqueeze(0),
    )
    assert torch.isclose(tool_cube.get_pos(0), expected_tool_pos, atol=0.05).all(), (
        f"Tool pos should align with hand grip offset, got {tool_cube.get_pos(0)} vs {expected_tool_pos}"
    )

    # detach tool from arm
    detach_tool_from_arm(
        scene, tool_cube, end_effector, entities, tool_state, torch.tensor([0])
    )
    for i in range(100):
        scene.step()
        camera.render()
    assert torch.isclose(
        tool_cube.get_pos()[:, 2], torch.tensor([0.05]), atol=0.05
    ).all(), f"Expected the tool_cube to fall, got Z pos {tool_cube.get_pos()[2]}"


@pytest.mark.xfail(
    reason="Assertions are incorrect until get_weld_constraints is available in Genesis API."
)
def test_attach_and_detach_tool_to_arm_with_fastener(
    scene_franka_and_two_cubes, fingers_dof
):
    """2-in-1: attach tool, raise the tool, pick up body with a tool, detach tool from arm
    both tool and body should fall to floor."""
    scene, entities = scene_franka_and_two_cubes
    franka = entities["franka@control"]
    tool_cube = entities["tool_cube"]
    fastener_cube = entities["0@fastener"]  # note: this is cube geom.
    end_effector = entities["end_effector"]
    camera = scene.visualizer.cameras[0]
    screwdriver = Screwdriver()

    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.Tensor([[0.65, 0.0, 0.5]]),
        camera,
        fingers_dof,
    )

    print("end_effector.get_pos(): ", end_effector.get_pos())
    print("end_effector.get_quat(): ", end_effector.get_quat())

    # attach tool to arm
    attach_tool_to_arm(scene, tool_cube, end_effector, screwdriver, torch.tensor([0]))
    rgb, _, _, _ = camera.render()
    Image.fromarray(rgb).save("cube_tool.png")

    assert torch.isclose(
        tool_cube.get_pos(0),
        torch.tensor([[0.65, 0.0, 0.5 - screwdriver.tool_grip_position()[2]]]),
        atol=0.15,
    ).all(), (
        f"Cube pos expected to be [0.65, 0.0, 0.5 - screwdriver.tool_grip_position()[2]], got {tool_cube.get_pos(0)}"
    )
    # give it a more efficient trajectory (didn't plan well w/o it.)
    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.Tensor([[0.4, 0.4, 0.7]]),
        camera,
        fingers_dof,
    )

    # move to second cube # (higher than usual)
    move_franka_to_pos(
        scene,
        franka,
        end_effector,
        torch.Tensor([[0.0, 0.65, 0.7]]),
        camera,
        fingers_dof,
    )
    # give it more time to get to pos.
    for i in range(200):
        scene.step()
        camera.render()

    assert torch.isclose(
        tool_cube.get_pos(0),
        torch.tensor([[0.0, 0.65, 0.7 - screwdriver.tool_grip_position()[2]]]),
        atol=0.05,
    ).all(), (
        f"Cube pos expected to be [0.0, 0.65, 0.7 - screwdriver.tool_grip_position()[2]], got {tool_cube.get_pos(0)}"
    )

    # attach second cube to tool
    screwdriver = Screwdriver()

    expected_z = (
        0.7
        - screwdriver.tool_grip_position()[2]
        + screwdriver.fastener_connector_pos_relative_to_center()[2]
    )  # hmm, this isn't even right in tests. "+" and "-" mismatch.

    reposition_cube_to_xyz = torch.tensor([0.0, 0.65, expected_z])
    attach_fastener_to_screwdriver(
        scene,
        fastener_cube,
        tool_cube,
        tool_state_to_update=screwdriver,
        fastener_id=0,
        env_id=0,
    )
    assert torch.isclose(
        fastener_cube.get_pos(0), reposition_cube_to_xyz, atol=0.05
    ).all(), (
        f"Cube pos expected to be {reposition_cube_to_xyz}, got {fastener_cube.get_pos(0)}"
    )
    assert screwdriver.has_picked_up_fastener
    assert screwdriver.picked_up_fastener_name(
        0
    ) == Fastener.fastener_name_in_simulation(0)
    assert torch.isclose(
        screwdriver.picked_up_fastener_tip_position,
        get_connector_pos(
            reposition_cube_to_xyz,
            tool_cube.get_quat(),
            Fastener.get_tip_pos_relative_to_center().unsqueeze(0),
        ),
        atol=0.05,
    ).all()
    assert torch.isclose(
        screwdriver.picked_up_fastener_quat,
        tool_cube.get_quat(),
        atol=0.01,
    ).all()

    # detach tool from arm
    detach_tool_from_arm(
        scene, tool_cube, end_effector, entities, [screwdriver], torch.tensor([0])
    )
    assert not screwdriver.has_picked_up_fastener
    assert screwdriver.picked_up_fastener_name is None
    assert torch.isnan(screwdriver.picked_up_fastener_tip_position).all()
    assert torch.isnan(screwdriver.picked_up_fastener_quat).all()

    for i in range(200):
        scene.step()
        camera.render()
    assert torch.isclose(
        tool_cube.get_pos(0)[:, 2], torch.tensor([0.05]), atol=0.05
    ).all(), f"Expected the tool cube to fall, got Z pos {tool_cube.get_pos(0)[:, 2]}"
    assert torch.isclose(
        fastener_cube.get_pos(0)[:, 2], torch.tensor([0.05]), atol=0.05
    ).all(), (
        f"Expected the fastener cube to fall, got Z pos {fastener_cube.get_pos(0)[:, 2]}"
    )
