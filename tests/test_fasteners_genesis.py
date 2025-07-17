import genesis as gs
import numpy as np

# import argparse
from genesis.engine.entities import RigidEntity
import torch
from PIL import Image
import pytest

from repairs_components.geometry.fasteners import (
    Fastener,
    attach_fastener_to_part,
    attach_fastener_to_screwdriver,
    detach_fastener_from_part,
    detach_fastener_from_screwdriver,
)
from repairs_components.logic.tools.screwdriver import Screwdriver


@pytest.fixture
def fastener():
    return Fastener(
        constraint_b_active=True,
        initial_body_a="body_a",
        initial_body_b="body_b",
        length=15.0,
        diameter=5.0,
        b_depth=5.0,
        head_diameter=7.5,
        head_height=3.0,
        thread_pitch=0.5,
        screwdriver_name="screwdriver",
    )


@pytest.fixture
def bd_geometry(fastener):
    return fastener.bd_geometry()


@pytest.fixture(scope="module")
def scene_with_fastener_screwdriver_and_two_parts():
    from repairs_components.logic.tools.screwdriver import Screwdriver
    from repairs_components.logic.tools.tool import attach_tool_to_arm

    ########################## init ##########################
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")

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
        "screwdriver_cube": screwdriver_cube,
        "fastener_cube": fastener_cube,
        "franka": franka,
        "end_effector": end_effector,
        "part_with_holes_1": part_with_holes_1,
        "part_with_holes_2": part_with_holes_2,
    }
    return scene, entities


def test_attach_and_detach_fastener_to_screwdriver(
    scene_with_fastener_screwdriver_and_two_parts,
):
    scene, entities = scene_with_fastener_screwdriver_and_two_parts
    screwdriver_pos = torch.tensor([[0.0, 0.0, 1.0]])
    camera = scene.visualizer.cameras[0]
    entities["screwdriver_cube"].set_pos(screwdriver_pos)
    screwdriver = Screwdriver()
    attach_fastener_to_screwdriver(
        scene,
        entities["fastener_cube"],
        entities["screwdriver_cube"],
        tool_state_to_update=screwdriver,
        fastener_id=0,
        env_id=0,
    )
    fastener_grip_pos = Screwdriver.fastener_connector_pos_relative_to_center()
    assert torch.isclose(
        entities["fastener_cube"].get_pos(0), screwdriver_pos + fastener_grip_pos
    ).all(), (
        f"Fastener cube should be attached to screwdriver cube at the fastener connector position. "
        f"Fastener cube pos: {entities['fastener_cube'].get_pos(0)}, screwdriver pos: {screwdriver_pos}, fastener grip pos: {fastener_grip_pos}"
    )
    assert torch.isclose(
        entities["fastener_cube"].get_quat(), entities["screwdriver_cube"].get_quat()
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )  # FIXME: this is likely wrong - a) screwdriver may be inherently incorrectly imported, b) franka is rotated as 0,1,0,0 as base.
    assert screwdriver.has_picked_up_fastener == True
    assert torch.isclose(
        screwdriver.picked_up_fastener_tip_position, screwdriver_pos + fastener_grip_pos
    ).all()
    assert screwdriver.picked_up_fastener_name == "0@fastener"

    # detach fastener from screwdriver and assert they fall down.
    
    detach_fastener_from_screwdriver(
        scene,
        entities["fastener_cube"],
        entities["screwdriver_cube"],
        screwdriver,
        env_id=0,
    )
    # assert tool state
    assert screwdriver.has_picked_up_fastener == False
    assert screwdriver.picked_up_fastener_tip_position is None
    assert screwdriver.picked_up_fastener_name is None

    for i in range(100):
        scene.step()
        camera.render()
    assert torch.isclose(
        entities["fastener_cube"].get_pos(0)[0, 2], torch.tensor([0.01]), atol=0.10
    ), "Fastener cube should be close to the ground"
    # assert fastener is far away from screwdriver
    assert torch.isclose(
        entities["screwdriver_cube"].get_pos(0),
        torch.tensor([[0.0, 0.0, 1.0]]),
        atol=0.15,
    ).all(), "Screwdriver should've left at the same place."
    


def test_attach_and_detach_fastener_to_part(
    scene_with_fastener_screwdriver_and_two_parts,
):
    scene, entities = scene_with_fastener_screwdriver_and_two_parts
    screwdriver_pos = torch.tensor([[0.0, 0.0, 1.0]])
    camera = scene.visualizer.cameras[0]
    entities["screwdriver_cube"].set_pos(screwdriver_pos)
    screwdriver = Screwdriver()
    attach_fastener_to_screwdriver(
        scene,
        entities["fastener_cube"],
        entities["screwdriver_cube"],
        tool_state_to_update=screwdriver,
        fastener_id=0,
        env_id=0,
    )
    fastener_grip_pos = Screwdriver.fastener_connector_pos_relative_to_center()
    assert torch.isclose(
        entities["fastener_cube"].get_pos(0), screwdriver_pos + fastener_grip_pos
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )
    assert torch.isclose(
        entities["fastener_cube"].get_quat(), entities["screwdriver_cube"].get_quat()
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position"
    )  # FIXME: this is likely wrong - a) screwdriver may be inherently incorrectly imported, b) franka is rotated as 0,1,0,0 as base.

    hole_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hole_pos = torch.tensor([0.5, 0.5, 0.04])
    # attach fastener to part
    attach_fastener_to_part(
        scene,
        entities["fastener_cube"],
        part_entity=entities["part_with_holes_1"],
        hole_pos=torch.tensor(hole_pos),
        hole_quat=torch.tensor(hole_quat),
        envs_idx=torch.tensor([0]),
    )
    assert torch.isclose(entities["fastener_cube"].get_pos(0), hole_pos).all(), (
        "Fastener cube should be attached to part with holes"
    )
    assert torch.isclose(entities["fastener_cube"].get_quat(), hole_quat).all(), (
        "Fastener cube should be attached to part with holes"
    )
    assert torch.isclose(
        entities["screwdriver_cube"].get_pos(0), screwdriver_pos + fastener_grip_pos
    ).all(), (
        "Screwdriver cube should be attached to screwdriver cube at the fastener connector position after attaching to part"
    )
    assert torch.isclose(
        entities["screwdriver_cube"].get_quat(), entities["screwdriver_cube"].get_quat()
    ).all(), (
        "Screwdriver cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )

    # move screwdriver to a new position
    screwdriver_pos = torch.tensor([[0.0, 0.0, 2.0]])
    entities["screwdriver_cube"].set_pos(screwdriver_pos)
    entities["screwdriver_cube"].set_quat(torch.tensor([1.0, 0, 0, 0]))

    assert torch.isclose(
        entities["screwdriver_cube"].get_pos(0), screwdriver_pos + fastener_grip_pos
    ).all(), (
        "Screwdriver cube should be attached to screwdriver cube at the fastener connector position after attaching to part"
    )
    assert torch.isclose(
        entities["screwdriver_cube"].get_quat(), entities["screwdriver_cube"].get_quat()
    ).all(), (
        "Screwdriver cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )

    # detach fastener from part
    detach_fastener_from_part(
        scene,
        entities["fastener_cube"],
        part_entity=entities["part_with_holes_1"],
        envs_idx=torch.tensor([0]),
    )
    assert torch.isclose(
        entities["fastener_cube"].get_pos(0), screwdriver_pos + fastener_grip_pos
    ).all(), (
        "Fastener cube should be attached to screwdriver cube at the fastener connector position after attaching to part"
    )
    assert torch.isclose(
        entities["fastener_cube"].get_quat(), entities["screwdriver_cube"].get_quat()
    ).all(), (
        "Fastener cube should be (still) attached to screwdriver cube at the fastener connector position after attaching to part"
    )
    # let the fastener fall down; let the part fall down too.
    for i in range(200):
        scene.step()
        camera.render()
    assert torch.isclose(
        entities["fastener_cube"].get_pos(0)[2], torch.tensor([0.01]), atol=0.10
    ), "Fastener cube should be close to the ground"
    assert torch.isclose(
        entities["part_with_holes_1"].get_pos(0)[2], torch.tensor([0.01]), atol=0.10
    ), "Part with holes should be close to the ground"


def test_attach_and_detach_fastener_to_two_parts(
    scene_with_fastener_screwdriver_and_two_parts,
):
    scene, entities = scene_with_fastener_screwdriver_and_two_parts
    camera = scene.visualizer.cameras[0]
    fastener = entities["fastener_cube"]

    attach_fastener_to_part(
        scene,
        fastener,
        part_entity=entities["part_with_holes_1"],
        envs_idx=torch.tensor([0]),
        hole_pos=torch.tensor([0.5, 0.5, 0.04]),
        hole_quat=torch.tensor([1, 0, 0, 0]),
    )
    attach_fastener_to_part(
        scene,
        fastener,
        part_entity=entities["part_with_holes_2"],
        envs_idx=torch.tensor([0]),
        hole_pos=torch.tensor([0.5, 0.5, 0.04]),
        hole_quat=torch.tensor([1, 0, 0, 0]),
    )
