import pytest
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity
import numpy as np

from repairs_components.geometry.connectors.connectors import Connector
from repairs_components.geometry.fasteners import Fastener
from repairs_components.logic.tools.tool import ToolsEnum
from repairs_components.processing.geom_utils import get_connector_pos, quat_multiply
from repairs_components.processing.translation import translate_genesis_to_python
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_components.geometry.connectors.models.europlug import Europlug
from repairs_components.logic.physical_state import (
    register_bodies_batch,
    register_fasteners_batch,
)
from tests.global_test_config import init_gs


@pytest.fixture
def scene_with_entities(init_gs):
    """Create a real Genesis scene with various entity types."""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )

    # Add basic entities
    plane = scene.add_entity(gs.morphs.Plane())

    # Add solid parts
    part_1 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.1, 0.1, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)),
    )
    part_2 = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(-0.1, -0.1, 0.02)),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )

    # Add fastener
    fastener = scene.add_entity(
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.02)),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )

    # Add connectors (using europlug geometry)
    male_connector = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.2, 0.0, 0.02)),
        surface=gs.surfaces.Plastic(color=(1, 1, 0)),
    )
    female_connector = scene.add_entity(
        gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(-0.2, 0.0, 0.02)),
        surface=gs.surfaces.Plastic(color=(1, 0, 1)),
    )

    # Add franka for control
    # franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    scene.build(n_envs=1)

    # get europlug only for name:
    europlug = Europlug(0)  # scrap it
    male_name = europlug.get_name(0, male_female_both=True)
    female_name = europlug.get_name(0, male_female_both=False)
    assert male_name == "europlug_0_male" and female_name == "europlug_0_female", (
        "Expected different names at europlug name!"
    )  # note: not @connector. However everything else should use @connector.

    # Create entities dict with proper naming conventions
    entities: dict[str, RigidEntity] = {
        "part_1@solid": part_1,
        "part_2@solid": part_2,
        "0@fastener": fastener,
        "europlug_0_male@connector": male_connector,
        "europlug_0_female@connector": female_connector,
        # "franka@control": franka,
    }

    # create and populate RepairsSimState
    sim_state = RepairsSimState(1)
    sim_state.tool_state.tool_ids[0] = ToolsEnum.SCREWDRIVER.value
    sim_state.tool_state.screwdriver_tc = Screwdriver(
        picked_up_fastener_id=torch.tensor([0]),
        picked_up_fastener_tip_position=torch.tensor([-1.0, -1.0, -1.0]),
        picked_up_fastener_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        # ^ note: the expected is 0,0,0, so this is predictably incorrect.
        # it is expected to change.
    )

    sim_state.electronics_state[0].register(Europlug(0))

    # Batch register bodies (parts and connectors)
    names = [
        "part_1@solid",
        "part_2@solid",
        male_name + "@connector",
        female_name + "@connector",
    ]
    device = entities["part_1@solid"].get_pos(0).device
    dtype = entities["part_1@solid"].get_pos(0).dtype
    positions = torch.stack(
        [
            entities["part_1@solid"].get_pos(0),
            entities["part_2@solid"].get_pos(0),
            entities[male_name + "@connector"].get_pos(0),
            entities[female_name + "@connector"].get_pos(0),
        ],
        dim=1,
    )
    rotations = torch.stack(
        [
            entities["part_1@solid"].get_quat(0),
            entities["part_2@solid"].get_quat(0),
            entities[male_name + "@connector"].get_quat(0),
            entities[female_name + "@connector"].get_quat(0),
        ],
        dim=1,
    )
    fixed = torch.zeros((1, len(names)), dtype=torch.bool, device=device)
    terminal_position_relative_to_center = torch.cat(
        [
            torch.full((2, 3), float("nan"), device=device, dtype=dtype),
            torch.tensor(
                europlug.terminal_pos_relative_to_center_male / 1000,
                device=device,
                dtype=dtype,
            ).unsqueeze(0),
            torch.tensor(
                europlug.terminal_pos_relative_to_center_female / 1000,
                device=device,
                dtype=dtype,
            ).unsqueeze(0),
        ],
        dim=0,
    )

    register_bodies_batch(
        sim_state.physical_state,
        names=names,
        positions=positions,
        rotations=rotations,
        fixed=fixed,
        terminal_position_relative_to_center=terminal_position_relative_to_center,
    )
    # Register fasteners in batch (replace deprecated single registration)
    # Prepare a minimal placeholder part_hole_batch to satisfy fastener registration
    # (fastener is not attached initially, so -1 holes are used)
    sim_state.physical_state.part_hole_batch = torch.zeros(
        (1, 1), dtype=torch.long, device=device
    )

    fastener_name = Fastener().name  # e.g., "fastener_d5.00_l15.00@fastener"
    fastener_pos = torch.stack([entities["0@fastener"].get_pos(0)], dim=1)  # [B,1,3]
    fastener_quat = torch.stack([entities["0@fastener"].get_quat(0)], dim=1)  # [B,1,4]
    init_hole_a = torch.full((1, 1), -1, dtype=torch.long, device=device)
    init_hole_b = torch.full((1, 1), -1, dtype=torch.long, device=device)

    register_fasteners_batch(
        sim_state.physical_state,
        fastener_pos=fastener_pos,
        fastener_quat=fastener_quat,
        fastener_init_hole_a=init_hole_a,
        fastener_init_hole_b=init_hole_b,
        fastener_compound_names=[fastener_name],
    )

    return scene, entities, sim_state


@pytest.fixture
def sample_hole_data():
    """Sample hole data for testing."""
    starting_hole_positions = torch.tensor(
        [
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.15],
            [0.0, 0.0, 0.2],
        ]
    )
    starting_hole_quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    part_hole_batch = torch.tensor([0, 0, 1, 1])
    assert (
        starting_hole_positions.shape[0]
        == part_hole_batch.shape[0]
        == starting_hole_quats.shape[0]
        == 4
    ), "Expected a batch of holes of shape 4."  # just in case.
    return starting_hole_positions, starting_hole_quats, part_hole_batch


def test_translate_genesis_to_python(scene_with_entities, sample_hole_data):
    """Test translation of Genesis scene to RepairsSimState.

    Test:
    1. Test solid parts translated.
    2. Test fasteners translated.
    3. Test connectors translated and connector pos translated.
    4. Test holes translated (and moved).
    5. Test fastener tip translated.
    """
    scene, entities, sim_state = scene_with_entities
    starting_hole_positions, starting_hole_quats, part_hole_batch = sample_hole_data

    # Call the translation function
    translate_genesis_to_python(
        scene=scene,
        gs_entities=entities,
        sim_state=sim_state,
        starting_hole_positions=starting_hole_positions,
        starting_hole_quats=starting_hole_quats,
        part_hole_batch=part_hole_batch,
        device=starting_hole_positions.device,
    )

    # test solid bodies translated (pos and quat)
    graph_device = sim_state.physical_state.position.device
    gs_device = entities["part_1@solid"].get_pos(0).device

    # body 1
    part_1_id = sim_state.physical_state.body_indices["part_1@solid"]
    assert torch.allclose(
        sim_state.physical_state.position[0, part_1_id],
        entities["part_1@solid"].get_pos(0).squeeze(0).to(graph_device),
    )
    assert torch.allclose(
        sim_state.physical_state.quat[0, part_1_id],
        entities["part_1@solid"].get_quat(0).squeeze(0).to(graph_device),
    )

    part_2_id = sim_state.physical_state.body_indices["part_2@solid"]
    assert torch.allclose(
        sim_state.physical_state.position[0, part_2_id],
        entities["part_2@solid"].get_pos(0).squeeze(0).to(graph_device),
    )
    assert torch.allclose(
        sim_state.physical_state.quat[0, part_2_id],
        entities["part_2@solid"].get_quat(0).squeeze(0).to(graph_device),
    )

    # fastners
    fastener_id = 0  # note that fastener ID is 0 because it's 0 in graph.
    assert torch.allclose(
        sim_state.physical_state.fasteners_pos[0, fastener_id],
        entities["0@fastener"].get_pos(0).squeeze(0).to(graph_device),
    )
    assert torch.allclose(
        sim_state.physical_state.fasteners_quat[0, fastener_id],
        entities["0@fastener"].get_quat(0).squeeze(0).to(graph_device),
    )

    # europlug physical positions
    male_id = sim_state.physical_state.body_indices["europlug_0_male@connector"]
    assert torch.allclose(
        sim_state.physical_state.position[0, male_id],
        entities["europlug_0_male@connector"].get_pos(0).squeeze(0).to(graph_device),
    )
    assert torch.allclose(
        sim_state.physical_state.quat[0, male_id],
        entities["europlug_0_male@connector"].get_quat(0).squeeze(0).to(graph_device),
    )

    female_id = sim_state.physical_state.body_indices["europlug_0_female@connector"]
    assert torch.allclose(
        sim_state.physical_state.position[0, female_id],
        entities["europlug_0_female@connector"].get_pos(0).squeeze(0).to(graph_device),
    )
    assert torch.allclose(
        sim_state.physical_state.quat[0, female_id],
        entities["europlug_0_female@connector"].get_quat(0).squeeze(0).to(graph_device),
    )

    # get untranslated connector positions
    male_name = "europlug_0_male@connector"
    female_name = "europlug_0_female@connector"
    europlug = Connector.from_name(male_name)
    m_connector_pos_untranslated = torch.from_numpy(
        europlug.terminal_pos_relative_to_center_male / 1000
    ).to(gs_device)
    f_connector_pos_untranslated = torch.from_numpy(
        europlug.terminal_pos_relative_to_center_female / 1000
    ).to(gs_device)

    # terminal_def pos should be at their positions too.
    # Get male connector index and position from tensor-based structure
    male_connector_idx = sim_state.physical_state[0].terminal_indices_from_name[
        male_name
    ]
    terminal_def_actual_m = sim_state.physical_state.male_terminal_positions[
        0, male_connector_idx
    ].to(gs_device)
    terminal_def_expected_m = get_connector_pos(
        entities[male_name].get_pos(0),
        entities[male_name].get_quat(0),
        m_connector_pos_untranslated.unsqueeze(0),
    ).squeeze(0)
    assert torch.allclose(terminal_def_actual_m, terminal_def_expected_m)

    # terminal_def pos should be at their positions too.
    # Get female connector index and position from tensor-based structure
    female_connector_idx = sim_state.physical_state[0].terminal_indices_from_name[
        female_name
    ]
    terminal_def_actual_f = sim_state.physical_state.female_terminal_positions[
        0, female_connector_idx
    ].to(gs_device)
    terminal_def_expected_f = get_connector_pos(
        entities[female_name].get_pos(0),
        entities[female_name].get_quat(0),
        f_connector_pos_untranslated.unsqueeze(0),
    ).squeeze(0)
    assert torch.allclose(terminal_def_actual_f, terminal_def_expected_f)

    # holes should be updated relative to bodies:
    holes_actual = sim_state.physical_state.hole_positions
    part_pos_batched = torch.stack(
        [  # note: as hole batch!
            entities["part_1@solid"].get_pos(0),  # 0
            entities["part_1@solid"].get_pos(0),  # 0
            entities["part_2@solid"].get_pos(0),  # 1
            entities["part_2@solid"].get_pos(0),  # 1
        ],
        dim=1,
    )
    part_quat_batched = torch.stack(
        [
            entities["part_1@solid"].get_quat(0),
            entities["part_1@solid"].get_quat(0),
            entities["part_2@solid"].get_quat(0),
            entities["part_2@solid"].get_quat(0),
        ],
        dim=1,
    )
    holes_expected = get_connector_pos(
        part_pos_batched,
        part_quat_batched,
        starting_hole_positions.unsqueeze(0),
    )
    assert torch.allclose(holes_actual, holes_expected)
    # quats of holes should be updated.
    holes_actual_quats = sim_state.physical_state.hole_quats
    holes_quats_expected = quat_multiply(starting_hole_quats, part_quat_batched)
    assert torch.allclose(holes_actual_quats, holes_quats_expected)

    # fastener tip should be updated.
    assert sim_state.tool_state.tool_ids[0] == ToolsEnum.SCREWDRIVER.value, (
        f"The tool shouldn't have changed during execution. Got {ToolsEnum(sim_state.tool_state.tool_ids[0]).name}"
    )
    fastener_tip_actual = (
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_tip_position[0]
    )
    tip_relative_to_center = Fastener.get_tip_pos_relative_to_center().unsqueeze(0)
    fastener_tip_expected = get_connector_pos(
        entities["0@fastener"].get_pos(0),
        entities["0@fastener"].get_quat(0),
        tip_relative_to_center,
    ).squeeze(0)
    # roughly there.
    assert torch.allclose(fastener_tip_actual, fastener_tip_expected, atol=0.1), (
        f"Fastener tip position in state {fastener_tip_actual} != expected {fastener_tip_expected}"
    )
    assert torch.allclose(
        sim_state.tool_state.screwdriver_tc.picked_up_fastener_quat[0],
        entities["0@fastener"].get_quat(0),
        atol=0.1,
    ), "Fastener tip quaternion in state != expected"
