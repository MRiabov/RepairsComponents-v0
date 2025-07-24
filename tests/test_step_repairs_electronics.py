import pytest
import genesis as gs
from repairs_components.geometry.connectors.models.europlug import Europlug
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.electronics.electronics_state import ElectronicsState
from repairs_components.logic.tools.screwdriver import Screwdriver
from repairs_sim_step import step_electronics
from genesis.engine.entities import RigidEntity


@pytest.fixture(scope="module")
def scene_with_two_connectors(init_gs):
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

    europlug_male_pos = (0.0, -0.2, 0.02)
    europlug_female_pos = (-0.2, 0.0, 0.02)

    # "tool cube" and "fastener cube" as stubs for real geometry. Functionally the same.

    connector_europlug_male = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=europlug_male_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)),  # blue
    )
    connector_europlug_female = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=europlug_female_pos,
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

    entities = {
        "europlug_0_male@connector": connector_europlug_male,
        "europlug_0_female@connector": connector_europlug_female,
    }
    repairs_sim_state = RepairsSimState(1)
    europlug_male = Europlug(0)
    # hmm, and how do I register electronics? #TODO check translation.
    repairs_sim_state.electronics_state[0].register(europlug_male)

    return scene, entities, repairs_sim_state  # desired state defined separately.


# --------------------------
# === step_electronics ===
# --------------------------


def test_step_electronics_no_electronics():
    """Test step_electronics when has_electronics is False"""
    # Create a minimal RepairsSimState without electronics
    repairs_sim_state = RepairsSimState(1)
    repairs_sim_state.has_electronics = False

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Should return the same state unchanged
    assert result is repairs_sim_state
    assert not result.has_electronics


def test_step_electronics_with_electronics_no_connections():
    """Test step_electronics with electronics but no valid connections"""
    import torch
    from repairs_components.geometry.connectors.models.europlug import Europlug

    # Create RepairsSimState with electronics
    repairs_sim_state = RepairsSimState(1)
    repairs_sim_state.has_electronics = True

    # Add connector positions that are too far apart to connect
    male_positions = torch.tensor([0.0, 0.0, 0.0])
    female_positions = torch.tensor([10.0, 0.0, 0.0])  # far away

    # Register some components in electronics state
    europlug = Europlug(0)
    repairs_sim_state.electronics_state[0].register(europlug)
    # Use component name (without @connector suffix) for connector positions
    component_name = europlug.get_name(europlug.in_sim_id, male_female_both=None)

    repairs_sim_state.physical_state[0].male_connector_positions = {
        component_name: male_positions
    }
    repairs_sim_state.physical_state[0].female_connector_positions = {
        component_name: female_positions
    }

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Should return the same state with no connections
    assert result is repairs_sim_state
    # Use the auto-generated names from europlug
    europlug_name = europlug.get_name(europlug.in_sim_id, male_female_both=None)
    assert len(result.electronics_state[0].components[europlug_name].connected_to) == 0


def test_step_electronics_with_valid_connections():
    """Test step_electronics with electronics that should connect"""
    import torch
    from repairs_components.geometry.connectors.models.europlug import Europlug

    # Create RepairsSimState with electronics
    repairs_sim_state = RepairsSimState(1)
    repairs_sim_state.has_electronics = True

    # Add connector positions that are close enough to connect
    male_positions = torch.tensor([0.0, 0.0, 0.0])
    female_positions = torch.tensor(
        [1.0, 0.0, 0.0]
    )  # close enough (distance = 1.0 < 2.5)

    # Register components in electronics state and get auto-generated names
    europlug_male = Europlug(0)
    europlug_female = Europlug(1)

    male_component_name = europlug_male.get_name(
        europlug_male.in_sim_id, male_female_both=None
    )
    female_component_name = europlug_female.get_name(
        europlug_female.in_sim_id, male_female_both=None
    )

    repairs_sim_state.physical_state[0].male_connector_positions = {
        male_component_name: male_positions
    }
    repairs_sim_state.physical_state[0].female_connector_positions = {
        female_component_name: female_positions
    }

    repairs_sim_state.electronics_state[0].register(europlug_male)
    repairs_sim_state.electronics_state[0].register(europlug_female)

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Should create a connection between the components
    assert result is repairs_sim_state

    assert (
        len(result.electronics_state[0].components[male_component_name].connected_to)
        == 1
    )
    assert (
        len(result.electronics_state[0].components[female_component_name].connected_to)
        == 1
    )
    assert (
        result.electronics_state[0].components[male_component_name].connected_to[0].name
        == female_component_name
    )
    assert (
        result.electronics_state[0]
        .components[female_component_name]
        .connected_to[0]
        .name
        == male_component_name
    )


def test_step_electronics_clears_previous_connections():
    """Test that step_electronics clears previous connections before applying new ones"""
    import torch
    from repairs_components.geometry.connectors.models.europlug import Europlug

    # Create RepairsSimState with electronics
    repairs_sim_state = RepairsSimState(1)
    repairs_sim_state.has_electronics = True

    # Add connector positions
    male_positions = torch.tensor([0.0, 0.0, 0.0])
    female_positions = torch.tensor([1.0, 0.0, 0.0])

    # Register components and get auto-generated names
    europlug_male = Europlug(0)
    europlug_female = Europlug(1)

    male_component_name = europlug_male.get_name(
        europlug_male.in_sim_id, male_female_both=None
    )
    female_component_name = europlug_female.get_name(
        europlug_female.in_sim_id, male_female_both=None
    )

    repairs_sim_state.physical_state[0].male_connector_positions = {
        male_component_name: male_positions
    }
    repairs_sim_state.physical_state[0].female_connector_positions = {
        female_component_name: female_positions
    }

    repairs_sim_state.electronics_state[0].register(europlug_male)
    repairs_sim_state.electronics_state[0].register(europlug_female)

    # Create a dummy previous connection
    europlug = Europlug(2)
    repairs_sim_state.electronics_state[0].register(europlug)
    dummy_component_name = europlug.get_name(
        europlug.in_sim_id, male_female_both=None
    )

    repairs_sim_state.electronics_state[0].components[male_component_name].connect(
        europlug
    )
    europlug.connect(
        repairs_sim_state.electronics_state[0].components[male_component_name]
    )

    # Verify the dummy connection exists
    assert (
        len(
            repairs_sim_state.electronics_state[0]
            .components[male_component_name]
            .connected_to
        )
        == 1
    )
    assert (
        repairs_sim_state.electronics_state[0]
        .components[male_component_name]
        .connected_to[0]
        .name
        == dummy_component_name
    )

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Should clear previous connections and create new ones
    assert result is repairs_sim_state
    assert (
        len(result.electronics_state[0].components[male_component_name].connected_to)
        == 1
    )
    assert (
        result.electronics_state[0].components[male_component_name].connected_to[0].name
        == female_component_name
    )
    assert (
        len(result.electronics_state[0].components[dummy_component_name].connected_to)
        == 0
    )


def test_step_electronics_multiple_batch_environments():
    """Test step_electronics with multiple batch environments"""
    import torch
    from repairs_components.geometry.connectors.models.europlug import Europlug

    # Create RepairsSimState with 2 environments
    repairs_sim_state = RepairsSimState(2)
    repairs_sim_state.has_electronics = True

    # Register components and get auto-generated names
    europlug_male_0 = Europlug(0)
    europlug_female_0 = Europlug(1)
    europlug_male_1 = Europlug(0)
    europlug_female_1 = Europlug(1)

    male_component_name = europlug_male_0.get_name(
        europlug_male_0.in_sim_id, male_female_both=None
    )
    female_component_name = europlug_female_0.get_name(
        europlug_female_0.in_sim_id, male_female_both=None
    )

    # Environment 0: connectors that should connect (close)
    male_positions_0 = torch.tensor([0.0, 0.0, 0.0])
    female_positions_0 = torch.tensor([1.0, 0.0, 0.0])  # close enough

    repairs_sim_state.physical_state[0].male_connector_positions = {
        male_component_name: male_positions_0
    }
    repairs_sim_state.physical_state[0].female_connector_positions = {
        female_component_name: female_positions_0
    }

    # Environment 1: connectors that should not connect (far)
    male_positions_1 = torch.tensor([0.0, 0.0, 0.0])
    female_positions_1 = torch.tensor([10.0, 0.0, 0.0])  # too far

    repairs_sim_state.physical_state[1].male_connector_positions = {
        male_component_name: male_positions_1
    }
    repairs_sim_state.physical_state[1].female_connector_positions = {
        female_component_name: female_positions_1
    }

    # Register components for both environments
    repairs_sim_state.electronics_state[0].register(europlug_male_0)
    repairs_sim_state.electronics_state[0].register(europlug_female_0)
    repairs_sim_state.electronics_state[1].register(europlug_male_1)
    repairs_sim_state.electronics_state[1].register(europlug_female_1)

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Environment 0 should have a connection
    assert (
        len(result.electronics_state[0].components[male_component_name].connected_to)
        == 1
    )
    assert (
        result.electronics_state[0].components[male_component_name].connected_to[0].name
        == female_component_name
    )

    # Environment 1 should have no connections
    assert (
        len(result.electronics_state[1].components[male_component_name].connected_to)
        == 0
    )
    assert (
        len(result.electronics_state[1].components[female_component_name].connected_to)
        == 0
    )


def test_step_electronics_multiple_connectors():
    """Test step_electronics with multiple connector types"""
    import torch
    from repairs_components.geometry.connectors.models.europlug import Europlug

    # Create RepairsSimState
    repairs_sim_state = RepairsSimState(1)
    repairs_sim_state.has_electronics = True

    # Register components and get auto-generated names
    europlug_male_1 = Europlug(0)
    europlug_male_2 = Europlug(1)
    europlug_female_1 = Europlug(2)
    europlug_female_2 = Europlug(3)

    male_component_name_1 = europlug_male_1.get_name(
        europlug_male_1.in_sim_id, male_female_both=None
    )
    male_component_name_2 = europlug_male_2.get_name(
        europlug_male_2.in_sim_id, male_female_both=None
    )
    female_component_name_1 = europlug_female_1.get_name(
        europlug_female_1.in_sim_id, male_female_both=None
    )
    female_component_name_2 = europlug_female_2.get_name(
        europlug_female_2.in_sim_id, male_female_both=None
    )

    # Add multiple connector positions
    male_positions_1 = torch.tensor([0.0, 0.0, 0.0])
    male_positions_2 = torch.tensor([5.0, 0.0, 0.0])
    female_positions_1 = torch.tensor([1.0, 0.0, 0.0])  # close to male_1
    female_positions_2 = torch.tensor([5.5, 0.0, 0.0])  # close to male_2

    repairs_sim_state.physical_state[0].male_connector_positions = {
        male_component_name_1: male_positions_1,
        male_component_name_2: male_positions_2,
    }
    repairs_sim_state.physical_state[0].female_connector_positions = {
        female_component_name_1: female_positions_1,
        female_component_name_2: female_positions_2,
    }

    # Register components
    repairs_sim_state.electronics_state[0].register(europlug_male_1)
    repairs_sim_state.electronics_state[0].register(europlug_male_2)
    repairs_sim_state.electronics_state[0].register(europlug_female_1)
    repairs_sim_state.electronics_state[0].register(europlug_female_2)

    # Call step_electronics
    result = step_electronics(repairs_sim_state)

    # Should create two connections: male_1-female_1 and male_2-female_2
    assert (
        len(result.electronics_state[0].components[male_component_name_1].connected_to)
        == 1
    )
    assert (
        result.electronics_state[0]
        .components[male_component_name_1]
        .connected_to[0]
        .name
        == female_component_name_1
    )

    assert (
        len(result.electronics_state[0].components[male_component_name_2].connected_to)
        == 1
    )
    assert (
        result.electronics_state[0]
        .components[male_component_name_2]
        .connected_to[0]
        .name
        == female_component_name_2
    )

    assert (
        len(
            result.electronics_state[0].components[female_component_name_1].connected_to
        )
        == 1
    )
    assert (
        result.electronics_state[0]
        .components[female_component_name_1]
        .connected_to[0]
        .name
        == male_component_name_1
    )

    assert (
        len(
            result.electronics_state[0].components[female_component_name_2].connected_to
        )
        == 1
    )
    assert (
        result.electronics_state[0]
        .components[female_component_name_2]
        .connected_to[0]
        .name
        == male_component_name_2
    )
