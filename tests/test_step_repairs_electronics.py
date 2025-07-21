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
        "europlug_0_male@control": connector_europlug_male,
        "europlug_0_female@control": connector_europlug_female,
    }
    repairs_sim_state = RepairsSimState(1)
    europlug_male = Europlug(0)
    # hmm, and how do I register electronics? #TODO check translation.
    repairs_sim_state.electronics_state[0].register(europlug_male)

    return scene, entities, repairs_sim_state  # desired state defined separately.


# --------------------------
# === step_electronics ===
# --------------------------
