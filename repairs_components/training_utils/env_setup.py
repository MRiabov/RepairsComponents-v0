from abc import ABC, abstractmethod
from build123d import Compound
from genesis.vis.camera import Camera
import genesis as gs
from repairs_components.geometry.base_env.tooling_stand_plate import (
    genesis_setup,
    plate_env_bd_geometry,
    render_and_save,
)
from repairs_components.processing.tasks import Task


class EnvSetup(ABC):
    """A setup for a single problem to solve. User needs to ONLY specify geometry and naming
    for parts in desired_state_geom.

    The initial geometry state would be then specified by `Task`s, and RepairsSimState
    will be created by `translation.translate_compound_to_sim_state`. The environment will be added by"""

    STANDARD_ENV_SIZE = (64, 64, 64)
    BOTTOM_CENTER_OF_PLATE = (0, STANDARD_ENV_SIZE[1] / 2 + 20, 20)

    @abstractmethod
    def desired_state_geom(self) -> Compound:
        """
        Set up the simulation environment's desired final state. Should be subclassed, but not called by others. Call get_desired_state instead.

        Args:
            sim: The simulation object to be set up.
        """
        pass

    def get_default_b123d_env(self):
        return plate_env_bd_geometry()

    def get_default_genesis_scene(self):
        return genesis_setup()

    def _debug_render(self, scene: gs.Scene, camera_1: Camera, camera_2: Camera):
        render_and_save(scene, camera_1, camera_2)
