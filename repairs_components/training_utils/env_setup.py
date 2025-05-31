from abc import ABC, abstractmethod
from typing import Tuple
from build123d import Compound, Part, RevoluteJoint
from repairs_components.geometry.fasteners import Fastener
from repairs_components.geometry.base import Component
import numpy as np
from genesis.vis.camera import Camera
from repairs_components.training_utils.sim_state_global import RepairsSimState
import genesis as gs
from repairs_components.geometry.base_env.tooling_stand_plate import (
    genesis_setup,
    plate_env_bd_geometry,
    render_and_save,
)


class EnvSetup(ABC):
    "A setup for a single problem to solve"

    STANDARD_ENV_SIZE = (64, 64, 64)
    BOTTOM_CENTER_OF_PLATE = (0, STANDARD_ENV_SIZE[1] / 2 + 20, 20)

    @abstractmethod
    def starting_state(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray], list[Camera]]:
        """
        Set up the simulation environment.

        Args:
            sim: The simulation object to be set up.
        """
        pass

    @abstractmethod
    def desired_state(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray]]:
        """
        Set up the simulation environment's desired final state.

        Args:
            sim: The simulation object to be set up.
        """
        pass

    def get_default_randomized_environment(self):
        """Get a default randomized environment for the simulation."""
        raise NotImplementedError  # TODO

    def get_default_b123d_env(self):
        return plate_env_bd_geometry()

    def get_default_genesis_scene(self):
        return genesis_setup()

    def _debug_render(self, camera_1: Camera, camera_2: Camera):
        render_and_save(camera_1, camera_2)
