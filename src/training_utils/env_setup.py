from abc import ABC, abstractmethod
from typing import Tuple
from build123d import Compound, Part, RevoluteJoint
from src.geometry.fasteners import Fastener
from src.geometry.base import Component
import numpy as np
from genesis.vis.camera import Camera
from src.training_utils.sim_state import RepairsSimState
import genesis as gs


class EnvSetup(ABC):
    "A setup for a single problem to solve"

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

    def create_state(self, b123d_compound: Compound) -> RepairsSimState:
        "Get RepairsSimState from the b123d_compound, i.e. translate from build123d to RepairsSimState."
        sim_state = RepairsSimState()
        for part in b123d_compound.descendants:
            part: Part

            if part.label:  # only parts with labels are expected.
                assert "@" in part.label, "part must annotate type."
                # physical state
                if part.label.endswith("@solid"):
                    sim_state.physical_state.register_body(
                        name=part.label,
                        position=part.position.to_tuple(),
                        rotation=part.rotation.to_tuple(),
                    )
                elif part.label.endswith(
                    "@fastener"
                ):  # collect constraints, get labels of bodies,
                    # collect constraints (in build123d named joints)
                    joint_a: RevoluteJoint = part.joints["fastener_joint_a"]
                    joint_b: RevoluteJoint = part.joints["fastener_joint_b"]
                    joint_tip: RevoluteJoint = part.joints["fastener_joint_tip"]

                    # are active
                    constraint_a_active = joint_a.connected_to is not None
                    constraint_b_active = joint_b.connected_to is not None

                    # if active, get connected to names
                    initial_body_a = (
                        joint_a.connected_to.parent if constraint_a_active else None
                    )
                    initial_body_b = (
                        joint_b.connected_to.parent if constraint_b_active else None
                    )

                    # collect names of bodies(?)
                    assert initial_body_a.label and initial_body_a.label, (
                        "Constrained parts must be labeled"
                    )
                    sim_state.physical_state.register_fastener(
                        Fastener(
                            initial_body_a=initial_body_a.label
                            if constraint_a_active
                            else None,
                            initial_body_b=initial_body_b.label
                            if constraint_b_active
                            else None,
                            constraint_a_active=constraint_a_active,
                            constraint_b_active=constraint_b_active,
                            name=part.label,
                        )
                    )
