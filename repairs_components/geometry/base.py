"""Base component class and core functionality for repair components.

This module provides base classes for creating components that can be used with the Genesis
physics simulator. Genesis supports both MJCF (MuJoCo) and UDRF (Unified Robot Description Format)
for model definition.
"""

from build123d import Part, Compound
from tensordict import TensorClass
from abc import abstractmethod


class Component(TensorClass):
    """Base class for all repair components.

    This class defines the common interface and functionality for all repair components.

    It is a TensorClass.
    """

    # required, but not always.
    def get_mjcf(self) -> str:
        """Get the MJCF representation of the component."""
        raise NotImplementedError

    @abstractmethod  # actually required
    def bd_geometry(self) -> Part | Compound:
        """Get the build123d geometry of the component."""
        raise NotImplementedError
