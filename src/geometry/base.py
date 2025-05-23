"""Base component class and core functionality for repair components.

This module provides base classes for creating components that can be used with the Genesis
physics simulator. Genesis supports both MJCF (MuJoCo) and UDRF (Unified Robot Description Format)
for model definition.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

import numpy as np
import genesis

if TYPE_CHECKING:
    from genesis.sim import Model, Data


class Component:
    def bd_geometry(self, moved_to: tuple[float, float, float]):
        pass
