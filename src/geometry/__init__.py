"""Repairs Components - A library of repair components for reinforcement learning.

This package provides a collection of reusable, physics-based repair components
for building realistic repair and maintenance simulations with MuJoCo.
"""

from repairs_components.base import Component
from repairs_components.fasteners import Screw
from repairs_components.sockets import BasicSocket, LockingSocket
from repairs_components.controls import Button

__version__ = "0.1.0"
__all__ = ["Component", "Screw", "BasicSocket", "LockingSocket", "Button"]
