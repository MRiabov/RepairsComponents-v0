from abc import abstractmethod
from enum import IntEnum
from pathlib import Path
from repairs_components.geometry.base import Component
import numpy as np


class ElectricalComponent(Component):
    def __init__(self, name: str, max_load: tuple[float, float] | None = None):
        # Use a list for connections; handle vectorization at the simulation level
        self.connected_to: list[ElectricalComponent] = []
        self.name: str = name  # all names must be unique, to be used later.
        self.max_load = max_load

    def connect(self, other: "ElectricalComponent"):
        self.connected_to.append(other)

    @abstractmethod
    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        pass

    @property
    @abstractmethod
    def component_type(self) -> int:
        "Type from ElectricalComponentsEnum using IntEnum.value"
        raise NotImplementedError

    @property
    @abstractmethod
    def get_path(self) -> Path:
        "Get ElectricalComponent's path in `shared` folder."
        raise NotImplementedError


class ElectricalConsumer(ElectricalComponent):
    @abstractmethod
    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        pass

    @abstractmethod
    def use_current(self, voltage: float, current: float) -> dict:
        pass


class ElectricalGate(ElectricalComponent):
    @abstractmethod
    def propagate(
        self, voltage: float, current: float, property
    ) -> tuple[float, float]:
        pass


class ElectricalComponentsEnum(IntEnum):
    CONNECTOR = 0
    # Note: After some thinking, I believe each wire with two connector ends should be its own component.
    # So connector connects to wire, and wire connects to other connector. And the connectors connect.
    # This is because encoding wire as a single component would create difficulties in modelling loose edges.

    WIRE = 1
    # And the wire has a tensor of points along which it is constrained.
    MOTOR = 2
    BUTTON = 3
    LED = 4
    RESISTOR = 5
    VOLTAGE_SOURCE = 6
