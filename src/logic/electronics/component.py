from abc import ABC, abstractmethod
import numpy as np


class ElectricalComponent(ABC):
    def __init__(self, name: str):
        # Use a list for connections; handle vectorization at the simulation level
        self.connected_to: list[ElectricalComponent] = []
        self.name: str = name  # all names must be unique, to be used later.
        self.state = None  # Optional: can be used for output, logic state, etc.

    def connect(self, other: "ElectricalComponent"):
        self.connected_to.append(other)

    @abstractmethod
    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        pass


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
