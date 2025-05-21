from abc import abstractmethod
from typing_extensions import Any


class PhysicalElectricalComponent:
    """A wrapper around a Genesis entity, representing a physical electrical component.
    Use it because of visualize_state"""

    name: str
    entity: Any  # I haven't found genesis.Entity yet.

    @abstractmethod
    def create_geometry(self):
        pass

    @abstractmethod
    def visualize_state(self, state: dict):
        pass
