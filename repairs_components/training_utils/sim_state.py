"Sim state definition (separate class due to circular import issues)"

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SimState(ABC):
    """Simply a convenience ABC to define the diff method."""

    @abstractmethod
    def diff(self, other: "SimState") -> tuple[dict[str, np.ndarray], int]:
        raise NotImplementedError
