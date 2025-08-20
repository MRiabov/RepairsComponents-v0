"Sim state definition (separate class due to circular import issues)"

from abc import abstractmethod
from typing import Any

import numpy as np
from tensordict import TensorClass


class SimState(TensorClass):
    """Simply a convenience ABC to define the diff method.
    Is a tensorclass."""

    @abstractmethod
    def diff(self, other: "SimState", info: Any) -> tuple[dict[str, np.ndarray], int]:
        raise NotImplementedError
