from abc import abstractmethod
from dataclasses import field
from enum import IntEnum
from pathlib import Path

import torch
from tensordict import TensorClass


class ElectricalComponentInfo(TensorClass):
    """Base TensorClass for electrical components.

    This class intentionally avoids Python objects (e.g., names) to remain tensor-only.
    Per-type parameters should be stored in batched arrays (see ElectronicsComponentInfo).

    This class is meant to be batched only as len(number_components)
    """

    # Optional per-instance limits (prefer storing globally in ElectronicsComponentInfo)
    max_voltage: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    max_current: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )

    @property
    @abstractmethod
    def component_type(self) -> int:
        "Type from ElectricalComponentsEnum using IntEnum.value"
        raise NotImplementedError

    @property
    @abstractmethod
    def get_path(self) -> Path:
        """Get ElectricalComponent's path in `shared` folder. This is
        necessary for some components for geometry persistence."""
        raise NotImplementedError


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


class TerminalRoleEnum(IntEnum):
    """Standardized terminal roles used by the solver when stamping.

    Keep this minimal; roles are only used to define polarity and special cases.
    """

    TERM_A = 0
    TERM_B = 1
    POS = 2
    NEG = 3
    ANODE = 4
    CATHODE = 5
    COM = 6
    NO = 7
    NC = 8


class ControlTypeEnum(IntEnum):
    NONE = 0
    CURRENT_THRESHOLD = 1
    VOLTAGE_THRESHOLD = 2
    EXTERNAL_ACTION = 3
