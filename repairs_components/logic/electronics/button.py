from .electronics_control import (
    SwitchInfo as ButtonInfo,  # noqa: F401 - re-export for compatibility
)
from .electronics_control import (
    SwitchState as ButtonState,  # noqa: F401 - re-export for compatibility
)

__all__ = ["ButtonInfo", "ButtonState"]
