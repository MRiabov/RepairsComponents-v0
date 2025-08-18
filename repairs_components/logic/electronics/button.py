from .electronics_control import (
    SwitchInfo as ButtonInfo,  # noqa: F401 - re-export for compatibility
    SwitchState as ButtonState,  # noqa: F401 - re-export for compatibility
)

__all__ = ["ButtonInfo", "ButtonState"]
