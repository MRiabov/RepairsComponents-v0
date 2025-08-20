"""Deprecated simulator shim.

This file used to contain a toy OOP-based simulator. The project has
transitioned to a tensorized MNA-based solver and per-type TensorClasses.

Use `repairs_components.logic.electronics.mna.solve_dc_once` instead.
"""

from .electronics_state import ElectronicsInfo, ElectronicsState  # noqa: F401


def simulate_circuit(*_args, **_kwargs):  # pragma: no cover - legacy API
    raise NotImplementedError(
        "simulate_circuit is deprecated. Use mna.solve_dc_once(component_info, state)."
    )
