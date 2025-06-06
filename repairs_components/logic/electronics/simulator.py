from repairs_components.logic.electronics.resistor import Resistor
from repairs_components.logic.electronics.led import LED
from repairs_components.logic.electronics.voltage_source import VoltageSource
from repairs_components.logic.electronics.component import (
    ElectricalComponent,
    ElectricalConsumer,
)
import numpy as np


def simulate_circuit(components):
    """
    Simulate the circuit, returning output states for all consumers.
    Args:
        components: list of all components in the circuit
    Returns:
        List of dicts describing consumer outputs (e.g., lightbulb brightness, motor speed)
    """
    # For now, assume a single voltage source and series connection
    voltage_sources = filter(lambda c: isinstance(c, VoltageSource), components)
    if not voltage_sources:
        return []
    voltage = voltage_sources[0].voltage

    # Calculate voltage drops for threshold devices (LEDs)
    fixed_drop = sum(c.forward_voltage for c in components if isinstance(c, LED))
    resistive_components = [c for c in components if isinstance(c, Resistor)]
    total_resistance = sum(
        c.resistance for c in resistive_components
    )  # uuh, this isn't right. it's more complicated than a sum, it can be paralel!
    if total_resistance > 0:
        effective_voltage = max(voltage - fixed_drop, 0)
        total_current = effective_voltage / total_resistance
    else:
        # no resistors: allow unit current if supply meets threshold
        total_current = 1.0 if voltage >= fixed_drop else 0.0

    # Propagate voltage/current through the chain, recording consumer outputs
    v, i = voltage, total_current
    results = []
    for comp in components[1:]:  # skip voltage source at index 0
        v_in, i_in = v, i
        v, i = comp.propagate(v, i)
        if isinstance(comp, ElectricalConsumer):
            results.append(comp.use_current(v_in, i_in))
    return results
