from .component import ElectricalComponent, ElectricalConsumer
import numpy as np

def simulate_circuit(components):
    """
    Simulate the circuit, returning output states for all consumers.
    Args:
        components: list of all components in the circuit
    Returns:
        List of dicts describing consumer outputs (e.g., lightbulb brightness, motor speed)
    """
    consumers = [c for c in components if isinstance(c, ElectricalConsumer)]
    results = []
    # For now, assume a single voltage source and series connection
    voltage_sources = [c for c in components if hasattr(c, 'voltage')]
    if not voltage_sources:
        return []
    voltage = voltage_sources[0].voltage
    # Calculate total resistance
    resistive_components = [c for c in components if hasattr(c, 'resistance')]
    total_resistance = sum([c.resistance for c in resistive_components])
    total_current = voltage / total_resistance if total_resistance != 0 else 0
    # Update consumers
    for consumer in consumers:
        consumer.modify_current(voltage, total_current)
        results.append(consumer.use_current(voltage, total_current))
    return results
