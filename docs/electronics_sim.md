# Electronics System Overview

This document describes the architecture and propagation model for the current electronics simulation system.

## Component Model

All electronic components implement a common interface:

- **propagate(voltage: float, current: float) -> (voltage: float, current: float)**

This method models the passage of both voltage and current through the component, allowing for realistic simulation of voltage drops, open/closed switches, and other circuit behaviors.

### Key Component Types

- **VoltageSource**: Sets the voltage at its node and provides current as determined by the circuit's total resistance.
- **Resistor**: Drops voltage according to Ohm's law (V = IR), passes current unchanged.
- **Wire**: Nearly ideal conductor with negligible resistance; passes current, drops almost no voltage.
- **Button (Switch)**: If closed, passes voltage/current; if open, blocks both (returns 0, 0).
- **Lightbulb**: Consumes power (P = VI), stores brightness for reporting.
- **Motor**: Consumes power, stores speed proportional to current.
- **Logic Gates (AND, OR, NOT)**: Pass voltage/current if logical output is 1, else block (return 0, 0).

## Simulation Flow

1. **Initialization**: Components are instantiated and connected to form a circuit graph.
2. **Propagation**: The simulator starts at the voltage source and propagates (voltage, current) pairs through each component using their `propagate` method.
3. **Consumption**: Consumer components (e.g., lightbulb, motor) use the propagated values to compute their output state (brightness, speed, etc.).

## Example

```
V_source -- Resistor -- Lightbulb

# Propagation:
# 1. VoltageSource.propagate(_, _) -> (V, I)
# 2. Resistor.propagate(V, I) -> (V - IR, I)
# 3. Lightbulb.propagate(V', I) -> (V', I)
# 4. Lightbulb uses V', I to compute brightness
```

## Extensibility

- The system is designed for easy extension: new components just need to implement the `propagate` interface.
- More complex behaviors (e.g., non-linear devices, dynamic switching) can be added by customizing the `propagate` logic.

## Testing

- All components are tested for correct voltage/current propagation and output state in a variety of circuit configurations.

---

For further details, see the code in `src/logic/electronics/` and the test suite in `tests/test_electronics.py`.
