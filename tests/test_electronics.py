import pytest
from src.logic.electronics.voltage_source import VoltageSource
from src.logic.electronics.resistor import Resistor
from src.logic.electronics.motor import Motor
from src.logic.electronics.lightbulb import Lightbulb
from src.logic.electronics.button import Button
from src.logic.electronics.gates import AndGate, OrGate, NotGate
from src.logic.electronics.simulator import simulate_circuit

# Test 1: Simple series circuit: VoltageSource -> Resistor -> Lightbulb
def test_series_lightbulb():
    v = VoltageSource(10)
    r = Resistor(5)
    l = Lightbulb(10)
    v.connect(r)
    r.connect(l)
    results = simulate_circuit([v, r, l])
    assert any(x['type'] == 'lightbulb' and x['brightness'] > 0 for x in results)

# Test 2: Series circuit with Motor and Lightbulb
def test_series_motor_lightbulb():
    v = VoltageSource(12)
    m = Motor(6)
    l = Lightbulb(6)
    v.connect(m)
    m.connect(l)
    results = simulate_circuit([v, m, l])
    motor = next(x for x in results if x['type'] == 'motor')
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert motor['speed'] > 0 and bulb['brightness'] > 0
    # Bulb should be dimmer than with no motor
    assert bulb['brightness'] < 12/6

# Test 3: Parallel circuit (approximate by two branches)
def test_parallel_lightbulbs():
    v = VoltageSource(9)
    l1 = Lightbulb(9)
    l2 = Lightbulb(9)
    # Both bulbs connected directly to voltage source
    v.connect(l1)
    v.connect(l2)
    results = simulate_circuit([v, l1, l2])
    assert sum(x['type'] == 'lightbulb' for x in results) == 2

# Test 4: Circuit with Button (open/closed)
def test_button_opens_circuit():
    v = VoltageSource(5)
    b = Button(closed=False)
    l = Lightbulb(5)
    v.connect(b)
    b.connect(l)
    results = simulate_circuit([v, b, l])
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert bulb['brightness'] == 0

# Test 5: Logic gate controls bulb
def test_and_gate_controls_bulb():
    v = VoltageSource(5)
    g = AndGate()
    l = Lightbulb(5)
    g.inputs = [1, 1]  # Both inputs True
    v.connect(g)
    g.connect(l)
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert bulb['brightness'] > 0
    # Now one input False
    g.inputs = [1, 0]
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert bulb['brightness'] == 0

# Test 6: NotGate inverts input
def test_not_gate_controls_bulb():
    v = VoltageSource(3)
    g = NotGate()
    l = Lightbulb(3)
    g.input = 0  # Input False, output True
    v.connect(g)
    g.connect(l)
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert bulb['brightness'] > 0
    g.input = 1  # Input True, output False
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x['type'] == 'lightbulb')
    assert bulb['brightness'] == 0
