import pytest
from repairs_components.logic.electronics.voltage_source import VoltageSource
from repairs_components.logic.electronics.resistor import Resistor
from repairs_components.logic.electronics.motor import Motor
from repairs_components.logic.electronics.led import LED
from repairs_components.logic.electronics.button import Button
from repairs_components.logic.electronics.gates import AndGate, OrGate, NotGate
from repairs_components.logic.electronics.simulator import simulate_circuit
from repairs_components.logic.electronics.wire import Wire


# Test 1: Simple series circuit: VoltageSource -> Resistor -> LED
def test_series_led():
    v = VoltageSource(10, "v1")
    r = Resistor(5, "r1")
    l = LED(1.0, "l1")
    v.connect(r)
    r.connect(l)
    results = simulate_circuit([v, r, l])
    assert any(x["type"] == "led" and x["brightness"] > 0 for x in results)


# Test 2: Series circuit with Motor and LED
def test_series_motor_led():
    v = VoltageSource(12, "v1")
    m = Motor(6, "m1")
    l = LED(1.0, "l1")
    v.connect(m)
    m.connect(l)
    results = simulate_circuit([v, m, l])
    motor = next(x for x in results if x["type"] == "motor")
    bulb = next(x for x in results if x["type"] == "led")
    assert motor["speed"] > 0 and bulb["brightness"] > 0
    # Bulb should be dimmer than with no motor
    assert bulb["brightness"] < 12 / 6


# Test 3: Parallel circuit (approximate by two branches)
def test_parallel_leds():
    v = VoltageSource(9, "v1")
    l1 = LED(1.0, "l1")
    l2 = LED(1.0, "l2")
    # Both bulbs connected directly to voltage source
    v.connect(l1)
    v.connect(l2)
    results = simulate_circuit([v, l1, l2])
    assert sum(x["type"] == "led" for x in results) == 2


# Test 4: Circuit with Button (open/closed)
def test_button_opens_circuit():
    v = VoltageSource(5, "v1")
    b = Button(normally_closed=False, name="b1")
    l = LED(1.0, "l1")
    v.connect(b)
    b.connect(l)
    results = simulate_circuit([v, b, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] == 0


# Test 5: Logic gate controls bulb
def test_and_gate_controls_bulb():
    v = VoltageSource(5, "v1")
    g = AndGate(name="and1")
    l = LED(1.0, "l1")
    g.inputs = [1, 1]  # Both inputs True
    v.connect(g)
    g.connect(l)
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] > 0
    # Now one input False
    g.inputs = [1, 0]
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] == 0


# Test 6: NotGate inverts input
def test_not_gate_controls_bulb():
    v = VoltageSource(3, "v1")
    g = NotGate(name="not1")
    l = LED(1.0, "l1")
    g.input = 0  # Input False, output True
    v.connect(g)
    g.connect(l)
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] > 0
    g.input = 1  # Input True, output False
    results = simulate_circuit([v, g, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] == 0


# Test 7: Wire in series circuit
def test_wire_series():
    v = VoltageSource(5, "v1")
    w = Wire("w1")
    l = LED(1.0, "l1")
    v.connect(w)
    w.connect(l)
    results = simulate_circuit([v, w, l])
    bulb = next(x for x in results if x["type"] == "led")
    assert bulb["brightness"] > 0


# Test 8: Wires forming a parallel circuit
def test_wire_parallel():
    v = VoltageSource(6, "v1")
    w1 = Wire("w1")
    w2 = Wire("w2")
    l1 = LED(1.0, "l1")
    l2 = LED(1.0, "l2")
    v.connect(w1)
    v.connect(w2)
    w1.connect(l1)
    w2.connect(l2)
    results = simulate_circuit([v, w1, w2, l1, l2])
    bulbs = [x for x in results if x["type"] == "led"]
    assert len(bulbs) == 2
    assert all(b["brightness"] > 0 for b in bulbs)
