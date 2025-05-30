import pytest
from repairs_components.logic.electronics.led import LED
from repairs_components.logic.electronics.voltage_source import VoltageSource
from repairs_components.logic.electronics.resistor import Resistor
from repairs_components.logic.electronics.simulator import simulate_circuit

# Test LED below forward-voltage threshold


def test_led_below_threshold():
    v = VoltageSource(1.5, "v1")
    r = Resistor(100, "r1")
    led = LED(2.0, name="led1")
    v.connect(r)
    r.connect(led)
    results = simulate_circuit([v, r, led])
    assert results[0]["brightness"] == pytest.approx(0.0)


# Test LED above threshold


def test_led_above_threshold():
    v = VoltageSource(5.0, "v1")
    r = Resistor(100, "r1")
    led = LED(2.0, name="led2")
    v.connect(r)
    r.connect(led)
    results = simulate_circuit([v, r, led])
    # I = (5 - 2)/100 = 0.03 A
    assert results[0]["brightness"] == pytest.approx(0.03, rel=1e-6)


# Test non-linear behavior (forward-voltage creates non-linear I-V)


def test_led_nonlinear_behavior():
    def run(v_supply):
        v = VoltageSource(v_supply, f"v{v_supply}")
        r = Resistor(100, "r1")
        led = LED(2.0, "led")
        v.connect(r)
        r.connect(led)
        return simulate_circuit([v, r, led])[0]["brightness"]

    b5 = run(5.0)
    b6 = run(6.0)
    assert b6 > b5
    # Not strictly proportional to supply: (b6/b5) != (6/5)
    assert abs((b6 / b5) - (6 / 5)) > 0.01
