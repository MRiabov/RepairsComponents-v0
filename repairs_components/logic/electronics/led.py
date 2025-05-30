from repairs_components.logic.electronics.component import ElectricalConsumer


class LED(ElectricalConsumer):
    """
    LED with forward-voltage threshold behavior.
    Brightness ∝ current once V > forward_voltage.
    TODO: add max-load test (if exceeded, LED should break).
    """

    def __init__(self, forward_voltage: float, name: str = None):
        super().__init__(name)
        self.forward_voltage = forward_voltage
        self.current = 0.0

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        # Block below threshold (conduct at or above forward voltage)
        if voltage < self.forward_voltage:
            self.current = 0.0
            return voltage, 0.0
        # Otherwise conduct and drop forward_voltage
        self.current = current
        return voltage - self.forward_voltage, current

    def use_current(self, voltage: float, current: float) -> dict:
        return {
            "type": "led",
            "brightness": self.current,
            "power": (voltage - self.forward_voltage) * self.current
            if voltage > self.forward_voltage
            else 0.0,
        }
