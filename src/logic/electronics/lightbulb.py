from .component import ElectricalConsumer


class Lightbulb(ElectricalConsumer):
    def __init__(self, resistance: float, name):
        super().__init__(name)
        self.resistance = resistance
        self.brightness = 0.0

    def modify_current(self, voltage: float, current: float) -> float:
        # Ohm's law: V = IR
        return voltage / self.resistance if self.resistance != 0 else float("inf")

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        # Store brightness for later use
        self.brightness = voltage * current
        return voltage, current

    def use_current(self, voltage: float, current: float) -> dict:
        return {
            "type": "lightbulb",
            "brightness": self.brightness,
            "power": voltage * current,
        }
