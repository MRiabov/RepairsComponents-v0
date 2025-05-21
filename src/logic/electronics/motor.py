from .component import ElectricalConsumer
import numpy as np


class Motor(ElectricalConsumer):
    def __init__(self, resistance: float, name: str):
        super().__init__(name)
        self.resistance = resistance
        self.speed = 0.0

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        self.speed = current
        return voltage, current

    def use_current(self, voltage: float, current: float) -> dict:
        return {"type": "motor", "speed": self.speed, "power": voltage * current}
