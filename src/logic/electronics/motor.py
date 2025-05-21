from .component import ElectricalConsumer
import numpy as np

class Motor(ElectricalConsumer):
    def __init__(self, resistance: float = 10.0):
        self.resistance = resistance
        self.speed = 0.0
        self.connected_to = []
    def modify_current(self, voltage: float, current: float):
        # For a simple model, treat as a resistor
        return voltage / self.resistance if self.resistance != 0 else np.inf
    def use_current(self, voltage: float, current: float):
        self.speed = current  # Simplified: speed proportional to current
        if callback:
            callback(self, voltage, current)
        return {'type': 'motor', 'speed': self.speed, 'power': voltage * current}
