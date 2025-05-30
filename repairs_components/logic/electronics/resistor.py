from .component import ElectricalComponent
import numpy as np


class Resistor(ElectricalComponent):
    def __init__(self, resistance: float, name: str):
        super().__init__(name=name)
        self.resistance = resistance
        self.connected_to = []

    def propagate(self, voltage: float, current: float):
        voltage_drop = current * self.resistance
        return voltage - voltage_drop, current
