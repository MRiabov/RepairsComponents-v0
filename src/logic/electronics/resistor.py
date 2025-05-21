from .component import ElectricalComponent
import numpy as np

class Resistor(ElectricalComponent):
    def __init__(self, resistance: float):
        self.resistance = resistance
        self.connected_to = []
    def modify_current(self, voltage: float, current: float):
        # Ohm's law: V = IR
        return voltage / self.resistance if self.resistance != 0 else np.inf
