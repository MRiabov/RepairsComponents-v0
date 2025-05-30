from .component import ElectricalComponent
import numpy as np


class VoltageSource(ElectricalComponent):
    def __init__(self, voltage: float, name: str):
        super().__init__(name)
        self.voltage = voltage
        self.connected_to = []

    def propagate(self, voltage: float, current: float):
        # As the source, set voltage to self.voltage, current is determined by load
        return self.voltage, current
