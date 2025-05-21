from .component import ElectricalComponent
import numpy as np

class VoltageSource(ElectricalComponent):
    def __init__(self, voltage: float):
        self.voltage = voltage
        self.connected_to = []
    def modify_current(self, voltage: float, current: float):
        return self.voltage
