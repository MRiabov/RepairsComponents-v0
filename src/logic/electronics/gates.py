from .component import ElectricalGate

class AndGate(ElectricalGate):
    def __init__(self):
        self.inputs = []
        self.output = None
    def modify_current(self, voltage: float, current: float, property=None):
        # property should be the input boolean values
        return int(all(self.inputs))

class OrGate(ElectricalGate):
    def __init__(self):
        self.inputs = []
        self.output = None
    def modify_current(self, voltage: float, current: float, property=None):
        return int(any(self.inputs))

class NotGate(ElectricalGate):
    def __init__(self):
        self.input = None
        self.output = None
    def modify_current(self, voltage: float, current: float, property=None):
        return int(not self.input)
