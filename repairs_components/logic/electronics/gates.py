from .component import ElectricalGate


class AndGate(ElectricalGate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.inputs = []
        self.output = None

    def modify_current(self, voltage: float, current: float, property=None):
        # property should be the input boolean values
        return int(all(self.inputs))

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        self.output = int(all(self.inputs))
        return (voltage, current) if self.output else (0.0, 0.0)


class OrGate(ElectricalGate):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.inputs = []
        self.output = None

    def modify_current(self, voltage: float, current: float, property=None):
        return int(any(self.inputs))

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        self.output = int(any(self.inputs))
        return (voltage, current) if self.output else (0.0, 0.0)


class NotGate(
    ElectricalGate
):  # note: this is suspicious. `not self.input?` would it return 1? does it have other inputs?
    def __init__(self, name: str):
        super().__init__(name=name)
        self.input = None
        self.output = None

    def modify_current(self, voltage: float, current: float, property=None):
        return int(not self.input)

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        self.output = int(not self.input)
        return (voltage, current) if self.output else (0.0, 0.0)
