from .component import ElectricalComponent


class Button(ElectricalComponent):
    def __init__(self, normally_closed: bool, name: str):
        super().__init__(name)
        self.closed = normally_closed

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        if self.closed:
            return voltage, current
        else:
            return 0.0, 0.0
