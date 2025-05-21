from .component import ElectricalComponent


class Wire(ElectricalComponent):
    "Convenience component for connecting/disconnecting other components"

    def __init__(self, name: str):
        super().__init__(name)

    def propagate(self, voltage: float, current: float) -> tuple[float, float]:
        return voltage, current
