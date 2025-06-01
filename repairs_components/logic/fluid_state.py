from dataclasses import dataclass, field


@dataclass
class FluidState:
    """Tracks fluid presence using index-based sensors."""

    positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    present: dict[int, bool] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize present dict if not provided
        if not self.present:
            self.present = {i: False for i in self.positions}

        # No need for to_dict() - use dataclasses.asdict() instead

    def set_presence(self, index: int, present: bool) -> None:
        """Mark sensor at index as having fluid present or absent."""
        if index not in self.positions:
            raise KeyError(f"Unknown sensor index: {index}")
        self.present[index] = present

    def diff(self, other: "FluidState") -> tuple[dict[int, dict[str, bool]], int]:
        """Compute differences in fluid presence between two states.

        Returns a dict mapping index to {'from': bool, 'to': bool}, and total changes count.
        """
        changes: dict[int, dict[str, bool]] = {}
        total = 0
        for idx, state in self.present.items():
            other_state = other.present.get(idx, False)
            if state != other_state:
                changes[idx] = {"from": state, "to": other_state}
                total += 1
        return changes, total

    def register_sensor(
        self, index: int, position: tuple[float, float, float], present: bool = False
    ):
        """Add a new sensor with index and position."""
        if index in self.positions:
            raise KeyError(f"Sensor {index} already registered")
        self.positions[index] = position
        self.present[index] = present
