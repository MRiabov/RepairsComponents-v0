from typing import Dict, Optional, Tuple


class FluidState:
    """Tracks fluid presence using index-based sensors."""

    def __init__(
        self,
        positions: Dict[int, Tuple[float, float, float]],
        present: Optional[Dict[int, bool]] = None,
    ):
        # Map sensor index to position
        self.positions = positions
        # Map sensor index to presence
        self.present = {i: False for i in positions} if present is None else present

    def set_presence(self, index: int, present: bool) -> None:
        """Mark sensor at index as having fluid present or absent."""
        if index not in self.positions:
            raise KeyError(f"Unknown sensor index: {index}")
        self.present[index] = present

    def diff(self, other: "FluidState") -> Tuple[Dict[int, Dict[str, bool]], int]:
        """Compute differences in fluid presence between two states.

        Returns a dict mapping index to {'from': bool, 'to': bool}, and total changes count.
        """
        changes: Dict[int, Dict[str, bool]] = {}
        total = 0
        for idx, state in self.present.items():
            other_state = other.present.get(idx, False)
            if state != other_state:
                changes[idx] = {"from": state, "to": other_state}
                total += 1
        return changes, total

    def register_sensor(
        self, index: int, position: Tuple[float, float, float], present: bool = False
    ):
        """Add a new sensor with index and position."""
        if index in self.positions:
            raise KeyError(f"Sensor {index} already registered")
        self.positions[index] = position
        self.present[index] = present
