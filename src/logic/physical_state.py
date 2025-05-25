"""
Holds state of assembly components:
- Fastener connections: bodies attached per fastener
- Rigid bodies: absolute positions & rotations

Provides diff methods:
- _fastener_diff: connection changes per fastener
- _body_diff: transform changes per body

diff(): combines both into {'fasteners', 'bodies'} with total change count
"""

from types import NoneType
from src.geometry.fasteners import Fastener
import math


class PhysicalState:
    def __init__(self):
        self.fasteners = []
        self.connected_parts: dict[str, tuple[str, str] | str | None] = {}
        self.joint_names: dict[
            str, tuple[str, str]
        ] = {}  # two joint names of fasteners fasteners.

        # Rigid body absolute transforms
        self.positions: dict[str, tuple[float, float, float]] = {}
        self.rotations: dict[str, tuple[float, float, float, float]] = {}

    def register_fastener(self, fastener: Fastener):
        self.fasteners.append(fastener)
        self.connected_parts.update(
            {fastener.name: (fastener.initial_body_a, fastener.initial_body_b)}
        )

    def register_body(
        self,
        name: str,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ):
        """Register a rigid body with position and rotation."""
        self.positions[name] = position
        self.rotations[name] = rotation

    def connect(self, fastener_name: str, body_a: str, body_b: str):
        current_fastener_state = self.connected_parts[fastener_name]
        assert isinstance(current_fastener_state, (NoneType, str))  # not a tuple
        self.connected_parts.update({fastener_name: (body_a, body_b)})

    def disconnect(self, fastener_name: str, disconnected_body: str):
        current_fastener_state = self.connected_parts[fastener_name]
        assert isinstance(current_fastener_state, (str, tuple[str, str]))
        other_body = (
            current_fastener_state[1]
            if disconnected_body == current_fastener_state[0]
            else current_fastener_state[0]
        )
        self.connected_parts.update({fastener_name: other_body})

    def _fastener_diff(
        self, other: "PhysicalState"
    ) -> tuple[dict[str, dict[str, list[str]]], int]:
        """Compute per-fastener connection differences.

        Returns:
            Tuple of:
            - conn_diff: mapping fastener name -> {'added', 'removed'} lists
            - total_changes: count of all connection changes
        """
        conn_diff: dict[str, dict[str, list[str]]] = {}
        total_changes = 0
        for name, state in self.connected_parts.items():
            other_state = other.connected_parts.get(name)
            if state is None:
                self_set = set()
            elif isinstance(state, str):
                self_set = {state}
            else:
                self_set = set(state)
            if other_state is None:
                other_set = set()
            elif isinstance(other_state, str):
                other_set = {other_state}
            else:
                other_set = set(other_state)
            added = list(other_set - self_set)
            removed = list(self_set - other_set)
            if added or removed:
                conn_diff[name] = {"added": added, "removed": removed}
                total_changes += len(added) + len(removed)
        return conn_diff, total_changes

    def _body_diff(
        self,
        other: "PhysicalState",
        deg_threshold: float = 5.0,
        pos_threshold: float = 3.0,
    ) -> tuple[dict[str, dict[str, tuple]], int]:
        """Compute differences in rigid body positions and rotations.

        Ignores position changes below pos_threshold units and rotation changes below deg_threshold degrees.

        Args:
            other (PhysicalState): state to compare against.
            deg_threshold (float): angle threshold in degrees.
            pos_threshold (float): positional threshold.

        Returns:
            Tuple of (body_diff, total_changes).
            body_diff[name] = {'position': (pos, other_pos), 'rotation': (rot, other_rot)}
        """
        body_diff: dict[str, dict[str, tuple]] = {}
        total_changes = 0
        for name, pos in self.positions.items():
            other_pos = other.positions.get(name)
            other_rot = other.rotations.get(name)
            changes: dict[str, tuple] = {}
            # position diff
            if other_pos is not None:  # math.dist - distance.
                if math.dist(pos, other_pos) > pos_threshold:
                    changes["position"] = (pos, other_pos)
                    total_changes += 1
            # rotation diff
            rot = self.rotations.get(name)
            if rot is not None and other_rot is not None:
                # quaternion angle diff
                dot = abs(sum(a * b for a, b in zip(rot, other_rot)))
                angle = 2 * math.degrees(math.acos(min(1.0, max(-1.0, dot))))
                if angle > deg_threshold:
                    changes["rotation"] = (rot, other_rot)
                    total_changes += 1
            if changes:
                body_diff[name] = changes
        return body_diff, total_changes

    def diff(self, other: "PhysicalState") -> tuple[dict[str, dict], int]:
        """Compute combined fastener and rigid body differences and total changes."""
        fast_diff, fast_changes = self._fastener_diff(other)
        body_diff, body_changes = self._body_diff(other)
        return {
            "fasteners": fast_diff,
            "bodies": body_diff,
        }, fast_changes + body_changes
