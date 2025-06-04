"""
Holds state of assembly components:
- Fastener connections: bodies attached per fastener
- Rigid bodies: absolute positions & rotations

Provides diff methods:
- _fastener_diff: connection changes per fastener
- _body_diff: transform changes per body

diff(): combines both into {'fasteners', 'bodies'} with total change count
"""

from dataclasses import dataclass, field

import scipy
import torch
from repairs_components.geometry.fasteners import Fastener
import math


@dataclass
class PhysicalState:
    fasteners: list[Fastener] = field(default_factory=list)
    connected_parts: dict[str, tuple[str, str] | str | None] = field(
        default_factory=dict
    )
    joint_names: dict[str, tuple[str, str]] = field(default_factory=dict)
    """Two joint names of fasteners."""

    positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    "Absolute positions of bodies"
    rotations: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    "Absolute rotations of bodies"
    used_tool: str = "gripper"

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
        assert name not in self.positions, f"Body {name} already registered"
        assert position[2] >= 0, (
            f"Body {name} is below the base plate. Position: {position}"
        )
        self.positions[name] = position
        self.rotations[name] = rotation

    def connect(self, fastener_name: str, body_a: str, body_b: str | None = None):
        current_fastener_state = self.connected_parts[fastener_name]
        # Check if the current state is not a tuple (i.e., it's either None or a string)
        assert current_fastener_state is None or isinstance(
            current_fastener_state, str
        ), "Cannot connect - fastener is already connected to two bodies"
        self.connected_parts[fastener_name] = (
            (body_a, body_b) if body_b is not None else body_a
        )

    def disconnect(self, fastener_name: str, disconnected_body: str):
        current_fastener_state = self.connected_parts[fastener_name]
        # Check if the current state is a tuple (connected to two bodies)
        assert isinstance(current_fastener_state, tuple), (
            "Cannot disconnect - fastener is not connected to two bodies"
        )

        body_a, body_b = current_fastener_state
        if body_a == disconnected_body:
            self.connected_parts[fastener_name] = body_b
        elif body_b == disconnected_body:
            self.connected_parts[fastener_name] = body_a
        else:
            raise ValueError(
                f"Body {disconnected_body} is not connected to fastener {fastener_name}"
            )

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
                print("pos", pos, "other_pos", other_pos, pos_threshold)

                if torch.dist(
                    pos, torch.tensor(other_pos, device=pos.device)
                ) > torch.tensor(pos_threshold, device=pos.device):
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
