from dataclasses import dataclass, field
from typing import Dict, List
from repairs_components.logic.electronics.component import ElectricalComponent


@dataclass
class ElectronicsState:
    components: dict[str, ElectricalComponent] = field(default_factory=dict)

    # No need for to_dict() - use dataclasses.asdict() instead

    def diff(
        self, other: "ElectronicsState"
    ) -> tuple[Dict[str, Dict[str, List[str]]], int]:
        """Compute per-component connection differences and total change count.

        Args:
            other (ElectronicsState): state to compare against.

        Returns:
            - connection_differences: mapping component names to {'added', 'removed'} lists
            - total_changes: int count of all added+removed connections
        """
        # Get component maps
        self_map = self.components
        other_map = other.components
        conn_diff: Dict[str, Dict[str, List[str]]] = {}
        total_changes = 0
        for name, comp in self_map.items():
            self_conn = {n.name for n in comp.connected_to}
            other_conn = {n.name for n in other_map[name].connected_to}
            added = list(other_conn - self_conn)
            removed = list(self_conn - other_conn)
            if added or removed:
                conn_diff[name] = {"added": added, "removed": removed}
                total_changes += len(added) + len(removed)
        return conn_diff, total_changes

    def register(self, component: ElectricalComponent):
        """Register a new electrical component."""
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        self.components[component.name] = component

    # def register_contacts(self, contacts: dict[str, tuple[str, str]]):
    #     "Register components of body A to bodies B"

    def connect(self, name: str, other_name: str):
        "Connect two components"
        self.components[name].connect(self.components[other_name])
        self.components[other_name].connect(self.components[name])

    def clear_connections(self):
        "Disconnect all connections between a component"
        for component in self.components.values():
            component.connected_to = []  # just clear everything.
