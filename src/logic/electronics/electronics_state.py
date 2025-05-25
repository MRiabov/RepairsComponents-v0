from src.logic.electronics.component import ElectricalComponent
from typing import Dict, List


class ElectronicsState:
    def __init__(self, components: dict[str, list[ElectricalComponent]] = {}):
        self.components = components

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
        # Flatten name->component maps
        self_map: Dict[str, ElectricalComponent] = {
            c.name: c for comps in self.components.values() for c in comps
        }
        other_map: Dict[str, ElectricalComponent] = {
            c.name: c for comps in other.components.values() for c in comps
        }
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
        if component.component_type in self.components:
            assert component.name not in self.components[component.component_type], (
                f"Component {component.name} already registered"
            )
            self.components[component.component_type].append(component)
        else:
            self.components[component.component_type] = [component]

    def register_contacts(self, contacts: dict[str, tuple[str, str]]):
        "Register components of body A to bodies B"
