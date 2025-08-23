from dataclasses import dataclass, field

import torch
from tensordict import TensorClass
from torch_geometric.data import Data

from repairs_components.logic.electronics.component import (
    ControlTypeEnum,
    ElectricalComponentsEnum,
    TerminalRoleEnum,
)
from repairs_components.logic.electronics.electronics_control import (
    SwitchInfo,
    SwitchState,
)
from repairs_components.logic.electronics.led import LedInfo, LedState
from repairs_components.logic.electronics.motor import MotorInfo, MotorState
from repairs_components.logic.electronics.resistor import ResistorInfo
from repairs_components.logic.electronics.voltage_source import VoltageSourceInfo

# Per-type Info/State classes are defined in their component modules and imported above.


@dataclass
class ElectronicsInfo:
    """Singleton information about all ElectricalComponentInfo components. Never has a batch dimension.

    Co-dependent with ElectronicsState, where ElectronicsInfo represents singleton info and ElectronicsState represents batch info.

    Holds ElectricalComponentInfo which represents per-component-type info."""

    component_type: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    component_id: torch.Tensor = field(
        default_factory=lambda: torch.empty((0), dtype=torch.long)
    )
    max_voltage: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    max_current: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    terminals_per_component: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    # Terminal mapping
    terminal_to_component_batch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """terminal_to_component_batch maps each terminal id in [0, T) to owning component index in [0, N). 
    Simply: Terminal indices per component in the batch. Shape: sum(terminals_per_component)"""

    # Optional Python-side name-index mappings (like PhysicalState)
    # Note that torch.stack(tensorclass(dict)) makes dicts list, so always use __post_init__ to unbatch them after stacking.
    component_indices_from_name: dict = field(default_factory=dict)
    inverse_component_indices: dict = field(default_factory=dict)
    has_electronics: bool = False
    "A flag that when False may prevent computation of electronics for speed."

    # ------------------------------------------------------------------
    # Derived terminal indexing helpers
    # ------------------------------------------------------------------
    component_first_terminal_index: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Prefix sum over terminals_per_component to slice terminals of component i quickly."""

    terminal_role: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Role per terminal (enum TerminalRoleEnum). Shape: [T]."""
    # Component parameters moved into type-specific ComponentInfo tensorclasses

    # Type-specific singleton component infos (unbatched over envs; sized by count per type)
    resistors: ResistorInfo = field(default_factory=ResistorInfo)
    vsources: VoltageSourceInfo = field(default_factory=VoltageSourceInfo)
    switches: SwitchInfo = field(default_factory=SwitchInfo)
    leds: LedInfo = field(default_factory=LedInfo)
    motors: MotorInfo = field(default_factory=MotorInfo)

    @property
    def num_terminals(self):
        return self.terminal_to_component_batch.shape[0]


class ElectronicsState(TensorClass, tensor_only=True):
    """TensorClass-based electronics state.

    Represents connectivity in an electronics state.
    Note: should always be initiated with tensor_only=True."""

    net_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    """Per-env terminal-to-net assignments. Varies over batches. 
    Values: -1 for unassigned else non-negative net id."""
    # Dynamic per-env fields used by solver and measurements
    state_bits: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.bool)
    )
    terminal_voltage: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float32)
    )
    component_branch_current: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float32)
    )
    # Per-batch dynamic state for control components
    switch_state: SwitchState = field(default_factory=SwitchState)
    # Per-batch dynamic outputs for effectors
    led_state: LedState = field(default_factory=LedState)
    motor_state: MotorState = field(default_factory=MotorState)

    def clear_all_connections(self) -> "ElectronicsState":
        """Reset all terminal nets to -1 (no connectivity)."""
        device = self.device
        # Single execution path: net_ids is always [B, T]
        B, T = self.net_ids.shape
        self.net_ids = torch.full((B, T), -1, dtype=torch.long, device=device)
        return self

    def _build_component_edges_from_nets(
        self, component_info: ElectronicsInfo
    ) -> torch.Tensor:
        """Derive undirected component-level edges from terminal net assignments.

        Returns:
            edge_index [2, E] for single-env.
        Note: If batched, this currently supports only single-env and will raise.
        """
        assert component_info.terminal_to_component_batch.numel() > 0, (
            "Register components first"
        )
        # Operate on the first batch element consistently
        net = self.net_ids[0]
        T = component_info.terminal_to_component_batch.shape[0]
        assert net.shape[0] == T, "net_id length must equal number of terminals"
        # Build nets -> components incidence without Python loops
        valid_mask = net >= 0
        if not valid_mask.any():
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        valid_nets = net[valid_mask]
        comps_for_valid_terms = component_info.terminal_to_component_batch[valid_mask]
        # Compact net ids to [0, G)
        nets_unique, net_compact = torch.unique(valid_nets, return_inverse=True)
        G = int(nets_unique.numel())
        if G == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        N = int(component_info.component_id.numel())
        # M[g, n] = 1 if component n participates in net g
        M = torch.zeros((G, N), dtype=torch.int32, device=self.device)
        M[net_compact, comps_for_valid_terms] = 1
        # Co-occurrence across nets: A[i, j] = number of nets where i and j co-occur
        A = M.t().matmul(M)
        A.fill_diagonal_(0)
        if (A > 0).any():
            iu, iv = torch.triu_indices(N, N, offset=1, device=self.device)
            mask = A[iu, iv] > 0
            if mask.any():
                u = iu[mask]
                v = iv[mask]
                return torch.stack([u, v], dim=0).contiguous()
        return torch.empty((2, 0), dtype=torch.long, device=self.device)

    def export_graph(self, component_info: ElectronicsInfo) -> Data:
        """Export a torch_geometric.Data graph for ML consumers.

        Node features: [max_voltage, max_current, component_type, component_id]
        Edge index: derived from terminal net connectivity at component level.
        """
        edge_index = self._build_component_edges_from_nets(component_info)

        mv = component_info.max_voltage.to(torch.float32)
        mc = component_info.max_current.to(torch.float32)
        component_type = component_info.component_type.to(torch.float32)
        cid = component_info.component_id.to(torch.float32)
        x = torch.stack([mv, mc, component_type, cid], dim=1).bfloat16()

        graph = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
        return graph

    def diff(
        self,
        other: "ElectronicsState",
        component_info: ElectronicsInfo,
    ) -> tuple[Data, int]:
        """Compute a graph diff between two electronics states (component-level edges).
        Note: component info is expected to be the same for both states, since we only ever compare electronic states from one env setup.

        Returns a PyG Data with:
            - x: node features (from self)
            - edge_index: combined edges from both states (union)
            - node_mask: bool mask of nodes whose scalar features differ
            - edge_mask: bool mask of edges that were added/removed/changed
        and an integer count of total differences (nodes + edges).
        """
        # Assert comparability: net_id shapes match terminal count and indices are in bounds.
        assert component_info.terminal_to_component_batch.ndim == 1, (
            "terminal_to_component_batch must be [T]"
        )
        T_total = component_info.terminal_to_component_batch.shape[0]
        assert self.net_ids.ndim == 2 and other.net_ids.ndim == 2, (
            "net_ids must be [B, T]"
        )
        B_self, T_self = self.net_ids.shape
        B_other, T_other = other.net_ids.shape
        assert B_self == B_other, "Batch dim mismatch between electronics states."
        assert T_self == T_other == T_total, (
            "Terminal count mismatch between electronics states and component info."
        )
        # Bounds: -1 indicates unassigned; otherwise in [0, T_total)
        assert torch.all((self.net_ids >= -1) & (self.net_ids < T_total)), (
            "self.net_ids indices out of bounds"
        )
        assert torch.all((other.net_ids >= -1) & (other.net_ids < T_total)), (
            "other.net_ids indices out of bounds"
        )

        # Build edge sets using provided component_info
        # Edges derive from terminal-to-component mapping and current net IDs.
        e_self = self._build_component_edges_from_nets(component_info)
        e_other = other._build_component_edges_from_nets(component_info)

        def _edges_to_set(ei: torch.Tensor) -> set[tuple[int, int]]:
            return set(map(tuple, ei.t().tolist())) if ei.numel() > 0 else set()

        set_a = _edges_to_set(e_self)
        set_b = _edges_to_set(e_other)
        all_edges_sorted = sorted(list(set_a.union(set_b)))
        edge_index = (
            torch.tensor(all_edges_sorted, dtype=torch.long, device=self.device).t()
            if all_edges_sorted
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )

        # set as no difference at the moment (before simulator incorporation.)
        node_mask = torch.zeros(
            (self.net_ids.shape[0],), dtype=torch.bool, device=self.device
        )

        # Edge mask: edges in symmetric difference
        set_sym = set_a.symmetric_difference(set_b)
        edge_mask = (
            torch.tensor(
                [tuple(e) in set_sym for e in all_edges_sorted],
                dtype=torch.bool,
                device=self.device,
            )
            if all_edges_sorted
            else torch.empty((0,), dtype=torch.bool, device=self.device)
        )

        diff_graph = Data(
            # x=torch.stack(
            #     [mv_a, mc_a, ct_a.to(torch.float32), id_a.to(torch.float32)], dim=1
            # ).bfloat16(),
            x=torch.zeros(
                (component_info.component_id.shape[0], 4),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            num_nodes=len(component_info.component_indices_from_name),
        )

        n_diffs = int(node_mask.sum().item() + edge_mask.sum().item())
        return diff_graph, n_diffs

    def diff_to_dict(self, diff_graph: Data, component_info: ElectronicsInfo) -> dict:
        """Convert diff graph to human-readable dict."""
        nodes_changed = (
            torch.nonzero(diff_graph.node_mask, as_tuple=False).flatten().tolist()
        )
        edges_changed = (
            torch.nonzero(diff_graph.edge_mask, as_tuple=False).flatten().tolist()
        )

        # Map indices to names if available
        def _idx_to_name(i: int) -> str:
            inv = component_info.inverse_component_indices
            if isinstance(inv, dict):
                return inv.get(i, str(i))
            if isinstance(inv, list) and len(inv) > 0 and isinstance(inv[0], dict):
                return inv[0].get(i, str(i))
            return str(i)

        # Build edge list tuples with names if possible
        edge_pairs: list[tuple[str, str]] = []
        edge_index = getattr(diff_graph, "edge_index", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            for u, v in edge_index.t().tolist():
                edge_pairs.append((_idx_to_name(u), _idx_to_name(v)))

        # Only include changed edges in listing
        changed_edge_pairs = (
            [edge_pairs[i] for i in edges_changed] if edge_pairs else []
        )

        return {
            "nodes_changed": [
                {"index": i, "name": _idx_to_name(i)} for i in nodes_changed
            ],
            "edges_changed": changed_edge_pairs,
            "num_nodes": diff_graph.num_nodes,  # FIXME: should never be different, remove.
            "num_edges": (
                edge_index.shape[1]
                if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2
                else 0
            ),
        }

    def diff_to_str(self, diff_graph: Data, component_info: ElectronicsInfo) -> str:
        d = self.diff_to_dict(diff_graph, component_info)
        parts: list[str] = []
        if d["nodes_changed"]:
            parts.append(
                "Changed nodes: "
                + ", ".join([f"{n['name']}({n['index']})" for n in d["nodes_changed"]])
            )
        if d["edges_changed"]:
            parts.append(
                "Changed edges: "
                + ", ".join([f"{u}-{v}" for u, v in d["edges_changed"]])
            )
        parts.append(f"Nodes changed: {len(d['nodes_changed'])}")
        parts.append(f"Edges changed: {len(d['edges_changed'])}")
        return "\n".join(parts)


# -----------------------------------------------------------------------------
# Standalone functions for batch processing ElectronicsStateTC
# -----------------------------------------------------------------------------
def register_components_batch(
    names: list[str],  # N
    component_types: torch.Tensor,  # [N]
    max_voltages: torch.Tensor,  # [N]
    max_currents: torch.Tensor,  # [N]
    terminals_per_component: torch.Tensor,  # [N]
    # component_ids: torch.Tensor | None = None, # [N]
    device: torch.device,
    batch_size: int,
    connector_terminal_connectivity_a: torch.Tensor | None = None,  # [len(connectors)]
    connector_terminal_connectivity_b: torch.Tensor | None = None,  # [len(connectors)]
) -> tuple[ElectronicsInfo, ElectronicsState]:
    """A function meant to register all components at once except connectors."""
    component_info = ElectronicsInfo()
    assert len(names) > 0, "At least one component must be registered."
    assert component_info.component_type.numel() == 0, (
        "Expected to have no components during registration (must register all components at once)."
    )

    N = len(names)
    assert (
        (N,)
        == component_types.shape
        == max_voltages.shape
        == max_currents.shape
        == terminals_per_component.shape
    ), "Shape mismatch in register_components_batch"

    component_info.component_type = component_types
    component_info.max_voltage = max_voltages
    component_info.max_current = max_currents
    component_info.terminals_per_component = terminals_per_component

    component_ids = torch.arange(N, dtype=torch.long, device=device)
    component_info.component_id = component_ids
    # TODO: assert that component_types match component names through ElectricalComponentsEnum ("@connector" and others.)

    base_terminal_to_components_batch = torch.repeat_interleave(
        component_ids, terminals_per_component
    )
    # Expand across batch; compute T from base (unexpanded) vector length
    component_info.terminal_to_component_batch = base_terminal_to_components_batch
    T = int(base_terminal_to_components_batch.shape[0])

    # Derived helpers
    start_terminal_idx = (
        torch.cumsum(component_info.terminals_per_component, dim=0)
        - component_info.terminals_per_component
    )
    component_info.component_first_terminal_index = start_terminal_idx

    # Default terminal roles: set all to TERM_A; call sites may refine.
    component_info.terminal_role = torch.full(
        (T,), int(TerminalRoleEnum.TERM_A.value), dtype=torch.long, device=device
    )

    # Initialize type-specific ComponentInfo holders using masks
    is_resistor = component_types == int(ElectricalComponentsEnum.RESISTOR.value)
    is_vsource = component_types == int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value)
    is_switch = component_types == int(ElectricalComponentsEnum.BUTTON.value)
    is_led = component_types == int(ElectricalComponentsEnum.LED.value)
    is_motor = component_types == int(ElectricalComponentsEnum.MOTOR.value)

    resistor_ids = component_ids[is_resistor]
    vsource_ids = component_ids[is_vsource]
    switch_ids = component_ids[is_switch]
    led_ids = component_ids[is_led]
    motor_ids = component_ids[is_motor]

    # Prepare per-type infos WITHOUT env batch dimension; sized only by count per type
    resistors_tc = ResistorInfo(
        component_ids=resistor_ids,
        resistance=torch.empty(
            (resistor_ids.shape[0],), dtype=torch.float32, device=device
        ),
    )
    vsources_tc = VoltageSourceInfo(
        component_ids=vsource_ids,
        voltage=torch.empty(
            (vsource_ids.shape[0],), dtype=torch.float32, device=device
        ),
    )
    switches_tc = SwitchInfo(
        component_ids=switch_ids,
        on_res_ohm=torch.empty(
            (switch_ids.shape[0],), dtype=torch.float32, device=device
        ),
        off_res_ohm=torch.empty(
            (switch_ids.shape[0],), dtype=torch.float32, device=device
        ),
        control_from_component=torch.full(
            (switch_ids.shape[0],), -1, dtype=torch.long, device=device
        ),
        control_type=torch.full(
            (batch_size, switch_ids.shape[0]),
            int(ControlTypeEnum.NONE.value),
            dtype=torch.long,
            device=device,
        ),
        control_threshold=torch.zeros(
            (switch_ids.shape[0],), dtype=torch.float32, device=device
        ),
        control_hysteresis=torch.zeros(
            (switch_ids.shape[0],), dtype=torch.float32, device=device
        ),
    )
    leds_tc = LedInfo(
        component_ids=led_ids,
        vf_drop=torch.empty((led_ids.shape[0],), dtype=torch.float32, device=device),
    )
    motors_tc = MotorInfo(
        component_ids=motor_ids,
        res_ohm=torch.empty((motor_ids.shape[0],), dtype=torch.float32, device=device),
        k_tau=torch.empty((motor_ids.shape[0],), dtype=torch.float32, device=device),
    )

    # Attach per-type infos to component_info
    component_info.resistors = resistors_tc
    component_info.vsources = vsources_tc
    component_info.switches = switches_tc
    component_info.leds = leds_tc
    component_info.motors = motors_tc

    # Create dynamic state (batched over envs)
    switch_state_tc = SwitchState(
        state_bits=torch.zeros(
            (batch_size, switch_ids.shape[0]), dtype=torch.bool, device=device
        )
    )
    led_state_tc = LedState(
        luminosity_pct=torch.zeros(
            (batch_size, led_ids.shape[0]), dtype=torch.float32, device=device
        )
    )
    motor_state_tc = MotorState(
        speed_pct=torch.zeros(
            (batch_size, motor_ids.shape[0]), dtype=torch.float32, device=device
        )
    )
    state = ElectronicsState(
        net_ids=torch.full((batch_size, T), -1, dtype=torch.long, device=device),
        state_bits=torch.zeros((batch_size, N), dtype=torch.bool, device=device),
        terminal_voltage=torch.zeros(
            (batch_size, T), dtype=torch.float32, device=device
        ),
        component_branch_current=torch.zeros(
            (batch_size, N), dtype=torch.float32, device=device
        ),
        switch_state=switch_state_tc,
        led_state=led_state_tc,
        motor_state=motor_state_tc,
        device=device,
        batch_size=batch_size,
    )

    # Set name-index maps once (kept identical across batch)
    component_info.component_indices_from_name = {
        name: i for i, name in enumerate(names)
    }
    component_info.inverse_component_indices = {
        i: n for n, i in component_info.component_indices_from_name.items()
    }

    # state already initialized with correct dynamic field shapes
    if (
        connector_terminal_connectivity_a is not None
        or connector_terminal_connectivity_b is not None
    ):
        assert (
            connector_terminal_connectivity_a is not None
            and connector_terminal_connectivity_b is not None
        ), (
            "if one of the connectivity tensors is not None, the other must be not None as well"
        )
        # connector_names = [n for n in names if n.endswith("@connector")]
        male_connector_ids = torch.tensor(
            [i for i, n in enumerate(names) if n.endswith("_male@connector")],
            dtype=torch.long,
            device=device,
        )
        female_connector_ids = torch.tensor(
            [i for i, n in enumerate(names) if n.endswith("_female@connector")],
            dtype=torch.long,
            device=device,
        )
        # NOTE: male connectors are A, and female are B.
        assert (
            male_connector_ids.shape[0]
            == female_connector_ids.shape[0]
            == connector_terminal_connectivity_a.shape[0]
            == connector_terminal_connectivity_b.shape[0]
        ), (
            "Expected len of terminal connectivity to be equal and equal to number of connectors"
        )
        # Each connector must expose exactly one terminal
        assert torch.all(
            component_info.terminals_per_component[male_connector_ids] == 1
        ), "Male connectors must have exactly one terminal each"
        assert torch.all(
            component_info.terminals_per_component[female_connector_ids] == 1
        ), "Female connectors must have exactly one terminal each"
        assert (connector_terminal_connectivity_a >= -1).all() & (
            connector_terminal_connectivity_a < T
        ).all(), "connector_terminal_connectivity_a is out of bounds"
        assert (connector_terminal_connectivity_b >= -1).all() & (
            connector_terminal_connectivity_b < T
        ).all(), "connector_terminal_connectivity_b is out of bounds"
        # Map component indices -> starting terminal indices (since connectors have 1 terminal, start index is their only terminal)
        start_terminal_idx = component_info.component_first_terminal_index
        male_connector_terminals = start_terminal_idx[male_connector_ids]
        female_connector_terminals = start_terminal_idx[female_connector_ids]
        # repeat batch ids for each connector terminal # but should be filtered since many are -1?
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=device),
            connector_terminal_connectivity_a.shape[0],
        )
        # connect A to B
        state = connect_terminal_to_net_or_create_new(
            state,
            component_info,
            batch_ids,
            male_connector_terminals,
            other_terminal_ids=female_connector_terminals,
        )

        # connect A to terminals
        state = connect_terminal_to_net_or_create_new(
            state,
            component_info,
            batch_ids,
            male_connector_terminals,  # connect other terminal to connectors, which are just nets.
            other_terminal_ids=connector_terminal_connectivity_a,
        )
        # connect B to terminals
        state = connect_terminal_to_net_or_create_new(
            state,
            component_info,
            batch_ids,
            female_connector_terminals,
            other_terminal_ids=connector_terminal_connectivity_b,
        )

    return component_info, state


def connect_terminal_to_net_or_create_new(
    electronics_states: "ElectronicsState",
    component_info: ElectronicsInfo,
    batch_ids: torch.Tensor,  # [K]
    terminal_ids: torch.Tensor,  # [K]
    *,
    target_net_ids: torch.Tensor | None = None,  # [K]
    other_terminal_ids: torch.Tensor | None = None,  # [K]
) -> "ElectronicsState":
    """Connect terminals to a net or create a new nets. If other_terminals is not None,
    use get target_nets from other terminals or create new ones (when both terminals==-1 and other_terminals==-1 in the same position.).

    Args:
      - batch_ids: [K]
      - terminal_ids: [K]
      - target_net_ids: Optional [K]
      - other_terminal_ids: Optional [K]

    Notes:
      - Terminals are global terminal indices in the flattened terminal space [0, T).
      - electronics_states must be batched (B >= 1), and terminal_to_component must be identical across batch.
    """
    es = electronics_states
    assert component_info.terminal_to_component_batch.numel() > 0, (
        "terminal_to_component_batch must be non-empty"
    )
    # assert shapes
    assert es.ndim > 0 and es.batch_size[0] >= 1, (
        "Expected electronics state to be batched."
    )
    assert batch_ids.ndim == terminal_ids.ndim == 1, (
        f"batch_ids, terminals must be 1D, got {batch_ids.shape}, {terminal_ids.shape}"
    )
    assert batch_ids.shape[0] == terminal_ids.shape[0], (
        f"batch_ids, terminals must have the same length, got {batch_ids.shape[0]}, {terminal_ids.shape[0]}"
    )
    # Exactly one of target_net_ids or other_terminal_ids must be provided
    provided_target = target_net_ids is not None
    provided_other = other_terminal_ids is not None
    assert provided_target ^ provided_other, (  # `^` is XOR
        "Provide exactly one of target_net_ids or other_terminal_ids"
    )

    # assert indices.
    assert torch.all((batch_ids >= 0) & (batch_ids < es.batch_size[0])), (
        "Batch indices out of range."
    )
    T_total = component_info.terminal_to_component_batch.shape[0]
    assert torch.all((terminal_ids >= 0) & (terminal_ids < T_total)), (
        f"Terminal indices out of range. Got {terminal_ids} ."
    )

    if other_terminal_ids is not None:
        assert (
            other_terminal_ids.ndim == 1
            and other_terminal_ids.shape[0] == terminal_ids.shape[0]
        ), "other_terminal_ids must be 1D and match terminal_ids length"
        # Allow -1 as a sentinel (ignored); otherwise must be a valid terminal index
        valid_other = other_terminal_ids[other_terminal_ids >= 0]
        assert torch.all((valid_other < T_total)), "other_terminal_ids out of range"
    if target_net_ids is not None:
        assert (
            target_net_ids.ndim == 1
            and target_net_ids.shape[0] == terminal_ids.shape[0]
        ), "target_net_ids must be 1D and match terminal_ids length"

    # sanity check/hint: net_ids already batched [B, T]
    assert es.net_ids.ndim == 2 and es.net_ids.shape[0] == es.batch_size[0]

    device = es.device
    batch_ids = batch_ids.to(device)
    terminal_ids = terminal_ids.to(device)
    if target_net_ids is not None:
        target_net_ids = target_net_ids.to(device)
    if other_terminal_ids is not None:
        other_terminal_ids = other_terminal_ids.to(device)

    # Compute base next-net per batch
    B = es.batch_size[0]
    cur = es.net_ids
    any_assigned = (cur >= 0).any(dim=1)
    max_ids = torch.where(
        any_assigned,
        cur.masked_fill(cur < 0, -1).max(dim=1).values,
        torch.full((B,), -1, dtype=cur.dtype, device=device),
    )
    base_next = max_ids + 1  # [B]

    if other_terminal_ids is not None:
        valid_mask = other_terminal_ids >= 0
        if valid_mask.any():
            b = batch_ids[valid_mask]
            a = terminal_ids[valid_mask]
            c = other_terminal_ids[valid_mask]

            g1 = es.net_ids[b, a]
            g2 = es.net_ids[b, c]

            both_neg = (g1 < 0) & (g2 < 0)
            only2neg = (g1 >= 0) & (g2 < 0)
            only1neg = (g1 < 0) & (g2 >= 0)
            merge_mask = (g1 >= 0) & (g2 >= 0) & (g1 != g2)

            # Assign new nets for both-unassigned pairs per-batch using segment ranks
            idx_bn = torch.nonzero(both_neg, as_tuple=False).flatten()
            if idx_bn.numel() > 0:
                b_bn = b[idx_bn]
                order = torch.argsort(b_bn)
                b_sorted = b_bn[order]
                start_flags = torch.ones_like(b_sorted, dtype=torch.bool)
                start_flags[1:] = b_sorted[1:] != b_sorted[:-1]
                group_first_pos = torch.nonzero(start_flags, as_tuple=False).flatten()
                group_id = torch.cumsum(start_flags.to(torch.long), dim=0) - 1
                first_idx_per_elem = group_first_pos[group_id]
                rank_sorted = torch.arange(b_sorted.numel(), device=device) - first_idx_per_elem
                new_ids_sorted = base_next[b_sorted] + rank_sorted
                inv = torch.empty_like(order)
                inv[order] = torch.arange(order.numel(), device=device)
                new_ids = new_ids_sorted[inv]
                es.net_ids[b[idx_bn], a[idx_bn]] = new_ids
                es.net_ids[b[idx_bn], c[idx_bn]] = new_ids

            # Attach single-unassigned to existing nets
            if only2neg.any():
                es.net_ids[b[only2neg], c[only2neg]] = g1[only2neg]
            if only1neg.any():
                es.net_ids[b[only1neg], a[only1neg]] = g2[only1neg]

            # Merge distinct existing nets; keep minimal id
            if merge_mask.any():
                u = torch.minimum(g1[merge_mask], g2[merge_mask])
                v = torch.maximum(g1[merge_mask], g2[merge_mask])
                bb = b[merge_mask]
                merges = torch.unique(torch.stack([bb.to(torch.long), u, v], dim=1), dim=0)
                # Apply merges (loop over unique pairs, not per-batch)
                for i in range(int(merges.shape[0])):
                    bb_i = int(merges[i, 0].item())
                    uu = int(merges[i, 1].item())
                    vv = int(merges[i, 2].item())
                    es.net_ids[bb_i, es.net_ids[bb_i] == vv] = uu
    else:
        # target_net_ids path
        assert target_net_ids is not None
        valid_mask = target_net_ids >= 0
        if valid_mask.any():
            b = batch_ids[valid_mask]
            a = terminal_ids[valid_mask]
            tnet = target_net_ids[valid_mask]
            g1 = es.net_ids[b, a]

            neg = g1 < 0
            if neg.any():
                es.net_ids[b[neg], a[neg]] = tnet[neg]

            merge_mask = (~neg) & (g1 != tnet)
            if merge_mask.any():
                u = torch.minimum(g1[merge_mask], tnet[merge_mask])
                v = torch.maximum(g1[merge_mask], tnet[merge_mask])
                bb = b[merge_mask]
                merges = torch.unique(torch.stack([bb.to(torch.long), u, v], dim=1), dim=0)
                for i in range(int(merges.shape[0])):
                    bb_i = int(merges[i, 0].item())
                    uu = int(merges[i, 1].item())
                    vv = int(merges[i, 2].item())
                    es.net_ids[bb_i, es.net_ids[bb_i] == vv] = uu

    return es
