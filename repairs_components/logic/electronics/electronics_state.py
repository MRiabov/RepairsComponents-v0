from dataclasses import field

import torch
from tensordict import TensorClass
from torch_geometric.data import Data
from typing_extensions import deprecated


# -----------------------------------------------------------------------------
# New TensorClass-based Electronics State (non-imperative, batchable)
# -----------------------------------------------------------------------------
class ElectronicsState(TensorClass):
    """TensorClass-based electronics state.

    Design goals:
    - Represent component features as tensors for accelerator-friendly ops.
    - Represent electrical connectivity at terminal level via net labels per terminal.
    - Provide functional, batchable APIs (no in-place Python mutation of components).

    Conventions:
    - Use "terminal" terminology for electronic connection points.
    - net_id labels group terminals that are electrically the same node.
      Terminals that share the same non-negative net_id belong to the same net.
      A value of -1 means the terminal is not connected to any net yet.
    - Batch dimension (if present) is the first dimension for batchable tensors.
    """

    # Per-component tensors (shape: [N] or [B, N])
    component_type: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    component_id: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
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

    # Terminal mapping (no batch dimension, constant mapping for all envs)
    # terminal_to_component maps each terminal id in [0, T) to owning component index in [0, N)
    terminal_to_component: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )

    # Connectivity: per-env terminal net assignments
    # Shape: [T] (no batch) or [B, T] (batched). Values: -1 for unassigned else non-negative net id.
    net_id: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )

    # Optional Python-side name-index mappings (like PhysicalState)
    # Can be a single dict for single env, or a list[dict] for batched envs #FIXME: should always be dict as changed by __post_init__
    component_indices_from_name: dict | list[dict] = field(default_factory=dict)
    inverse_component_indices: dict | list[dict] = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.component_indices_from_name, (list, dict))
        # Establish name maps; debatch Python dict fields like PhysicalState does.
        if isinstance(self.component_indices_from_name, list):
            assert self.batch_size and self.batch_size[0] >= 1, (
                "Batch size must be set for batched state"
            )
            assert len(self.component_indices_from_name) == self.batch_size[0], (
                "Expected one name-index dict per batch element"
            )
            first = self.component_indices_from_name[0]
            assert isinstance(first, dict), (
                "component_indices_from_name[0] must be a dict"
            )
            # Debatch to a single dict reference (assumed identical across envs)
            self.component_indices_from_name = first
            self.inverse_component_indices = {v: k for k, v in first.items()}
        else:
            self.inverse_component_indices = {
                v: k for k, v in self.component_indices_from_name.items()
            }

        return self

    def clear_all_connections(self) -> "ElectronicsState":
        """Reset all terminal nets to -1 (no connectivity)."""
        device = self.device
        assert self.net_id.ndim in (1, 2), "net_id must be 1D or 2D"
        if self.net_id.ndim == 2:
            B, T = self.net_id.shape
            self.net_id = torch.full((B, T), -1, dtype=torch.long, device=device)
        elif self.net_id.ndim == 1:
            T = self.net_id.shape[0]
            self.net_id = torch.full((T,), -1, dtype=torch.long, device=device)
        return self

    def _build_component_edges_from_nets(self) -> torch.Tensor:
        """Derive undirected component-level edges from terminal net assignments.

        Returns:
            edge_index [2, E] for single-env.
        Note: If batched, this currently supports only single-env and will raise.
        """
        assert self.terminal_to_component.numel() > 0, "Register components first"
        if self.net_id.ndim == 2:
            # For now, keep export single-env to avoid mixing batches
            raise NotImplementedError(
                "export for batched ElectronicsStateTC not yet supported"
            )

        net = self.net_id
        T = int(self.terminal_to_component.numel())
        assert net.shape[0] == T, "net_id length must equal number of terminals"

        # Build nets -> terminals mapping
        valid_mask = net >= 0
        if not valid_mask.any():
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        unique_nets = torch.unique(net[valid_mask])
        edges_set: set[tuple[int, int]] = set()
        for gid in unique_nets.tolist():
            term_idx = torch.nonzero(net == gid, as_tuple=False).flatten()
            comp_idx = self.terminal_to_component[term_idx]
            # Unique components participating in this net
            comps = torch.unique(comp_idx).tolist()
            # Generate all i<j pairs among components in this net
            for i in range(len(comps)):  # TODO refactor to batch comparison.
                for j in range(i + 1, len(comps)):
                    a, b = comps[i], comps[j]
                    if a == b:
                        continue
                    u, v = (a, b) if a < b else (b, a)
                    edges_set.add((int(u), int(v)))

        if not edges_set:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_list = torch.tensor(
            sorted(list(edges_set)), dtype=torch.long, device=self.device
        )
        return edge_list.t().contiguous()

    def export_graph(self) -> Data:
        """Export a torch_geometric.Data graph for ML consumers.

        Node features: [max_voltage, max_current, component_type, component_id]
        Edge index: derived from terminal net connectivity at component level.
        """
        edge_index = self._build_component_edges_from_nets()

        mv = _assert_1d(self.max_voltage).to(torch.float32)
        mc = _assert_1d(self.max_current).to(torch.float32)
        ctype = _assert_1d(self.component_type).to(torch.float32)
        cid = _assert_1d(self.component_id).to(torch.float32)
        x = torch.stack([mv, mc, ctype, cid], dim=1).bfloat16()

        graph = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
        return graph

    def diff(self, other: "ElectronicsState") -> tuple[Data, int]:
        """Compute a graph diff between two electronics states (component-level edges).

        Returns a PyG Data with:
            - x: node features (from self)
            - edge_index: combined edges from both states (union)
            - node_mask: bool mask of nodes whose scalar features differ
            - edge_mask: bool mask of edges that were added/removed/changed
        and an integer count of total differences (nodes + edges).
        """
        # Build edge sets
        e_self = self._build_component_edges_from_nets()
        e_other = other._build_component_edges_from_nets()

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

        mv_a, mv_b = _assert_1d(self.max_voltage), _assert_1d(other.max_voltage)
        mc_a, mc_b = _assert_1d(self.max_current), _assert_1d(other.max_current)
        ct_a, ct_b = _assert_1d(self.component_type), _assert_1d(other.component_type)
        id_a, id_b = _assert_1d(self.component_id), _assert_1d(other.component_id)

        # Align N
        assert mv_a.shape[0] == mv_b.shape[0], "Node count mismatch between states"
        node_mask = (mv_a != mv_b) | (mc_a != mc_b) | (ct_a != ct_b) | (id_a != id_b)

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
            x=torch.stack(
                [mv_a, mc_a, ct_a.to(torch.float32), id_a.to(torch.float32)], dim=1
            ).bfloat16(),
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            num_nodes=mv_a.shape[0],
        )

        n_diffs = int(node_mask.sum().item()) + int(edge_mask.sum().item())
        return diff_graph, n_diffs

    def diff_to_dict(self, diff_graph: Data) -> dict:
        """Convert diff graph to human-readable dict."""
        nodes_changed = (
            torch.nonzero(diff_graph.node_mask, as_tuple=False).flatten().tolist()
        )
        edges_changed = (
            torch.nonzero(diff_graph.edge_mask, as_tuple=False).flatten().tolist()
        )

        # Map indices to names if available
        def _idx_to_name(i: int) -> str:
            if isinstance(self.inverse_component_indices, dict):
                return self.inverse_component_indices.get(i, str(i))
            if (
                isinstance(self.inverse_component_indices, list)
                and len(self.inverse_component_indices) > 0
                and isinstance(self.inverse_component_indices[0], dict)
            ):
                return self.inverse_component_indices[0].get(i, str(i))
            return str(i)

        # Build edge list tuples with names if possible
        edge_pairs: list[tuple[str, str]] = []
        edge_index = getattr(diff_graph, "edge_index", None)
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            for u, v in edge_index.t().tolist():
                edge_pairs.append((_idx_to_name(int(u)), _idx_to_name(int(v))))

        # Only include changed edges in listing
        changed_edge_pairs = (
            [edge_pairs[i] for i in edges_changed] if edge_pairs else []
        )

        return {
            "nodes_changed": [
                {"index": i, "name": _idx_to_name(i)} for i in nodes_changed
            ],
            "edges_changed": changed_edge_pairs,
            "num_nodes": int(diff_graph.num_nodes),
            "num_edges": (
                int(edge_index.shape[1])
                if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2
                else 0
            ),
        }

    def diff_to_str(self, diff_graph: Data) -> str:
        d = self.diff_to_dict(diff_graph)
        parts = [
            f"Nodes changed: {len(d['nodes_changed'])}",
            f"Edges changed: {len(d['edges_changed'])}",
        ]
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
        return "\n".join(parts)


# -----------------------------------------------------------------------------
# Standalone functions for batch processing ElectronicsStateTC
# -----------------------------------------------------------------------------
def register_components_batch(
    electronics_states: "ElectronicsState",
    names: list[str],
    component_types: torch.Tensor,
    max_voltages: torch.Tensor,
    max_currents: torch.Tensor,
    terminals_per_component: torch.Tensor,
    component_ids: torch.Tensor | None = None,
) -> "ElectronicsState":
    state = electronics_states
    device = state.device
    assert len(names) > 0, "At least one component must be registered."
    B = state.batch_size[0]
    assert B is not None and len(state.batch_size) >= 1

    N = len(names)
    assert component_types.shape[-1] == N
    assert max_voltages.shape[-1] == N
    assert max_currents.shape[-1] == N
    assert terminals_per_component.shape[-1] == N

    def to_dev(x: torch.Tensor) -> torch.Tensor:
        return x.to(device)

    # Normalize inputs to batched shape if state is batched
    def _expand_to_batch(x: torch.Tensor) -> torch.Tensor:
        if state.batch_size and state.batch_size[0] >= 1:
            B_ = state.batch_size[0]
            if x.ndim == 1:
                return x.to(device).unsqueeze(0).expand(B_, -1)
            elif x.ndim == 2:
                # if provided as [B,N], ensure identical across batch
                assert x.shape[0] == B_
                if not torch.allclose(x, x[:1].expand_as(x)):
                    raise AssertionError("Inputs must be identical across batch")
                return x.to(device)
        return x.to(device)

    state.component_type = _expand_to_batch(component_types)
    state.max_voltage = _expand_to_batch(max_voltages)
    state.max_current = _expand_to_batch(max_currents)
    state.terminals_per_component = _expand_to_batch(terminals_per_component)

    if component_ids is None:
        state.component_id = torch.arange(N, dtype=torch.long, device=device)
    else:
        assert component_ids.shape[-1] == N
        state.component_id = to_dev(component_ids)

    assert state.terminals_per_component.ndim in (1, 2)
    tpc = state.terminals_per_component
    if tpc.ndim == 2:
        assert torch.allclose(tpc, tpc[:1].expand_as(tpc))
        tpc_1d = tpc[0]
    else:
        tpc_1d = tpc

    comp_idx = torch.arange(N, device=device)
    base_ttc = torch.repeat_interleave(comp_idx, tpc_1d)
    if state.batch_size and state.batch_size[0] >= 1:
        B_ = state.batch_size[0]
        state.terminal_to_component = base_ttc.unsqueeze(0).expand(B_, -1)
        T = int(base_ttc.numel())
    else:
        state.terminal_to_component = base_ttc
        T = int(base_ttc.numel())

    # Set name-index maps once (kept identical across batch)
    state.component_indices_from_name = {name: i for i, name in enumerate(names)}
    state.inverse_component_indices = {
        i: n for n, i in state.component_indices_from_name.items()
    }

    if state.batch_size and state.batch_size[0] >= 1:
        B = state.batch_size[0]
        state.net_id = torch.full((B, T), -1, dtype=torch.long, device=device)
    else:
        state.net_id = torch.full((T,), -1, dtype=torch.long, device=device)

    return state


def register_connectors_batch(
    electronics_states: "ElectronicsState",
    connector_names: list[str],
    component_ids: torch.Tensor | None = None,
) -> "ElectronicsState":
    """Register electrical connectors as components with two terminals each.

    - Requires a batched ElectronicsState (B >= 1).
    - Appends connectors to existing components, expanding all tensors accordingly.
    - Per-connector max_voltage/max_current are initialized to 0.0.
    - terminals_per_component for connectors is fixed to 2.
    """
    es = electronics_states
    assert len(es.batch_size) > 0 and es.batch_size[0] >= 1, (
        "Expected electronics state to be batched."
    )
    assert len(connector_names) > 0, "At least one connector must be registered."

    device = es.device
    B = es.batch_size[0]
    C = len(connector_names)

    # Existing component count and terminals
    old_N = int(es.component_id.shape[-1]) if es.component_id.numel() > 0 else 0
    old_tpc = es.terminals_per_component
    if old_tpc.numel() > 0 and old_tpc.ndim == 2:
        assert torch.allclose(old_tpc, old_tpc[:1].expand_as(old_tpc))
        old_tpc_batched = old_tpc
        old_tpc_1d = old_tpc[0]
    else:
        old_tpc_batched = (
            old_tpc.unsqueeze(0).expand(B, -1) if old_tpc.numel() > 0 else old_tpc
        )
        old_tpc_1d = old_tpc
    # Number of existing terminals (not used directly, kept for clarity)
    # old_T = int(es.terminal_to_component.shape[-1]) if es.terminal_to_component.ndim == 2 else int(es.terminal_to_component.numel())

    # Append component_type, component_id, max_voltage, max_current, terminals_per_component
    from .component import ElectricalComponentsEnum as ECE  # local import to avoid cycles

    new_types = torch.full((C,), int(ECE.CONNECTOR), dtype=torch.long, device=device)
    new_mv = torch.zeros((C,), dtype=torch.float32, device=device)
    new_mc = torch.zeros((C,), dtype=torch.float32, device=device)
    new_tpc = torch.full((C,), 2, dtype=torch.long, device=device)

    if es.component_type.numel() == 0:
        # Initialize respecting batch shape
        if len(es.batch_size) > 0 and es.batch_size[0] >= 1:
            B_ = es.batch_size[0]
            es.component_type = new_types.unsqueeze(0).expand(B_, -1)
            es.max_voltage = new_mv.unsqueeze(0).expand(B_, -1)
            es.max_current = new_mc.unsqueeze(0).expand(B_, -1)
            es.terminals_per_component = new_tpc.unsqueeze(0).expand(B_, -1)
        else:
            es.component_type = new_types
            es.max_voltage = new_mv
            es.max_current = new_mc
            es.terminals_per_component = new_tpc
    else:
        if es.component_type.ndim == 2:
            B_ = es.batch_size[0]
            es.component_type = torch.cat(
                [es.component_type.to(device), new_types.unsqueeze(0).expand(B_, -1)], dim=-1
            )
            es.max_voltage = torch.cat(
                [es.max_voltage.to(device), new_mv.unsqueeze(0).expand(B_, -1)], dim=-1
            )
            es.max_current = torch.cat(
                [es.max_current.to(device), new_mc.unsqueeze(0).expand(B_, -1)], dim=-1
            )
            es.terminals_per_component = torch.cat(
                [old_tpc_batched.to(device), new_tpc.unsqueeze(0).expand(B_, -1)], dim=-1
            )
        else:
            es.component_type = torch.cat([es.component_type.to(device), new_types], dim=-1)
            es.max_voltage = torch.cat([es.max_voltage.to(device), new_mv], dim=-1)
            es.max_current = torch.cat([es.max_current.to(device), new_mc], dim=-1)
            es.terminals_per_component = torch.cat([old_tpc_1d.to(device), new_tpc], dim=-1)

    # Component IDs
    if component_ids is None:
        new_ids = torch.arange(old_N, old_N + C, dtype=torch.long, device=device)
    else:
        assert component_ids.shape[-1] == C
        new_ids = component_ids.to(device)
    es.component_id = (
        new_ids
        if es.component_id.numel() == 0
        else torch.cat([es.component_id.to(device), new_ids], dim=-1)
    )

    # Update terminal_to_component by appending mapping for connectors
    conn_comp_idx = torch.arange(old_N, old_N + C, device=device)
    conn_ttc = torch.repeat_interleave(conn_comp_idx, new_tpc)
    if es.terminal_to_component.numel() == 0:
        # Respect batch: initialize as [B, add_T] if batched
        if len(es.batch_size) > 0 and es.batch_size[0] >= 1:
            es.terminal_to_component = conn_ttc.unsqueeze(0).expand(B, -1)
        else:
            es.terminal_to_component = conn_ttc
    else:
        if es.terminal_to_component.ndim == 2:
            es.terminal_to_component = torch.cat(
                [es.terminal_to_component.to(device), conn_ttc.unsqueeze(0).expand(B, -1)], dim=-1
            )
        else:
            es.terminal_to_component = torch.cat(
                [es.terminal_to_component.to(device), conn_ttc], dim=-1
            )

    # Expand net_id with -1 for new terminals
    add_T = int(conn_ttc.numel())
    if es.net_id.ndim == 2:
        pad = torch.full((B, add_T), -1, dtype=torch.long, device=device)
        es.net_id = torch.cat([es.net_id.to(device), pad], dim=1)
    elif es.net_id.ndim == 1:
        pad = torch.full((add_T,), -1, dtype=torch.long, device=device)
        # If single-env net, expand to batched to be consistent
        es.net_id = torch.cat([es.net_id.to(device), pad], dim=0)

    # Update name maps (coerce to dict if needed)
    if isinstance(es.component_indices_from_name, list):
        assert len(es.component_indices_from_name) == B
        base = es.component_indices_from_name[0]
        assert isinstance(base, dict)
        es.component_indices_from_name = dict(base)
    for i, nm in enumerate(connector_names):
        es.component_indices_from_name[nm] = old_N + i
    es.inverse_component_indices = {
        v: k for k, v in es.component_indices_from_name.items()
    }

    return es


def connect_connector_to_one_terminal(
    electronics_states: "ElectronicsState",
    terminal_pair: torch.Tensor,  # [K, 3] -> (batch_idx, connector_terminal, target_terminal)
) -> "ElectronicsState":
    """Connect connector terminals to target terminals using explicit batch indices.

    Input format:
      - [K, 3]: (b, connector_terminal, target_terminal), 0 <= b < B.

    Notes:
      - Terminals are global terminal indices in the flattened terminal space [0, T).
      - electronics_states must be batched (B >= 1), and terminal_to_component must be identical across batch.
    """
    es = electronics_states
    assert es.terminal_to_component.numel() > 0
    assert len(es.batch_size) > 0 and es.batch_size[0] >= 1, (
        "Expected electronics state to be batched."
    )

    device = es.device
    pair = terminal_pair.to(device)

    # Normalize terminal count T
    if es.terminal_to_component.ndim == 2:
        ttc = es.terminal_to_component
        assert torch.allclose(ttc, ttc[:1].expand_as(ttc)), (
            "terminal_to_component must be identical across batch"
        )
        T = int(ttc.shape[1])
    else:
        T = int(es.terminal_to_component.numel())

    net = es.net_id
    assert net.ndim == 2 and net.shape[0] == es.batch_size[0]
    B = net.shape[0]

    # Expect [K,3] input: (b, connector_terminal, target_terminal)
    assert pair.ndim == 2 and pair.shape[1] == 3, "terminal_pair must be [K,3]"
    b_all = pair[:, 0]
    t1_all = pair[:, 1]
    t2_all = pair[:, 2]

    # Validate indices
    assert torch.all((b_all >= 0) & (b_all < B)), "batch indices out of range"
    assert torch.all((t1_all >= 0) & (t1_all < T) & (t2_all >= 0) & (t2_all < T)), (
        "terminal indices out of range"
    )

    # Process per batch to allow multiple pairs per batch
    unique_b = torch.unique(b_all)
    for b in unique_b.tolist():
        mask = (b_all == b)
        if not torch.any(mask):
            continue
        t1 = t1_all[mask]
        t2 = t2_all[mask]
        # Iterate pairs for this batch (safe, typically small K)
        for i in range(t1.shape[0]):
            a = int(t1[i].item())
            c = int(t2[i].item())
            g1 = int(net[b, a].item())
            g2 = int(net[b, c].item())
            big = int(T * 2)
            cand1 = g1 if g1 >= 0 else big
            cand2 = g2 if g2 >= 0 else big
            gid = min(cand1, cand2, a, c)
            # Merge nets equal to g1 or g2 into gid (ignore negatives)
            if g1 >= 0:
                net[b, net[b] == g1] = gid
            if g2 >= 0 and g2 != g1:
                net[b, net[b] == g2] = gid
            # Assign the two terminals to gid
            net[b, a] = gid
            net[b, c] = gid

    es.net_id = net.to(device)
    return es

def connect_terminals_batch(
    electronics_states: "ElectronicsState",
    terminal_pairs: torch.Tensor,
) -> "ElectronicsState":
    """Deprecated alias.

    Accepts historical [B,2] format (one pair per batch). Converts it to [B,3]
    with explicit batch indices and delegates to connect_connector_to_one_terminal.
    If [K,3] is provided, it is forwarded directly.
    """
    es = electronics_states
    if terminal_pairs.ndim == 2 and terminal_pairs.shape[1] == 2:
        B = es.batch_size[0]
        assert terminal_pairs.shape[0] == B, "terminal_pairs must be [B,2] for this path"
        b_idx = torch.arange(B, dtype=terminal_pairs.dtype, device=terminal_pairs.device).view(B, 1)
        pairs_k3 = torch.cat([b_idx, terminal_pairs], dim=1)
        return connect_connector_to_one_terminal(es, pairs_k3)
    # Otherwise assume already [K,3]
    return connect_connector_to_one_terminal(es, terminal_pairs)


# Normalize shapes of node features to [N, F]
def _assert_1d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        # Validate equal across batch then use first env
        assert torch.allclose(x, x[:1].expand_as(x)), (
            "Per-component features must match across batch for export"
        )
        return x[0]
    return x
