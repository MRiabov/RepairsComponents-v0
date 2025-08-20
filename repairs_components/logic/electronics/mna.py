from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorClass

from .electronics_state import (
    ElectronicsInfo,
    ElectronicsState,
)


class SolveResult(TensorClass):
    # tensorclass so it can be stacked if necessary. all should be init with batch_size=None, and then stacked if necessary.
    state: ElectronicsState
    terminated: torch.Tensor
    burned_component_indices: torch.Tensor


def _build_net_mapping(
    net_ids_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build mapping from global net ids to contiguous [0..G-1] local ids.

    Returns (assigned_mask[T], global_to_local[T], ground_global_id)
    """
    T = net_ids_b.shape[0]
    assert net_ids_b.ndim == 1 and T > 0
    assigned_mask = net_ids_b >= 0
    if not torch.any(assigned_mask):
        # No nets assigned in this batch element
        g2l = torch.full((T,), -1, dtype=torch.long, device=net_ids_b.device)
        return assigned_mask, g2l, -1
    assigned = torch.unique(net_ids_b[assigned_mask])
    # Choose smallest global id as ground (0 after mapping)
    ground_gid: int = int(torch.min(assigned).item())
    others = assigned[assigned != ground_gid]
    # local ids: ground -> 0, others -> 1..G-1
    g2l = torch.full((T,), -1, dtype=torch.long, device=net_ids_b.device)
    g2l[ground_gid] = 0
    if others.numel() > 0:
        g2l[others] = torch.arange(
            1, others.numel() + 1, dtype=torch.long, device=net_ids_b.device
        )
    return assigned_mask, g2l, ground_gid


# -----------------------------------------------------------------------------
# Per-type stamping helpers (pure torch; suitable for torch.compile)
# -----------------------------------------------------------------------------
@dataclass
class _ResistLike:
    component_ids: torch.Tensor
    resistance: torch.Tensor


def _stamp_resistors(
    A: torch.Tensor,
    z: torch.Tensor,
    *,
    net_b: torch.Tensor,
    g2l: torch.Tensor,
    start_idx: torch.Tensor,
    resistors,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Stamp 2-terminal resistors into A (conductance matrix).

    Returns (A, z, (cids_r, a_terms_r, b_terms_r, Rr)) for downstream current calc.
    """
    device = A.device
    Nr = int(resistors.component_ids.shape[0])
    if Nr == 0:
        empty = (torch.empty(0, dtype=torch.long, device=device),) * 3 + (
            torch.empty(0, dtype=torch.float32, device=device),
        )
        return A, z, empty  # type: ignore

    cids_r = resistors.component_ids  # [Nr]
    a_terms_r = start_idx[cids_r]
    b_terms_r = a_terms_r + 1
    ga = net_b[a_terms_r]
    gb = net_b[b_terms_r]
    assert torch.all(ga >= 0).item() and torch.all(gb >= 0).item(), (
        "All resistor terminals must be connected"
    )
    la = g2l[ga]
    lb = g2l[gb]
    ia = la - 1  # ground(0) -> -1, others -> [0..G-2]
    ib = lb - 1
    Rr = resistors.resistance.to(torch.float32)  # [Nr]
    assert torch.all(Rr > 0).item(), "Resistor resistance must be > 0"
    gr = 1.0 / Rr
    # Diagonal contributions
    mask_a = ia >= 0
    if torch.any(mask_a):
        rows = ia[mask_a]
        vals = gr[mask_a]
        A.index_put_((rows, rows), vals, accumulate=True)
    mask_b = ib >= 0
    if torch.any(mask_b):
        rows = ib[mask_b]
        vals = gr[mask_b]
        A.index_put_((rows, rows), vals, accumulate=True)
    # Off-diagonal contributions
    both = mask_a & mask_b
    if torch.any(both):
        ra = ia[both]
        rb = ib[both]
        vals = -gr[both]
        A.index_put_((ra, rb), vals, accumulate=True)
        A.index_put_((rb, ra), vals, accumulate=True)

    return A, z, (cids_r, a_terms_r, b_terms_r, Rr)


def _stamp_vsources(
    A: torch.Tensor,
    z: torch.Tensor,
    *,
    net_b: torch.Tensor,
    g2l: torch.Tensor,
    start_idx: torch.Tensor,
    vsources,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stamp ideal voltage sources (first terminal -> second terminal)."""
    device = A.device
    Nv = int(vsources.component_ids.shape[0])
    if Nv == 0:
        return A, z
    cids_v = vsources.component_ids  # [Nv] (unused here; kept for symmetry)
    a_terms_v = start_idx[cids_v]
    b_terms_v = a_terms_v + 1
    ga = net_b[a_terms_v]
    gb = net_b[b_terms_v]
    assert torch.all(ga >= 0).item() and torch.all(gb >= 0).item(), (
        "All voltage source terminals must be connected"
    )
    la = g2l[ga]
    lb = g2l[gb]
    ia = la - 1
    ib = lb - 1
    ivar = (G - 1) + torch.arange(Nv, device=device)
    # Node-source couplings
    mask_ia = ia >= 0
    if torch.any(mask_ia):
        rows = ia[mask_ia]
        cols = ivar[mask_ia]
        vals = torch.ones_like(rows, dtype=torch.float32)
        A.index_put_((rows, cols), vals, accumulate=True)
        A.index_put_((cols, rows), vals, accumulate=True)
    mask_ib = ib >= 0
    if torch.any(mask_ib):
        rows = ib[mask_ib]
        cols = ivar[mask_ib]
        vals = -torch.ones_like(rows, dtype=torch.float32)
        A.index_put_((rows, cols), vals, accumulate=True)
        A.index_put_((cols, rows), vals, accumulate=True)
    # RHS for sources
    z[ivar] = vsources.voltage.to(torch.float32)
    return A, z


def solve_dc_once(
    component_info: ElectronicsInfo, state: ElectronicsState
) -> SolveResult:
    """Solve a single DC operating point for a batch of circuits using MNA.

    Currently supports:
      - Resistors (2-terminal, series/parallel arbitrary)
      - Motors as stall resistors (2-terminal via res_ohm)
      - Ideal independent voltage sources (2-terminal)

    Returns updated state voltages/currents, a termination flag, and burned indices (empty for now).
    """
    device = state.device
    B = int(state.net_ids.shape[0])

    # Aliases to per-type infos (unbatched over envs)
    resistors = component_info.resistors
    motors = component_info.motors
    vsources = component_info.vsources
    switches = component_info.switches
    leds = component_info.leds

    # Prepare output tensors
    out_terminal_voltage = torch.zeros_like(state.terminal_voltage)
    out_component_current = torch.zeros_like(state.component_branch_current)

    # Precompute starts and terminal counts
    start_idx = component_info.component_first_terminal_index  # [N]
    tpc = component_info.terminals_per_component  # [N]

    # Helper closures
    def comp_terms(cid: int) -> tuple[int, int]:
        s = int(start_idx[cid].item())
        k = int(tpc[cid].item())
        assert k == 2, "Only 2-terminal components are supported in MNA stamping now"
        return s, s + 1

    Nr = int(resistors.component_ids.shape[0])
    Nm = int(motors.component_ids.shape[0])
    Nv = int(vsources.component_ids.shape[0])
    Ns = int(switches.component_ids.shape[0])
    Nl = int(leds.component_ids.shape[0])

    # Dimensions: use fixed T nodes + Nv source currents for all batches
    T = int(state.net_ids.shape[1])
    n_unknown = T + Nv
    A = torch.zeros((B, n_unknown, n_unknown), dtype=torch.float32, device=device)
    z = torch.zeros((B, n_unknown), dtype=torch.float32, device=device)

    # Compute per-batch ground as the smallest assigned net id
    net_ids = state.net_ids  # [B, T]
    assigned_mask = net_ids >= 0
    any_assigned = torch.any(assigned_mask, dim=1)
    assert torch.all(any_assigned).item(), (
        "Each batch must have at least one assigned net"
    )
    large = T + 1
    masked = net_ids.masked_fill(~assigned_mask, large)
    ground_gid = masked.min(dim=1).values  # [B]

    # Determine which net indices [0..T-1] are actually present per batch
    idx = net_ids.clone()
    idx[~assigned_mask] = 0  # dummy where not assigned; src below will be 0 there
    present_counts = torch.zeros((B, T), dtype=torch.float32, device=device)
    present_counts.scatter_add_(1, idx, assigned_mask.to(torch.float32))
    present_mask_full = present_counts > 0  # [B, T]

    batch_idx_nr = None
    batch_idx_nv = None

    # ---------- Stamp resistors (2-terminal) ----------
    if Nr > 0:
        cids_r = resistors.component_ids  # [Nr]
        a_terms_r = start_idx[cids_r]
        b_terms_r = a_terms_r + 1
        ga = net_ids[:, a_terms_r]  # [B, Nr]
        gb = net_ids[:, b_terms_r]  # [B, Nr]
        assert torch.all(ga >= 0).item() and torch.all(gb >= 0).item(), (
            "All resistor terminals must be connected"
        )
        Rr = resistors.resistance.to(torch.float32)  # [Nr]
        assert torch.all(Rr > 0).item(), "Resistor resistance must be > 0"
        gr = (1.0 / Rr).unsqueeze(0).expand(B, -1)  # [B, Nr]
        if batch_idx_nr is None:
            batch_idx_nr = torch.arange(B, device=device).unsqueeze(1).expand(B, Nr)
        # Diagonal
        A.index_put_((batch_idx_nr, ga, ga), gr, accumulate=True)
        A.index_put_((batch_idx_nr, gb, gb), gr, accumulate=True)
        # Off-diagonal
        A.index_put_((batch_idx_nr, ga, gb), -gr, accumulate=True)
        A.index_put_((batch_idx_nr, gb, ga), -gr, accumulate=True)

    # ---------- Stamp motors as stall resistors ----------
    if Nm > 0:
        cids_m = motors.component_ids  # [Nm]
        a_terms_m = start_idx[cids_m]
        b_terms_m = a_terms_m + 1
        gma = net_ids[:, a_terms_m]  # [B, Nm]
        gmb = net_ids[:, b_terms_m]  # [B, Nm]
        assert torch.all(gma >= 0).item() and torch.all(gmb >= 0).item(), (
            "All motor terminals must be connected"
        )
        Rm = motors.res_ohm.to(torch.float32)  # [Nm]
        assert torch.all(Rm > 0).item(), "Motor stall resistance must be > 0"
        gm = (1.0 / Rm).unsqueeze(0).expand(B, -1)  # [B, Nm]
        batch_idx_nm = torch.arange(B, device=device).unsqueeze(1).expand(B, Nm)
        # Diagonal
        A.index_put_((batch_idx_nm, gma, gma), gm, accumulate=True)
        A.index_put_((batch_idx_nm, gmb, gmb), gm, accumulate=True)
        # Off-diagonal
        A.index_put_((batch_idx_nm, gma, gmb), -gm, accumulate=True)
        A.index_put_((batch_idx_nm, gmb, gma), -gm, accumulate=True)

    # ---------- Stamp LEDs as linear resistors R = Vf / Imax ----------
    if Nl > 0:
        cids_l = leds.component_ids  # [Nl]
        a_terms_l = start_idx[cids_l]
        b_terms_l = a_terms_l + 1
        gla = net_ids[:, a_terms_l]  # [B, Nl]
        glb = net_ids[:, b_terms_l]  # [B, Nl]
        assert torch.all(gla >= 0).item() and torch.all(glb >= 0).item(), (
            "All LED terminals must be connected"
        )
        # R_led per LED from Vf and component max_current
        Imax_l = component_info.max_current[cids_l].to(torch.float32)  # [Nl]
        Vf = leds.vf_drop.to(torch.float32)  # [Nl]
        assert torch.all(Imax_l > 0).item(), "LED max_current must be > 0"
        assert torch.all(Vf > 0).item(), "LED vf_drop must be > 0"
        Rl = Vf / Imax_l  # [Nl]
        gl = (1.0 / Rl).unsqueeze(0).expand(B, -1)  # [B, Nl]
        batch_idx_nl = torch.arange(B, device=device).unsqueeze(1).expand(B, Nl)
        # Diagonal
        A.index_put_((batch_idx_nl, gla, gla), gl, accumulate=True)
        A.index_put_((batch_idx_nl, glb, glb), gl, accumulate=True)
        # Off-diagonal
        A.index_put_((batch_idx_nl, gla, glb), -gl, accumulate=True)
        A.index_put_((batch_idx_nl, glb, gla), -gl, accumulate=True)

    # ---------- Stamp switches/buttons as resistors with on/off resistance ----------
    if Ns > 0:
        cids_s = switches.component_ids  # [Ns]
        a_terms_s = start_idx[cids_s]
        b_terms_s = a_terms_s + 1
        gsa = net_ids[:, a_terms_s]  # [B, Ns]
        gsb = net_ids[:, b_terms_s]  # [B, Ns]
        assert torch.all(gsa >= 0).item() and torch.all(gsb >= 0).item(), (
            "All switch terminals must be connected"
        )
        on_res = switches.on_res_ohm.to(torch.float32)  # [Ns]
        off_res = switches.off_res_ohm.to(torch.float32)  # [Ns]
        assert torch.all(on_res > 0).item() and torch.all(off_res > 0).item(), (
            "Switch on/off resistances must be > 0"
        )
        # Per-batch on/off bits
        on_bits = state.switch_state.state_bits.to(torch.bool)  # [B, Ns]
        assert on_bits.shape[1] == Ns, "switch_state bits must match switch count"
        Rs = torch.where(on_bits, on_res.unsqueeze(0), off_res.unsqueeze(0))  # [B, Ns]
        gs = 1.0 / Rs  # [B, Ns]
        batch_idx_ns = torch.arange(B, device=device).unsqueeze(1).expand(B, Ns)
        # Diagonal
        A.index_put_((batch_idx_ns, gsa, gsa), gs, accumulate=True)
        A.index_put_((batch_idx_ns, gsb, gsb), gs, accumulate=True)
        # Off-diagonal
        A.index_put_((batch_idx_ns, gsa, gsb), -gs, accumulate=True)
        A.index_put_((batch_idx_ns, gsb, gsa), -gs, accumulate=True)

    # ---------- Stamp ideal voltage sources (first->second) ----------
    if Nv > 0:
        cids_v = vsources.component_ids  # [Nv]
        a_terms_v = start_idx[cids_v]
        b_terms_v = a_terms_v + 1
        gva = net_ids[:, a_terms_v]  # [B, Nv]
        gvb = net_ids[:, b_terms_v]  # [B, Nv]
        assert torch.all(gva >= 0).item() and torch.all(gvb >= 0).item(), (
            "All voltage source terminals must be connected"
        )
        ivar = T + torch.arange(Nv, device=device)  # [Nv]
        if batch_idx_nv is None:
            batch_idx_nv = torch.arange(B, device=device).unsqueeze(1).expand(B, Nv)
        ivar_b = ivar.unsqueeze(0).expand(B, Nv)
        ones = torch.ones((B, Nv), dtype=torch.float32, device=device)
        # Node-source couplings
        A.index_put_((batch_idx_nv, gva, ivar_b), ones, accumulate=True)
        A.index_put_((batch_idx_nv, ivar_b, gva), ones, accumulate=True)
        A.index_put_((batch_idx_nv, gvb, ivar_b), -ones, accumulate=True)
        A.index_put_((batch_idx_nv, ivar_b, gvb), -ones, accumulate=True)
        # RHS
        z.index_put_(
            (batch_idx_nv, ivar_b),
            vsources.voltage.to(torch.float32).unsqueeze(0).expand(B, Nv),
        )

    # ---------- Enforce ground per batch: V(ground) = 0 via row constraint ----------
    row_idx = torch.arange(B, device=device)
    A[row_idx, ground_gid] = 0.0
    A[row_idx, ground_gid, ground_gid] = 1.0
    z[row_idx, ground_gid] = 0.0

    # ---------- Enforce identity rows for nets that are not present to avoid singularity ----------
    unused_mask = ~present_mask_full  # [B, T]
    if torch.any(unused_mask):
        b_sel, k_sel = torch.nonzero(unused_mask, as_tuple=True)
        A[b_sel, k_sel, :] = 0.0
        A[b_sel, k_sel, k_sel] = 1.0
        z[b_sel, k_sel] = 0.0

    # ---------- Solve batched linear systems ----------
    x = torch.linalg.solve(A, z)  # [B, n_unknown]

    # ---------- Write terminal voltages ----------
    v_nodes = x[:, :T]  # [B, T]
    mask = assigned_mask
    bmat = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    out_terminal_voltage[mask] = v_nodes[bmat[mask], net_ids[mask]]

    # ---------- Write branch currents ----------
    # Resistors
    if Nr > 0:
        va = v_nodes.gather(1, ga)
        vb = v_nodes.gather(1, gb)
        Ir = (va - vb) / Rr.unsqueeze(0)
        comp_idx_r = cids_r.unsqueeze(0).expand(B, Nr)
        out_component_current[batch_idx_nr, comp_idx_r] = Ir
    # Motors (stall)
    if Nm > 0:
        va = v_nodes.gather(1, gma)
        vb = v_nodes.gather(1, gmb)
        Im = (va - vb) / Rm.unsqueeze(0)
        comp_idx_m = cids_m.unsqueeze(0).expand(B, Nm)
        out_component_current[batch_idx_nm, comp_idx_m] = Im
    # LEDs (linearized resistor model)
    if Nl > 0:
        va = v_nodes.gather(1, gla)
        vb = v_nodes.gather(1, glb)
        Il = (va - vb) / Rl.unsqueeze(0)
        comp_idx_l = cids_l.unsqueeze(0).expand(B, Nl)
        out_component_current[batch_idx_nl, comp_idx_l] = Il
    # Switches
    if Ns > 0:
        va = v_nodes.gather(1, gsa)
        vb = v_nodes.gather(1, gsb)
        Isw = (va - vb) / Rs  # [B, Ns]
        comp_idx_s = cids_s.unsqueeze(0).expand(B, Ns)
        out_component_current[batch_idx_ns, comp_idx_s] = Isw
    # Voltage sources: current variable is x[:, T:]
    if Nv > 0:
        Iv = -x[:, T : T + Nv]  # [B, Nv]
        comp_idx_v = cids_v.unsqueeze(0).expand(B, Nv)
        out_component_current[batch_idx_nv, comp_idx_v] = Iv

    # ---------- Write outputs back ----------
    state.terminal_voltage = out_terminal_voltage
    state.component_branch_current = out_component_current
    # Per-type outputs: LED luminosity and motor speed percentage in [0,1]
    if Nl > 0:
        Imax_l = (
            component_info.max_current[cids_l].to(torch.float32).unsqueeze(0)
        )  # [1, Nl]
        led_pct = (Il.abs() / Imax_l).clamp(0.0, 1.0)
        state.led_state.luminosity_pct = led_pct
    if Nm > 0:
        Imax_m = (
            component_info.max_current[cids_m].to(torch.float32).unsqueeze(0)
        )  # [1, Nm]
        motor_pct = (Im.abs() / Imax_m).clamp(0.0, 1.0)
        state.motor_state.speed_pct = motor_pct

    terminated = torch.zeros((B,), dtype=torch.bool, device=device)
    burned = torch.empty((0,), dtype=torch.long, device=device)
    return SolveResult(
        state=state, terminated=terminated, burned_component_indices=burned
    )
