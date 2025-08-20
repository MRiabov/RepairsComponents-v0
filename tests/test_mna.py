import math

import torch

from repairs_components.logic.electronics.component import ElectricalComponentsEnum
from repairs_components.logic.electronics.electronics_state import (
    connect_terminal_to_net_or_create_new,
    register_components_batch,
)
from repairs_components.logic.electronics.mna import solve_dc_once


def test_dc_series_vsource_resistor():
    device = torch.device("cpu")
    B = 1
    names = ["V1@vsource", "R1@resistor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([50.0, 50.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    # Fill parameters (infos are unbatched across envs)
    V = 9.0
    R = 3.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor([R], dtype=torch.float32, device=device)

    # Terminals: [V.a, V.b, R.a, R.b]
    v_start = cinfo.component_first_terminal_index[0].item()
    r_start = cinfo.component_first_terminal_index[1].item()
    v_a, v_b = int(v_start), int(v_start + 1)
    r_a, r_b = int(r_start), int(r_start + 1)

    b0 = torch.tensor([0, 0], dtype=torch.long, device=device)
    # Connect V.a to R.a -> net 0; V.b to R.b -> net 1
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b0,
        torch.tensor([v_a, v_b], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([r_a, r_b], dtype=torch.long, device=device),
    )

    # Solve
    res = solve_dc_once(cinfo, state)
    st = res.state

    # Currents equal V/R
    rid = 1  # second component
    vid = 0  # first component
    I_expected = V / R
    assert math.isclose(
        float(st.component_branch_current[0, rid].item()), I_expected, rel_tol=1e-5
    )
    assert math.isclose(
        float(st.component_branch_current[0, vid].item()), I_expected, rel_tol=1e-5
    )

    # Voltage difference across each component equals V
    dv_r = float(
        st.terminal_voltage[0, r_a].item() - st.terminal_voltage[0, r_b].item()
    )
    dv_v = float(
        st.terminal_voltage[0, v_a].item() - st.terminal_voltage[0, v_b].item()
    )
    assert math.isclose(dv_r, V, rel_tol=1e-5)
    assert math.isclose(dv_v, V, rel_tol=1e-5)


def test_dc_series_two_resistors():
    device = torch.device("cpu")
    B = 1
    names = ["V1@vsource", "R1@resistor", "R2@resistor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    V = 12.0
    R1 = 2.0
    R2 = 4.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor(
        [R1, R2], dtype=torch.float32, device=device
    )

    v_s = int(cinfo.component_first_terminal_index[0].item())
    r1_s = int(cinfo.component_first_terminal_index[1].item())
    r2_s = int(cinfo.component_first_terminal_index[2].item())
    v_a, v_b = v_s, v_s + 1
    r1_a, r1_b = r1_s, r1_s + 1
    r2_a, r2_b = r2_s, r2_s + 1

    b = torch.tensor([0], dtype=torch.long, device=device)
    # net0: V.a - R1.a
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([v_a], device=device),
        other_terminal_ids=torch.tensor([r1_a], device=device),
    )
    # net1: R1.b - R2.a
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([r1_b], device=device),
        other_terminal_ids=torch.tensor([r2_a], device=device),
    )
    # net2: V.b - R2.b
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([v_b], device=device),
        other_terminal_ids=torch.tensor([r2_b], device=device),
    )

    st = solve_dc_once(cinfo, state).state

    I_series = V / (R1 + R2)
    # Branch currents
    assert math.isclose(
        float(st.component_branch_current[0, 0].item()), I_series, rel_tol=1e-5
    )  # V1
    assert math.isclose(
        float(st.component_branch_current[0, 1].item()), I_series, rel_tol=1e-5
    )  # R1
    assert math.isclose(
        float(st.component_branch_current[0, 2].item()), I_series, rel_tol=1e-5
    )  # R2

    # Voltage drops
    dv_r1 = float(
        st.terminal_voltage[0, r1_a].item() - st.terminal_voltage[0, r1_b].item()
    )
    dv_r2 = float(
        st.terminal_voltage[0, r2_a].item() - st.terminal_voltage[0, r2_b].item()
    )
    dv_v = float(
        st.terminal_voltage[0, v_a].item() - st.terminal_voltage[0, v_b].item()
    )
    assert math.isclose(dv_r1, I_series * R1, rel_tol=1e-5)
    assert math.isclose(dv_r2, I_series * R2, rel_tol=1e-5)
    assert math.isclose(dv_v, V, rel_tol=1e-5)


def test_dc_parallel_two_resistors():
    device = torch.device("cpu")
    B = 1
    names = ["V1@vsource", "R1@resistor", "R2@resistor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    V = 5.0
    R1 = 10.0
    R2 = 5.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor(
        [R1, R2], dtype=torch.float32, device=device
    )

    v_s = int(cinfo.component_first_terminal_index[0].item())
    r1_s = int(cinfo.component_first_terminal_index[1].item())
    r2_s = int(cinfo.component_first_terminal_index[2].item())
    v_a, v_b = v_s, v_s + 1
    r1_a, r1_b = r1_s, r1_s + 1
    r2_a, r2_b = r2_s, r2_s + 1

    b = torch.tensor([0], dtype=torch.long, device=device)
    # Positive rail net0: V.a - R1.a
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([v_a], device=device),
        other_terminal_ids=torch.tensor([r1_a], device=device),
    )
    # Merge R2.a into net0 via V.a
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([r2_a], device=device),
        other_terminal_ids=torch.tensor([v_a], device=device),
    )
    # Negative rail net1: V.b - R1.b
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([v_b], device=device),
        other_terminal_ids=torch.tensor([r1_b], device=device),
    )
    # Merge R2.b into net1 via V.b
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        b,
        torch.tensor([r2_b], device=device),
        other_terminal_ids=torch.tensor([v_b], device=device),
    )

    st = solve_dc_once(cinfo, state).state

    I1 = V / R1
    I2 = V / R2
    Is = I1 + I2
    # Branch currents
    assert math.isclose(
        float(st.component_branch_current[0, 1].item()), I1, rel_tol=1e-5
    )  # R1
    assert math.isclose(
        float(st.component_branch_current[0, 2].item()), I2, rel_tol=1e-5
    )  # R2
    assert math.isclose(
        float(st.component_branch_current[0, 0].item()), Is, rel_tol=1e-5
    )  # V1

    # Voltage across each resistor equals source V
    dv_r1 = float(
        st.terminal_voltage[0, r1_a].item() - st.terminal_voltage[0, r1_b].item()
    )
    dv_r2 = float(
        st.terminal_voltage[0, r2_a].item() - st.terminal_voltage[0, r2_b].item()
    )
    dv_v = float(
        st.terminal_voltage[0, v_a].item() - st.terminal_voltage[0, v_b].item()
    )
    assert math.isclose(dv_r1, V, rel_tol=1e-5)
    assert math.isclose(dv_r2, V, rel_tol=1e-5)
    assert math.isclose(dv_v, V, rel_tol=1e-5)


def test_batched_two_identical_series_envs():
    device = torch.device("cpu")
    B = 2
    names = ["V1@vsource", "R1@resistor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([50.0, 50.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([10.0, 10.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    V = 9.0
    R = 3.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor([R], dtype=torch.float32, device=device)

    v_s = int(cinfo.component_first_terminal_index[0].item())
    r_s = int(cinfo.component_first_terminal_index[1].item())
    v_a, v_b = v_s, v_s + 1
    r_a, r_b = r_s, r_s + 1

    # Connect both envs in one call using expanded batch ids
    batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    term_ids = torch.tensor([v_a, v_b, v_a, v_b], dtype=torch.long, device=device)
    other_ids = torch.tensor([r_a, r_b, r_a, r_b], dtype=torch.long, device=device)
    state = connect_terminal_to_net_or_create_new(
        state, cinfo, batch_ids, term_ids, other_terminal_ids=other_ids
    )

    st = solve_dc_once(cinfo, state).state

    I_expected = V / R
    for b in range(B):
        assert math.isclose(
            float(st.component_branch_current[b, 1].item()), I_expected, rel_tol=1e-5
        )
        assert math.isclose(
            float(st.component_branch_current[b, 0].item()), I_expected, rel_tol=1e-5
        )


def test_torch_compile_mna_series_parallel():
    device = torch.device("cpu")
    B = 2
    names = ["V1@vsource", "R1@resistor", "R2@resistor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
            int(ElectricalComponentsEnum.RESISTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    V = 9.0
    R1, R2 = 3.0, 6.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor(
        [R1, R2], dtype=torch.float32, device=device
    )

    v_s = int(cinfo.component_first_terminal_index[0].item())
    r1_s = int(cinfo.component_first_terminal_index[1].item())
    r2_s = int(cinfo.component_first_terminal_index[2].item())

    v_a, v_b = v_s, v_s + 1
    r1_a, r1_b = r1_s, r1_s + 1
    r2_a, r2_b = r2_s, r2_s + 1

    # Build a small network per batch: V in parallel with R1 and R2
    # Batch ids interleaved to exercise batched scatter
    batch_ids = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device
    )
    term_ids = torch.tensor(
        [
            v_a,
            v_b,  # V terminals to net1
            r1_a,
            r1_b,  # R1 to net1
            r2_a,
            r2_b,  # R2 to net1
            v_a,
            v_b,  # batch 1
            r1_a,
            r1_b,
            r2_a,
            r2_b,
        ],
        dtype=torch.long,
        device=device,
    )
    other_ids = torch.tensor(
        [
            r1_a,
            r1_b,  # tie V to R1 to create a net
            v_a,
            v_b,  # tie back
            v_a,
            v_b,  # tie R2 in via V
            r1_a,
            r1_b,
            v_a,
            v_b,
            v_a,
            v_b,
        ],
        dtype=torch.long,
        device=device,
    )

    state = connect_terminal_to_net_or_create_new(
        state, cinfo, batch_ids, term_ids, other_terminal_ids=other_ids
    )

    # Eager solve (reference)
    eager = solve_dc_once(cinfo, state).state

    # Compiled solve
    compiled_fn = torch.compile(solve_dc_once)
    compiled = compiled_fn(cinfo, state).state

    # Check equality of branch currents (R1, R2)
    I1 = V / R1
    I2 = V / R2
    for b in range(B):
        assert math.isclose(
            float(eager.component_branch_current[b, 1].item()), I1, rel_tol=1e-5
        )
        assert math.isclose(
            float(eager.component_branch_current[b, 2].item()), I2, rel_tol=1e-5
        )
        assert math.isclose(
            float(compiled.component_branch_current[b, 1].item()), I1, rel_tol=1e-5
        )
        assert math.isclose(
            float(compiled.component_branch_current[b, 2].item()), I2, rel_tol=1e-5
        )

    # Check node voltages match eager
    assert torch.allclose(
        compiled.terminal_voltage, eager.terminal_voltage, atol=1e-6, rtol=0.0
    )


def test_switch_led_series_with_outputs_and_compile():
    device = torch.device("cpu")
    B = 2
    names = ["V1@vsource", "S1@button", "D1@led"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.BUTTON.value),
            int(ElectricalComponentsEnum.LED.value),
        ],
        dtype=torch.long,
        device=device,
    )
    # Per-component limits: V,I
    max_v = torch.tensor([12.0, 12.0, 5.0], dtype=torch.float32, device=device)
    max_i = torch.tensor(
        [2.0, 2.0, 0.02], dtype=torch.float32, device=device
    )  # LED Imax=20mA
    tpc = torch.tensor([2, 2, 2], dtype=torch.long, device=device)

    cinfo, state = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    # Parameters
    V = 2.0  # volts
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)

    # Switch on/off resistances
    Ron, Roff = 1.0, 1000.0
    cinfo.switches.on_res_ohm = torch.tensor([Ron], dtype=torch.float32, device=device)
    cinfo.switches.off_res_ohm = torch.tensor(
        [Roff], dtype=torch.float32, device=device
    )

    # LED forward drop Vf; effective R = Vf / Imax
    Vf = 2.0
    cinfo.leds.vf_drop = torch.tensor([Vf], dtype=torch.float32, device=device)

    # Turn batch0 ON, batch1 OFF
    state.switch_state.state_bits[:] = torch.tensor(
        [[True], [False]], dtype=torch.bool, device=device
    )

    # Terminals
    v_s = int(cinfo.component_first_terminal_index[0].item())
    s_s = int(cinfo.component_first_terminal_index[1].item())
    d_s = int(cinfo.component_first_terminal_index[2].item())

    v_a, v_b = v_s, v_s + 1
    s_a, s_b = s_s, s_s + 1
    d_a, d_b = d_s, d_s + 1

    # Build series: V+ -> S.a -> S.b -> D.a -> D.b -> V-
    # Connect V+ to S.a
    batch_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        batch_ids,
        torch.tensor([v_a, v_a], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([s_a, s_a], dtype=torch.long, device=device),
    )
    # Connect S.b to D.a
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        batch_ids,
        torch.tensor([s_b, s_b], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([d_a, d_a], dtype=torch.long, device=device),
    )
    # Connect D.b to V-
    state = connect_terminal_to_net_or_create_new(
        state,
        cinfo,
        batch_ids,
        torch.tensor([d_b, d_b], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([v_b, v_b], dtype=torch.long, device=device),
    )

    # Eager and compiled
    eager = solve_dc_once(cinfo, state).state
    compiled = torch.compile(solve_dc_once)(cinfo, state).state

    # Expected currents per batch
    Rled = Vf / max_i[2].item()
    I_on = V / (Ron + Rled)
    I_off = V / (Roff + Rled)

    # Component indices: [V, S, D]
    # Check eager currents
    assert math.isclose(
        float(eager.component_branch_current[0, 1].item()), I_on, rel_tol=1e-4
    )
    assert math.isclose(
        float(eager.component_branch_current[1, 1].item()), I_off, rel_tol=1e-4
    )
    assert math.isclose(
        float(eager.component_branch_current[0, 2].item()), I_on, rel_tol=1e-4
    )
    assert math.isclose(
        float(eager.component_branch_current[1, 2].item()), I_off, rel_tol=1e-4
    )

    # Check LED luminosity pct = |I_LED| / Imax
    Imax = max_i[2].item()
    led0 = float(eager.led_state.luminosity_pct[0, 0].item())
    led1 = float(eager.led_state.luminosity_pct[1, 0].item())
    assert math.isclose(led0, min(abs(I_on) / Imax, 1.0), rel_tol=1e-4)
    assert math.isclose(led1, min(abs(I_off) / Imax, 1.0), rel_tol=1e-4)

    # Compiled matches eager
    assert torch.allclose(compiled.terminal_voltage, eager.terminal_voltage, atol=1e-6)
    assert torch.allclose(
        compiled.component_branch_current, eager.component_branch_current, atol=1e-6
    )
    assert torch.allclose(
        compiled.led_state.luminosity_pct, eager.led_state.luminosity_pct, atol=1e-6
    )
