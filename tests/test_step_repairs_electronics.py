import math

import genesis as gs
import pytest
import torch

from repairs_components.geometry.connectors.models.europlug import Europlug
from repairs_components.logic.electronics.component import ElectricalComponentsEnum
from repairs_components.logic.electronics.electronics_state import (
    connect_terminal_to_net_or_create_new,
    register_components_batch,
)
from repairs_components.training_utils.sim_state_global import (
    RepairsSimInfo,
    RepairsSimState,
)
from repairs_sim_step import step_electronics

# Rely on pytest discovery for fixtures from `tests/global_test_config.py`


@pytest.fixture(scope="module")
def scene_with_two_connectors(init_gs, test_device):
    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    europlug_male_pos = (0.0, -0.2, 0.02)
    europlug_female_pos = (-0.2, 0.0, 0.02)

    # "tool cube" and "fastener cube" as stubs for real geometry. Functionally the same.

    connector_europlug_male = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=europlug_male_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 0, 1)),  # blue
    )
    connector_europlug_female = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=europlug_female_pos,
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 1)),  # cyan
    )

    camera = scene.add_camera(
        pos=(1.0, 2.5, 3.5),
        lookat=(0.0, 0.0, 0.2),
        res=(256, 256),
    )
    scene.build(n_envs=1)
    camera.start_recording()

    entities = {
        "europlug_0_male@connector": connector_europlug_male,
        "europlug_0_female@connector": connector_europlug_female,
    }
    repairs_sim_state = RepairsSimState(device=test_device).unsqueeze(0)
    europlug_male = Europlug(in_sim_id=0)
    # hmm, and how do I register electronics? #TODO check translation.
    repairs_sim_state.electronics_state[0].register(europlug_male)

    return scene, entities, repairs_sim_state  # desired state defined separately.


def test_step_electronics_series_vsource_resistor(test_device):
    device = test_device
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

    cinfo, estate = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    # Parameters
    V = 9.0
    R = 3.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.resistors.resistance = torch.tensor([R], dtype=torch.float32, device=device)
    cinfo.has_electronics = True

    # Terminals
    v_s = int(cinfo.component_first_terminal_index[0].item())
    r_s = int(cinfo.component_first_terminal_index[1].item())
    v_a, v_b = v_s, v_s + 1
    r_a, r_b = r_s, r_s + 1

    # Connect V.a to R.a and V.b to R.b in both envs
    batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    term_ids = torch.tensor([v_a, v_b, v_a, v_b], dtype=torch.long, device=device)
    other_ids = torch.tensor([r_a, r_b, r_a, r_b], dtype=torch.long, device=device)
    estate = connect_terminal_to_net_or_create_new(
        estate, cinfo, batch_ids, term_ids, other_terminal_ids=other_ids
    )

    # Wrap in sim state/info and step electronics
    sim_state = torch.stack([RepairsSimState(device=device)] * B)
    sim_state.electronics_state = estate
    sim_info = RepairsSimInfo(
        "test_step_repairs_electronics_series_vsource_resistor", component_info=cinfo
    )
    sim_state, terminated, burned = step_electronics(sim_state, sim_info)
    st = sim_state.electronics_state

    I_expected = V / R
    assert torch.all(~terminated)
    for b in range(B):
        # Branch currents match V/R for both components
        assert math.isclose(
            float(st.component_branch_current[b, 0].item()), I_expected, rel_tol=1e-5
        )
        assert math.isclose(
            float(st.component_branch_current[b, 1].item()), I_expected, rel_tol=1e-5
        )
        # Voltage drops across elements equal V
        dv_r = float(
            st.terminal_voltage[b, r_a].item() - st.terminal_voltage[b, r_b].item()
        )
        dv_v = float(
            st.terminal_voltage[b, v_a].item() - st.terminal_voltage[b, v_b].item()
        )
        assert math.isclose(dv_r, V, rel_tol=1e-5)
        assert math.isclose(dv_v, V, rel_tol=1e-5)


def test_step_electronics_switch_led_series_batched(test_device):
    device = test_device
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
    # Per-component limits: V,I (LED Imax = 20mA)
    max_v = torch.tensor([12.0, 12.0, 5.0], dtype=torch.float32, device=device)
    max_i = torch.tensor([2.0, 2.0, 0.02], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2, 2], dtype=torch.long, device=device)

    cinfo, estate = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    # Parameters
    V = 2.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    Ron, Roff = 1.0, 1000.0
    cinfo.switches.on_res_ohm = torch.tensor([Ron], dtype=torch.float32, device=device)
    cinfo.switches.off_res_ohm = torch.tensor(
        [Roff], dtype=torch.float32, device=device
    )
    Vf = 2.0
    cinfo.leds.vf_drop = torch.tensor([Vf], dtype=torch.float32, device=device)
    cinfo.has_electronics = True

    # Batch0 ON, Batch1 OFF
    estate.switch_state.state_bits[:] = torch.tensor(
        [[True], [False]], dtype=torch.bool, device=device
    )

    # Terminals
    v_s = int(cinfo.component_first_terminal_index[0].item())
    s_s = int(cinfo.component_first_terminal_index[1].item())
    d_s = int(cinfo.component_first_terminal_index[2].item())
    v_a, v_b = v_s, v_s + 1
    s_a, s_b = s_s, s_s + 1
    d_a, d_b = d_s, d_s + 1

    batch_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
    # V+ -> S.a
    estate = connect_terminal_to_net_or_create_new(
        estate,
        cinfo,
        batch_ids,
        torch.tensor([v_a, v_a], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([s_a, s_a], dtype=torch.long, device=device),
    )
    # S.b -> D.a
    estate = connect_terminal_to_net_or_create_new(
        estate,
        cinfo,
        batch_ids,
        torch.tensor([s_b, s_b], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([d_a, d_a], dtype=torch.long, device=device),
    )
    # D.b -> V-
    estate = connect_terminal_to_net_or_create_new(
        estate,
        cinfo,
        batch_ids,
        torch.tensor([d_b, d_b], dtype=torch.long, device=device),
        other_terminal_ids=torch.tensor([v_b, v_b], dtype=torch.long, device=device),
    )

    # Wrap and step
    sim_state = torch.stack([RepairsSimState(device=device)] * B)
    sim_state.electronics_state = estate
    sim_info = RepairsSimInfo(
        "test_step_repairs_electronics_switch_led", component_info=cinfo
    )
    sim_state, terminated, burned = step_electronics(sim_state, sim_info)
    st = sim_state.electronics_state

    Rled = Vf / max_i[2].item()
    I_on = V / (Ron + Rled)
    I_off = V / (Roff + Rled)

    assert torch.all(~terminated)
    # Component indices: [V, S, D]
    # Batch 0 (ON)
    assert math.isclose(
        float(st.component_branch_current[0, 1].item()), I_on, rel_tol=1e-4
    )
    assert math.isclose(
        float(st.component_branch_current[0, 2].item()), I_on, rel_tol=1e-4
    )
    # Batch 1 (OFF)
    assert math.isclose(
        float(st.component_branch_current[1, 1].item()), I_off, rel_tol=1e-4
    )
    assert math.isclose(
        float(st.component_branch_current[1, 2].item()), I_off, rel_tol=1e-4
    )
    # LED luminosity pct
    Imax_led = max_i[2].item()
    led0 = float(st.led_state.luminosity_pct[0, 0].item())
    led1 = float(st.led_state.luminosity_pct[1, 0].item())
    assert math.isclose(led0, min(abs(I_on) / Imax_led, 1.0), rel_tol=1e-4)
    assert math.isclose(led1, min(abs(I_off) / Imax_led, 1.0), rel_tol=1e-4)


def test_step_electronics_motor_series(test_device):
    device = test_device
    B = 2
    names = ["V1@vsource", "M1@motor"]
    types = torch.tensor(
        [
            int(ElectricalComponentsEnum.VOLTAGE_SOURCE.value),
            int(ElectricalComponentsEnum.MOTOR.value),
        ],
        dtype=torch.long,
        device=device,
    )
    max_v = torch.tensor([24.0, 24.0], dtype=torch.float32, device=device)
    # Motor Imax = 2A
    max_i = torch.tensor([5.0, 2.0], dtype=torch.float32, device=device)
    tpc = torch.tensor([2, 2], dtype=torch.long, device=device)

    cinfo, estate = register_components_batch(
        names, types, max_v, max_i, tpc, device=device, batch_size=B
    )

    V = 12.0
    Rm = 6.0
    cinfo.vsources.voltage = torch.tensor([V], dtype=torch.float32, device=device)
    cinfo.motors.res_ohm = torch.tensor([Rm], dtype=torch.float32, device=device)
    cinfo.has_electronics = True

    # Terminals
    v_s = int(cinfo.component_first_terminal_index[0].item())
    m_s = int(cinfo.component_first_terminal_index[1].item())
    v_a, v_b = v_s, v_s + 1
    m_a, m_b = m_s, m_s + 1

    # Connect V.a to M.a and V.b to M.b in both envs
    batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    term_ids = torch.tensor([v_a, v_b, v_a, v_b], dtype=torch.long, device=device)
    other_ids = torch.tensor([m_a, m_b, m_a, m_b], dtype=torch.long, device=device)
    estate = connect_terminal_to_net_or_create_new(
        estate, cinfo, batch_ids, term_ids, other_terminal_ids=other_ids
    )

    # Wrap and step
    sim_state = torch.stack([RepairsSimState(device=device)] * B)
    sim_state.electronics_state = estate
    sim_info = RepairsSimInfo(
        "test_step_repairs_electronics_motor_series", component_info=cinfo
    )
    sim_state, terminated, burned = step_electronics(sim_state, sim_info)
    st = sim_state.electronics_state

    I_expected = V / Rm
    assert torch.all(~terminated)
    for b in range(B):
        # Currents equal V/R for both components
        assert math.isclose(
            float(st.component_branch_current[b, 0].item()), I_expected, rel_tol=1e-5
        )
        assert math.isclose(
            float(st.component_branch_current[b, 1].item()), I_expected, rel_tol=1e-5
        )
        # Voltage drops equal V
        dv_m = float(
            st.terminal_voltage[b, m_a].item() - st.terminal_voltage[b, m_b].item()
        )
        dv_v = float(
            st.terminal_voltage[b, v_a].item() - st.terminal_voltage[b, v_b].item()
        )
        assert math.isclose(dv_m, V, rel_tol=1e-5)
        assert math.isclose(dv_v, V, rel_tol=1e-5)

    # Motor speed pct = |I| / Imax_motor
    Imax_motor = max_i[1].item()
    expected_pct = min(abs(I_expected) / Imax_motor, 1.0)
    for b in range(B):
        assert math.isclose(
            float(st.motor_state.speed_pct[b, 0].item()), expected_pct, rel_tol=1e-4
        )


# --------------------------
# === step_electronics ===
# --------------------------

# note: step electronics tests were removed because the functionality of that method was a duplicate (yet)
# and likely be in the future, unless there will be a direct action on electronics, e.g. multimeter?
