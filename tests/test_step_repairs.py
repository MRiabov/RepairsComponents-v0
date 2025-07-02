import pytest
import torch
from types import SimpleNamespace
from repairs_components.training_utils.sim_state_global import RepairsSimState
from repairs_components.logic.physical_state import PhysicalState
from repairs_components.logic.electronics.electronics_state import ElectronicsState

import repairs_sim_step as step_mod
from repairs_sim_step import step_repairs


# Dummy classes for testing
class DummyScene:
    def __init__(self, n_envs):
        self.n_envs = n_envs


class DummyTool:
    def __init__(self, picked_up_fastener_name=None):
        self.picked_up_fastener_name = picked_up_fastener_name


class DummyToolState:
    def __init__(self, tool):
        self.current_tool = tool


@pytest.fixture(autouse=True)
def mock_translate(monkeypatch):
    # Patch translate and activate at step_repairs module
    def fake_translate(scene, gs_entities, sim_state):
        B = scene.n_envs
        tip = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        holes = {"h": torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])}
        male = {"m": torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])}
        female = {"f": torch.tensor([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]])}
        return sim_state, tip, holes, male, female

    monkeypatch.setattr(step_mod, "translate_genesis_to_python", fake_translate)
    monkeypatch.setattr(step_mod, "activate_connection", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        step_mod,
        "check_connections",
        lambda male, female, connection_threshold=2.5: torch.tensor(
            [[0, 0, 0]], dtype=torch.long
        ),  # type: ignore
    )
    yield


@pytest.mark.parametrize("has_electronics", [True, False])
def test_step_repairs(has_electronics):
    # Setup
    B = 2
    scene = DummyScene(B)
    # actions: only env1 tries to screw
    actions = {"screw_in": torch.tensor([False, True])}
    gs_entities = {"h": None}
    # Build sim_state
    sim_state = RepairsSimState(B)
    # stub diff
    sim_state.diff = lambda other: ({}, torch.zeros(B, dtype=torch.int64))
    # stub physical connect
    sim_state.physical_state = []  # type: ignore[reportGeneralTypeIssues]
    for _ in range(B):
        ps = PhysicalState()
        ps.calls = []  # type: ignore

        def stub_connect(self, fastener_name, body_a, body_b):
            self.calls.append((fastener_name, body_a))

        ps.connect_fastener_to_one_body = stub_connect.__get__(ps, PhysicalState)  # type: ignore
        sim_state.physical_state.append(ps)
    # stub tool
    sim_state.tool_state = [DummyToolState(DummyTool()), DummyToolState(DummyTool("h"))]  # type: ignore[reportGeneralTypeIssues]
    # stub electronics clear and connect
    sim_state.electronics_state = []  # type: ignore[reportGeneralTypeIssues]
    for _ in range(B):
        es = ElectronicsState()
        es.calls = []  # type: ignore

        def stub_clear(self):
            self.calls = []

        def stub_elec_connect(self, m, f):
            self.calls.append((m, f))

        es.clear_connections = stub_clear.__get__(es, ElectronicsState)  # type: ignore
        es.connect = stub_elec_connect.__get__(es, ElectronicsState)  # type: ignore
        sim_state.electronics_state.append(es)
    sim_state.has_electronics = has_electronics
    desired_state = sim_state

    # Call step
    # ignore type mismatch for sim_state and desired_state stubs
    success, total_diff_left, out_state, diff = step_repairs(
        scene,
        actions,
        gs_entities,
        sim_state,
        desired_state,  # type: ignore[arg-type]
    )

    # success should be all True
    assert isinstance(success, torch.Tensor)
    assert success.all().item()

    # Physical insertion: only if has no electronics or even if electronics, physical always processed
    # With our fake positions, insert_indices = [0, -1]
    # screw_mask = [False, True] -> valid_insert = [False, False]
    assert sim_state.physical_state[0].calls == []
    assert sim_state.physical_state[1].calls == []

    if has_electronics:
        # Only env0 has male-female within threshold
        assert sim_state.electronics_state[0].calls == [(0, 0)]
        assert sim_state.electronics_state[1].calls == []
    else:
        # No electronics applied
        assert sim_state.electronics_state[0].calls == []
        assert sim_state.electronics_state[1].calls == []
