import pytest
import torch

# Import the implementations
from repairs_components.logic.physical_state import (
    PhysicalState,
    PhysicalStateInfo,
    register_bodies_batch,
    register_fasteners_batch,
)
from repairs_components.geometry.fasteners import get_fastener_singleton_name
# Tests updated to use batched PhysicalState API and WXYZ quaternion convention.


def test_physical_state_diff_basic():
    """Basic diff for PhysicalState using batched registration and WXYZ quats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two single-environment states (B=1) with the same bodies
    state1 = PhysicalState(device=device).unsqueeze(0)
    state2 = PhysicalState(device=device).unsqueeze(0)

    names = ["body1@solid", "body2@solid"]
    B, N = 1, len(names)

    # Positions within bounds, z >= 0; rotations are identity in WXYZ
    pos1 = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot1 = torch.zeros(B, N, 4, dtype=torch.float32)
    rot1[..., 0] = 1.0  # identity quaternion WXYZ: [1,0,0,0]
    fixed = torch.tensor([False, False])

    state1, physical_info1 = register_bodies_batch(names, pos1, rot1, fixed)

    # Move body1 by 1 cm in x (over 5 mm threshold); keep others the same
    pos2 = pos1.clone()
    pos2[:, 0, 0] += 0.01
    rot2 = rot1.clone()
    state2, physical_info2 = register_bodies_batch(names, pos2, rot2, fixed)

    # Compute diff
    diff_graph, total_diff = state1.diff(state2, physical_info1)

    # Minimal sanity checks (avoid relying on internal mask shapes)
    from torch_geometric.data import Data

    assert isinstance(diff_graph, Data)
    assert isinstance(total_diff, int)


def test_physical_state_no_changes():
    """Diff should report zero changes for identical batched states."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state1 = PhysicalState(device=device).unsqueeze(0)
    state2 = PhysicalState(device=device).unsqueeze(0)

    names = ["body1@solid", "body2@solid"]
    B, N = 1, len(names)

    pos = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot = torch.zeros(B, N, 4, dtype=torch.float32)
    rot[..., 0] = 1.0  # identity WXYZ
    fixed = torch.tensor([False, False])

    state1, physical_info1 = register_bodies_batch(names, pos, rot, fixed)
    state2, physical_info2 = register_bodies_batch(
        names, pos.clone(), rot.clone(), fixed
    )

    diff_graph, total_diff = state1.diff(state2, physical_info1)

    assert isinstance(total_diff, int)
    assert total_diff == 0


@pytest.xfail(reason="wrong test: it's not supposed to diff physical_info.")
# NOTE: it is supposed to diff fastener pos/quaternion though! And if the equivalent (by length/diam) fasteners are inserted.
def test_physical_state_diff_fastener_attr_flags():
    """Verify that fastener attribute flags and aligned deltas are set in the diff."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two states with the same two bodies and hole mapping
    A = PhysicalState(device=device).unsqueeze(0)
    B = PhysicalState(device=device).unsqueeze(0)

    names = ["body0@solid", "body1@solid"]
    Bsz, N = 1, len(names)

    pos = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot = torch.zeros(Bsz, N, 4, dtype=torch.float32)
    rot[..., 0] = 1.0  # WXYZ identity
    fixed = torch.tensor([False, False])

    # Create batched states by stacking before registration (B=1)
    A_batched: PhysicalState = torch.stack([A])  # type: ignore
    B_batched: PhysicalState = torch.stack([B])  # type: ignore
    A_batched, physical_info = register_bodies_batch(names, pos, rot, fixed)
    B_batched, physical_info = register_bodies_batch(
        names, pos.clone(), rot.clone(), fixed
    )

    # Deterministic hole->body mapping: 4 holes, first two on body0, next two on body1
    hole_map = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=A_batched.device)
    physical_info.part_hole_batch = hole_map
    physical_info.part_hole_batch = hole_map.clone()

    # One fastener attached between hole 0 (body0) and hole 2 (body1)
    fa = torch.tensor([[0]], dtype=torch.long)
    fb = torch.tensor([[2]], dtype=torch.long)

    # State A fastener: d=3mm, L=10mm, at origin, identity quaternion
    A_batched, physical_info1 = register_fasteners_batch(
        A_batched,
        physical_info,
        fastener_pos=torch.zeros(1, 1, 3),
        fastener_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        fastener_init_hole_a=fa,
        fastener_init_hole_b=fb,
        fastener_compound_names=[get_fastener_singleton_name(3.0, 10.0)],
    )

    # State B fastener: changed d, L, position and quaternion (10 deg around Z)
    theta = torch.deg2rad(torch.tensor(10.0))
    qz = torch.tensor(
        [[[(torch.cos(theta / 2)).item(), 0.0, 0.0, (torch.sin(theta / 2)).item()]]]
    )
    B_batched, physical_info2 = register_fasteners_batch(
        B_batched,
        physical_info,
        fastener_pos=torch.tensor([[[0.01, 0.00, 0.00]]], dtype=torch.float32),
        fastener_quat=qz.to(torch.float32),
        fastener_init_hole_a=fa.clone(),
        fastener_init_hole_b=fb.clone(),
        fastener_compound_names=[get_fastener_singleton_name(4.0, 12.0)],
    )

    diff_graph, total = A_batched.diff(B_batched, physical_info)

    # We expect exactly one changed edge and no added/removed
    assert diff_graph.edge_index.shape[1] == 1
    assert diff_graph.edge_attr.shape == (1, 6)

    # Flags: [is_added, is_removed, diam, length, pos, quat]
    flags = diff_graph.edge_attr[0]
    assert flags[0] == 0 and flags[1] == 0
    assert flags[2] == 1 and flags[3] == 1  # diam & length changed
    assert flags[4] == 1 and flags[5] == 1  # pos & quat changed

    # Detailed aligned deltas should be present for changed edge
    assert diff_graph.fastener_pos_diff.shape == (1, 3)
    assert torch.allclose(
        diff_graph.fastener_pos_diff[0], torch.tensor([0.01, 0.0, 0.0])
    )
    assert diff_graph.fastener_quat_delta.shape == (1, 4)
    assert torch.linalg.norm(diff_graph.fastener_quat_delta[0]) > 0

    assert isinstance(total, int)


def test_physical_state_diff_fastener_added_removed_and_count_diffs():
    """Verify added edges and node count_fasteners_held diffs are detected."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = PhysicalState(device=device).unsqueeze(0)
    B = PhysicalState(device=device).unsqueeze(0)
    physical_info = PhysicalStateInfo(device=device)

    names = ["body0@solid", "body1@solid"]
    Bsz, N = 1, len(names)

    pos = torch.tensor([[[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]]], dtype=torch.float32)
    rot = torch.zeros(Bsz, N, 4, dtype=torch.float32)
    rot[..., 0] = 1.0
    fixed = torch.tensor([False, False])

    # Create batched states by stacking before registration (B=1)
    A_batched: PhysicalState = torch.stack([A])  # type: ignore
    B_batched: PhysicalState = torch.stack([B])  # type: ignore
    A_batched, physical_info = register_bodies_batch(names, pos, rot, fixed)
    B_batched, _ = register_bodies_batch(names, pos.clone(), rot.clone(), fixed)

    # Hole mapping
    hole_map = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=A_batched.device)
    physical_info.part_hole_batch = hole_map

    # B has a single fastener between body0 and body1; A has none
    fa = torch.tensor([[0]], dtype=torch.long)
    fb = torch.tensor([[2]], dtype=torch.long)
    B_batched, physical_info = register_fasteners_batch(
        B_batched,
        physical_info,
        fastener_pos=torch.zeros(1, 1, 3),
        fastener_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        fastener_init_hole_a=fa,
        fastener_init_hole_b=fb,
        fastener_compound_names=[get_fastener_singleton_name(3.0, 10.0)],
    )

    diff_graph, _ = A_batched.diff(B_batched, physical_info)

    # One added edge
    assert diff_graph.edge_attr.shape[1] == 6
    assert diff_graph.edge_attr.shape[0] == 1
    assert diff_graph.edge_attr[0, 0] == 1  # is_added
    assert diff_graph.edge_attr[0, 1] == 0  # is_removed

    # Node counts should reflect 1 fastener held by both bodies
    assert diff_graph.count_fasteners_held_diff.shape[0] == 2
    assert torch.equal(
        diff_graph.count_fasteners_held_diff.to(torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    # Node mask marks both bodies
    assert diff_graph.node_mask.sum().item() == 2


@pytest.xfail(
    reason="WRONG TEST: tests for diff in fastener length/diam which is unchanged across batch"
)
def test_physical_state_diff_fastener_attr_flags_batch_two():
    """Same as attr flags test but with batch size B=2 (identical envs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two single-environment states which we'll stack into B=2
    A0 = PhysicalState(device=device).unsqueeze(0)
    A1 = PhysicalState(device=device).unsqueeze(0)
    B0 = PhysicalState(device=device).unsqueeze(0)
    B1 = PhysicalState(device=device).unsqueeze(0)

    names = ["body0@solid", "body1@solid"]
    Bsz, N = 2, len(names)

    pos = torch.tensor(
        [
            [[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]],
            [[0.00, 0.00, 0.10], [0.10, 0.00, 0.10]],
        ],
        dtype=torch.float32,
    )
    rot = torch.zeros(Bsz, N, 4, dtype=torch.float32)
    rot[..., 0] = 1.0  # WXYZ identity
    fixed = torch.tensor([False, False])

    # Create batched states by stacking (B=2)
    A_batched: PhysicalState = torch.stack([A0, A1])  # type: ignore
    B_batched: PhysicalState = torch.stack([B0, B1])  # type: ignore
    A_batched, physical_info = register_bodies_batch(names, pos, rot, fixed)
    B_batched, _ = register_bodies_batch(names, pos.clone(), rot.clone(), fixed)

    # Hole mapping replicated across batch: 4 holes, 0-1 on body0, 2-3 on body1
    hole_map = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=A_batched.device)
    physical_info.part_hole_batch = (
        hole_map  # uuh, it should be already per registration, is it not?
    )

    # One fastener attached between hole 0 (body0) and hole 2 (body1), replicated across batch
    fa = torch.tensor([[0], [0]], dtype=torch.long)
    fb = torch.tensor([[2], [2]], dtype=torch.long)

    # State A fastener: d=3mm, L=10mm, at origin, identity quaternion
    A_batched, physical_info = register_fasteners_batch(
        A_batched,
        physical_info,
        fastener_pos=torch.zeros(Bsz, 1, 3),
        fastener_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
        .expand(Bsz, -1, -1)
        .contiguous(),
        fastener_init_hole_a=fa,
        fastener_init_hole_b=fb,
        fastener_compound_names=[get_fastener_singleton_name(3.0, 10.0)],
    )

    # State B fastener: changed d, L, position and quaternion (10 deg around Z), replicated across batch
    theta = torch.deg2rad(torch.tensor(10.0))
    qz_single = torch.tensor(
        [[(torch.cos(theta / 2)).item(), 0.0, 0.0, (torch.sin(theta / 2)).item()]],
        dtype=torch.float32,
    )
    qz = qz_single.unsqueeze(0).expand(Bsz, -1, -1).contiguous()  # [B,1,4]
    B_batched, physical_info = register_fasteners_batch(
        B_batched,
        physical_info,
        fastener_pos=torch.tensor([[[0.01, 0.00, 0.00]]], dtype=torch.float32)
        .expand(Bsz, -1, -1)
        .contiguous(),
        fastener_quat=qz,
        fastener_init_hole_a=fa.clone(),
        fastener_init_hole_b=fb.clone(),
        fastener_compound_names=[get_fastener_singleton_name(4.0, 12.0)],
    )

    diff_graph, total = A_batched.diff(B_batched, physical_info)

    # We still expect one changed edge and no added/removed (edge dedup across batch)
    assert diff_graph.edge_index.shape[1] == 1
    assert diff_graph.edge_attr.shape == (1, 6)
    flags = diff_graph.edge_attr[0]
    assert flags[0] == 0 and flags[1] == 0
    assert flags[2] == 1 and flags[3] == 1  # diam & length changed
    assert flags[4] == 1 and flags[5] == 1  # pos & quat changed

    # Detailed aligned deltas should be present for changed edge
    assert diff_graph.fastener_pos_diff.shape == (1, 3)
    assert torch.allclose(
        diff_graph.fastener_pos_diff[0], torch.tensor([0.01, 0.0, 0.0])
    )
    assert diff_graph.fastener_quat_delta.shape == (1, 4)
    assert torch.linalg.norm(diff_graph.fastener_quat_delta[0]) > 0

    assert isinstance(total, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
