import torch
import pytest
from repairs_components.geometry.fasteners import (
    Fastener,
    check_fastener_possible_insertion,
)
from repairs_components.processing.translation import are_quats_within_angle


# -----------------------------------------------
# === check_fastener_possible_insertion tests ===
# -----------------------------------------------
def test_simple_match_within_distance():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])
    batch = torch.tensor([0, 0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip, holes[0], batch, connection_dist_threshold=0.05
    )
    print("hole_idx: ", hole_idx)
    print("part_idx: ", part_idx)
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_no_hole_within_distance():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    batch = torch.tensor([0, 0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip, holes[0], batch, connection_dist_threshold=0.05
    )
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_multiple_batches():
    tip = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    holes = torch.tensor([[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]])
    batch = torch.tensor([0, 1])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip, holes, batch, connection_dist_threshold=0.05
    )
    print("hole_idx: ", hole_idx)
    print("part_idx: ", part_idx)
    assert hole_idx.tolist() == [0, -1]
    assert part_idx.tolist() == [0, -1]


def test_orientation_mask_rejects():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04]]])
    batch = torch.tensor([0])
    fast_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_quats = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # 180° around x

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip,
        holes[0],
        batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=30,
        part_hole_quats=hole_quats,
        active_fastener_quat=fast_quat,
    )
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_orientation_mask_accepts():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04]]])
    batch = torch.tensor([0])
    fast_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # ~10° around y axis
    hole_quats = torch.tensor([[0.9848, 0.0, 0.1736, 0.0]])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip,
        holes[0],
        batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=30,
        part_hole_quats=hole_quats,
        active_fastener_quat=fast_quat,
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_ignore_part_index_masks():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])
    batch = torch.tensor([0, 0])
    ignore_part_idx = torch.tensor([[0, -1]])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip,
        holes[0],
        batch,
        connection_dist_threshold=0.05,
        ignore_part_idx=ignore_part_idx,
    )
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_ignore_part_index_does_not_mask():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])
    batch = torch.tensor([0, 0])
    ignore_part_idx = torch.tensor([[-1, -1]])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip,
        holes[0],
        batch,
        connection_dist_threshold=0.05,
        ignore_part_idx=ignore_part_idx,
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_distance_tie_selects_first():
    tip = torch.tensor([[0.0, 0.0, 0.0]])
    holes = torch.tensor([[[0.0, 0.0, 0.04], [0.0, 0.0, -0.04]]])
    batch = torch.tensor([0, 0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip,
        holes[0],
        batch,
        connection_dist_threshold=0.05,
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


# === bd_geometry tests ===
@pytest.fixture
def fastener():
    return Fastener(
        constraint_b_active=True,
        initial_body_a="body_a",
        initial_body_b="body_b",
        length=15.0,
        diameter=5.0,
        b_depth=5.0,
        head_diameter=7.5,
        head_height=3.0,
        thread_pitch=0.5,
        screwdriver_name="screwdriver",
    )


@pytest.fixture
def bd_geometry(fastener):
    return fastener.bd_geometry()


def test_tip_is_in_correct_place(fastener, bd_geometry):
    assert bd_geometry.get_tip_pos_relative_to_center() * 1000 == torch.tensor(
        bd_geometry.joints["fastener_joint_tip"].joint_location
    )
    assert (bd_geometry.get_tip_pos_relative_to_center()[:2] == 0).all()
    assert bd_geometry.get_tip_pos_relative_to_center()[2] == -fastener.length / 1000
