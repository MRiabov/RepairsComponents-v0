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
    # for batch of 1, with two holes, both of which in the first part, test that one close hole is selected.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])  # [B,3]
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])  # [B,H,3]
    part_hole_batch = torch.tensor([0, 0])  # [H]

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos, part_hole_positions, part_hole_batch, connection_dist_threshold=0.05
    )
    assert hole_idx.shape == part_idx.shape
    assert hole_idx.ndim == 1
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_no_hole_within_distance():
    # for batch of 1, with two holes, both of which in the first part, test that all are too far.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    part_hole_batch = torch.tensor([0, 0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos, part_hole_positions, part_hole_batch, connection_dist_threshold=0.05
    )
    assert hole_idx.shape == part_idx.shape
    assert hole_idx.ndim == 1
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_batch():
    # for batch of 2, with two holes each, test that the closest hole is selected in each part.
    tip_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    part_hole_positions = torch.tensor(
        [[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]], [[0.2, 0.0, 0.0], [0.1, 0.0, 0.03]]]
    )
    hole_batch = torch.tensor([0, 1])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos, part_hole_positions, hole_batch, connection_dist_threshold=0.05
    )
    assert hole_idx.shape == part_idx.shape
    assert hole_idx.ndim == 1
    assert hole_idx.tolist() == [0, 1]
    assert part_idx.tolist() == [0, 1]


def test_orientation_mask_rejects():
    # for batch of 1, with one hole, test that the hole is rejected if it is not within angle threshold.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04]]])
    part_hole_batch = torch.tensor([0])
    fast_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_quats = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])  # 180° around x

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=30,
        part_hole_quats=hole_quats,
        active_fastener_quat=fast_quat,
    )
    assert hole_idx.shape == part_idx.shape
    assert hole_idx.ndim == 1
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_orientation_mask_accepts():
    # for batch of 1, with one hole, test that the hole is accepted if it is within angle threshold.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04]]])
    part_hole_batch = torch.tensor([0])
    fast_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_quats = torch.tensor([[[0.9848, 0.0, 0.1736, 0.0]]])  # ~10° around y

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=30,
        part_hole_quats=hole_quats,
        active_fastener_quat=fast_quat,
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_ignore_part_index_masks():
    # for batch of 1, with two holes, test that the hole in the ignored part is not selected.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])
    part_hole_batch = torch.tensor([0, 0])
    ignore_part_idx = torch.tensor([0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        ignore_part_idx=ignore_part_idx,
    )
    assert hole_idx.shape == part_idx.shape
    assert hole_idx.ndim == 1
    assert hole_idx.tolist() == [-1]
    assert part_idx.tolist() == [-1]


def test_ignore_part_index_does_not_mask():
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])
    part_hole_batch = torch.tensor([0, 0])
    ignore_part_idx = torch.tensor([-1])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        ignore_part_idx=ignore_part_idx,
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


def test_distance_tie_selects_first():
    # for batch of 1, with two holes, test that the first hole is selected if they are at the same distance.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04], [0.0, 0.0, -0.04]]])
    part_hole_batch = torch.tensor([0, 0])

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
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
    assert torch.isclose(
        fastener.get_tip_pos_relative_to_center() * 1000,
        torch.tensor(tuple(bd_geometry.joints["fastener_joint_tip"].location.position)),
    ).all()
    assert (fastener.get_tip_pos_relative_to_center()[:2] == 0).all()
    assert fastener.get_tip_pos_relative_to_center()[2] == -fastener.length / 1000
    # note: this is a kind of pointless test because fastener_joint_tip is not called anywhere, but let it be.
