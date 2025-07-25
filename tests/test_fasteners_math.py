import torch
import pytest
from repairs_components.geometry.fasteners import (
    Fastener,
    check_fastener_possible_insertion,
    recalculate_fastener_pos_with_offset_to_hole,
)


# -----------------------------------------------
# === check_fastener_possible_insertion tests ===
# -----------------------------------------------
def test_simple_match_within_distance():
    # for batch of 1, with two holes, both of which in the first part, test that one close hole is selected.
    tip_pos = torch.tensor([[0.0, 0.0, 0.0]])  # [B,3]
    part_hole_positions = torch.tensor([[[0.0, 0.0, 0.04], [0.2, 0.0, 0.0]]])  # [B,H,3]
    part_hole_batch = torch.tensor([0, 0])  # [H]

    part_idx, hole_idx = check_fastener_possible_insertion(
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=torch.full((1,), 30),
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
        tip_pos,
        part_hole_positions,
        part_hole_batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=torch.full((1,), 30),
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
        tip_pos,
        part_hole_positions,
        hole_batch,
        connection_dist_threshold=0.05,
        connection_angle_threshold=torch.full((1,), 30),
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
        connection_angle_threshold=torch.full((1,), 30),
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
        connection_angle_threshold=torch.full((1,), 30),
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
        connection_angle_threshold=torch.full((1,), 30),
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
        connection_angle_threshold=torch.full((1,), 30),
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
        connection_angle_threshold=torch.full((1,), 30),
    )
    assert hole_idx.tolist() == [0]
    assert part_idx.tolist() == [0]


# === bd_geometry tests ===
@pytest.fixture
def fastener():
    return Fastener(
        initial_hole_id_a=0,
        initial_hole_id_b=1,
        length=15.0,
        diameter=5.0,
        b_depth=5.0,
        head_diameter=7.5,
        head_height=3.0,
        thread_pitch=0.5,
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


# ----------------------------------------------------------
# === recalculate_fastener_pos_with_offset_to_hole tests ===
# ----------------------------------------------------------


def test_recalculate_fastener_pos_with_offset_to_hole_through_hole_returns_equal_pos():
    hole_pos = torch.tensor([[0.0, 0.0, 0.0]])
    hole_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_depth = torch.tensor([5.0])
    hole_is_through = torch.tensor([True])
    fastener_length = torch.tensor([15.0])
    top_hole_depth = torch.tensor([0.0])
    fastener_pos = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    assert torch.isclose(fastener_pos, hole_pos).all(), (
        "fastener_pos (base joint pos) should be equal to hole_pos"
    )


def test_recalculate_fastener_pos_with_offset_to_hole_blind_hole_returns_offset_pos():
    hole_pos = torch.tensor([[0.0, 0.0, 0.0]])
    hole_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_depth = torch.tensor([5.0])
    hole_is_through = torch.tensor([False])
    fastener_length = torch.tensor([15.0])
    top_hole_depth = torch.tensor([0.0])
    fastener_pos = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    # With identity quaternion, get_connector_pos applies no rotation, just addition
    # So [0, 0, 10] becomes [0, 0, 10] and is added to hole_pos
    expected_offset = (
        fastener_length[0] - hole_depth[0]
    )  # get_connector_pos no longer negates
    assert torch.isclose(
        fastener_pos,
        hole_pos + torch.tensor([0.0, 0.0, expected_offset]),
    ).all(), "fastener_pos should be hole_pos + (fastener_length - hole_depth)"


def test_recalculate_fastener_pos_with_offset_to_hole_partial_insertion_returns_offset_pos():
    hole_pos = torch.tensor([[0.0, 0.0, 0.0]])
    hole_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hole_depth = torch.tensor([5.0])
    fastener_length = torch.tensor([15.0])
    top_hole_depth = torch.tensor([10.0])
    # blind hole
    hole_is_through = torch.tensor([False])
    fastener_pos_blind = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    # With identity quaternion, get_connector_pos applies no rotation, just addition
    expected_offset = top_hole_depth[0]  # get_connector_pos no longer negates
    assert torch.isclose(
        fastener_pos_blind, hole_pos + torch.tensor([0.0, 0.0, expected_offset])
    ).all(), "fastener_pos should be hole_pos + top_hole_depth"
    # through hole (should be equal.)
    hole_is_through = torch.tensor([True])
    fastener_pos_through = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    # Through hole with partial insertion should also use get_connector_pos
    expected_offset = top_hole_depth[0]  # get_connector_pos no longer negates
    assert torch.isclose(
        fastener_pos_through,
        hole_pos + torch.tensor([0.0, 0.0, expected_offset]),
    ).all(), "fastener_pos should be hole_pos + top_hole_depth (same as blind case)"


def test_recalculate_fastener_pos_with_offset_to_hole_with_quaternion():
    """Test all three cases with non-identity quaternion to ensure get_connector_pos is applied correctly."""
    # Use a 90-degree rotation around Y axis: [cos(45°), 0, sin(45°), 0]
    hole_quat = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])  # 90° rotation around Y
    hole_pos = torch.tensor([[1.0, 2.0, 3.0]])
    hole_depth = torch.tensor([5.0])
    fastener_length = torch.tensor([15.0])

    # Case 1: Through hole without partial insertion - should return hole_pos (no offset)
    hole_is_through = torch.tensor([True])
    top_hole_depth = torch.tensor([0.0])
    fastener_pos_through = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    assert torch.isclose(fastener_pos_through, hole_pos).all(), (
        "Through hole without partial insertion should return hole_pos exactly"
    )

    # Case 2: Blind hole without partial insertion - should apply quaternion transformation
    hole_is_through = torch.tensor([False])
    top_hole_depth = torch.tensor([0.0])
    fastener_pos_blind = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    # Expected: get_connector_pos(hole_pos, hole_quat, [0, 0, fastener_length - hole_depth])
    # get_connector_pos no longer negates input: [0, 0, 10] stays [0, 0, 10]
    # Then applies 90° Y rotation: [0, 0, 10] -> [10, 0, 0] (Z becomes X)
    # Then adds to hole_pos: [1, 2, 3] + [10, 0, 0] = [11, 2, 3]
    expected_blind = torch.tensor([[11.0, 2.0, 3.0]])
    assert torch.isclose(fastener_pos_blind, expected_blind, atol=1e-3).all(), (
        f"Blind hole should apply quaternion transformation. Got {fastener_pos_blind}, expected {expected_blind}"
    )

    # Case 3: Partial insertion (through hole) - should apply quaternion transformation
    hole_is_through = torch.tensor([True])
    top_hole_depth = torch.tensor([7.0])
    fastener_pos_partial = recalculate_fastener_pos_with_offset_to_hole(
        hole_pos,
        hole_quat,
        hole_depth,
        hole_is_through,
        fastener_length,
        top_hole_depth,
    )
    # Expected: get_connector_pos(hole_pos, hole_quat, [0, 0, top_hole_depth])
    # get_connector_pos no longer negates input: [0, 0, 7] stays [0, 0, 7]
    # Then applies 90° Y rotation: [0, 0, 7] -> [7, 0, 0] (Z becomes X)
    # Then adds to hole_pos: [1, 2, 3] + [7, 0, 0] = [8, 2, 3]
    expected_partial = torch.tensor([[8.0, 2.0, 3.0]])
    assert torch.isclose(fastener_pos_partial, expected_partial, atol=1e-3).all(), (
        f"Partial insertion should apply quaternion transformation. Got {fastener_pos_partial}, expected {expected_partial}"
    )
