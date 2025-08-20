"""Tests for the assembly task functionality."""

import numpy as np
from build123d import Box, Sphere, Cylinder, Compound, Pos
from repairs_components.geometry.b123d_utils import filtered_intersection_check
from repairs_components.processing.tasks import AssembleTask
import pytest


@pytest.fixture
def bd_test_compound():
    """Create a test compound with various shapes."""
    shapes = [
        Box(10, 20, 5),  # Flat box
        Sphere(7.5).move(Pos(30, 0, 0)),  # Sphere
        Cylinder(5, 15).move(Pos(0, 30, 0)),  # Tall cylinder
        Box(15, 10, 8).move(Pos(30, 30, 0)),  # Medium box
        Box(25, 10, 3).move(Pos(15, 15, 0)),  # Wide flat box
    ]
    return Compound(children=shapes)


def test_perturb_initial_state(bd_test_compound):
    """Test that parts are properly disassembled within bounds."""
    task = AssembleTask()
    env_size = (640, 640, 640)  # 50cm x 50cm x 50cm environment

    for i in range(10):  # 10 test cases to catch errors.
        # Run the disassembly
        new_compound = task.perturb_initial_state(bd_test_compound, env_size)

        assert (
            np.array(tuple(new_compound.bounding_box().size)) < np.array(env_size)
        ).all(), "Parts are not within bounds."
        filtered_intersection_check(new_compound, assertion=True)


def test_stable_orientation():
    """Test that elongated parts are oriented stably."""
    task = AssembleTask()

    # Test with an elongated box (should be oriented vertically)
    aabb_size = np.array([10, 10, 50])  # Tall box
    (_, _, up_axis) = task._get_stable_orientation(aabb_size)
    assert up_axis == 2, "Tall box should be Z-up"

    # Test with a wide box (should be oriented with X up)
    aabb_size = np.array([50, 10, 10])
    (_, _, up_axis) = task._get_stable_orientation(aabb_size)
    assert up_axis == 0, "Wide box should be X-up"

    # Test with a cube (should default to Z-up)
    aabb_size = np.array([10, 10, 10])
    (_, _, up_axis) = task._get_stable_orientation(aabb_size)
    assert up_axis == 2, "Cube should default to Z-up"


def debug_perturb_and_vis():
    from ocp_vscode import show
    import build123d as bd

    task = AssembleTask()
    bd_test_compound = bd_test_compound()
    env_size = (640, 640, 640)  # 50cm x 50cm x 50cm environment

    # Run the disassembly
    new_compound = task.perturb_initial_state(bd_test_compound, env_size)
    with bd.BuildPart() as vis_bounding_box:
        with bd.Locations((0, 0, env_size[2] / 2)):
            bd.Box(*env_size)
    vis_bounding_box.part.color = bd.Color(0.2, 0.2, 0.2, 0.2)
    show(new_compound, vis_bounding_box.part)


def test_perturb_desired_state():
    """Test that perturb_desired_state moves the part to the desired location and sets Z-min to 0."""
    task = AssembleTask()

    # Create a box at (10, 10, 10) with size (5, 5, 5)
    box = Box(5, 5, 5).move(Pos(10, 10, 10))

    # Store the original position for verification
    original_bbox = box.bounding_box()
    original_min = np.array(tuple(original_bbox.min))

    # Perturb the desired state
    env_size = (640, 640, 640)  # Large enough environment
    result = task.perturb_desired_state(box, env_size)

    # Verify the result
    result_bbox = result.bounding_box()
    result_min = np.array(tuple(result_bbox.min))
    result_max = np.array(tuple(result_bbox.max))

    # The box should be moved in XY and have Z-min at 0
    assert np.allclose(result_min[2], 0), f"Z-min should be 0, got {result_min[2]}"

    # Dimensions under Z-yaw:
    # Z size is preserved; XY AABB grows within [orig, orig*sqrt(2)].
    size = result_max - result_min
    assert np.allclose(size[2], 5), f"Z size should be 5, got {size[2]}"
    max_xy = 5 * np.sqrt(2) + 1e-6
    min_xy = 5 - 1e-6
    assert min_xy <= size[0] <= max_xy, f"X size out of expected range: {size[0]}"
    assert min_xy <= size[1] <= max_xy, f"Y size out of expected range: {size[1]}"

    # The XY position should be different from original
    assert not np.allclose(result_min[:2], original_min[:2]), (
        "XY position should be different after perturbation"
    )


@pytest.fixture
def bd_test_compound_labeled():
    """Two labeled parts for clustered path tests."""
    a = Box(10, 10, 5)
    a.label = "a@solid"
    b = Box(10, 10, 5).move(Pos(50, 0, 0))
    b.label = "b@solid"
    return Compound(children=[a, b])


def _center_np(part: Compound) -> np.ndarray:
    """Compute center from min/max to avoid API differences for bbox.center."""
    bbox = part.bounding_box()
    vmin = np.array(tuple(bbox.min), dtype=float)
    vmax = np.array(tuple(bbox.max), dtype=float)
    return 0.5 * (vmin + vmax)


def test_cluster_cohesion_and_grounding(
    bd_test_compound_labeled,
):
    np.random.seed(0)
    task = AssembleTask()
    comp = bd_test_compound_labeled
    # Precompute original relative vector between labeled parts
    a0, b0 = comp.children[0], comp.children[1]
    d0 = _center_np(a0) - _center_np(b0)
    assert np.linalg.norm(d0[:2]) > 1e-6, "Initial XY separation must be non-zero"

    linked = {"mech_linked": (["a@solid", "b@solid"],)}
    env_size = (640, 640, 640)
    res = task.perturb_initial_state(comp, env_size, linked_groups=linked)

    # Collect by label from result
    parts = {p.label: p for p in res.children}
    assert set(parts.keys()) == {"a@solid", "b@solid"}
    d1 = _center_np(parts["a@solid"]) - _center_np(parts["b@solid"])

    # Z-delta preserved (cluster yaw about Z and translations only)
    assert np.isclose(d0[2], d1[2], atol=1e-6)
    # Cohesion: parts remain relatively close as a cluster (well within env span)
    assert np.linalg.norm(d1[:2]) < 0.4 * min(640, 640)
    # Each child must be grounded
    for child in res.children:
        assert np.isclose(child.bounding_box().min.Z, 0), "Child must be Z-grounded"


def test_cluster_invalid_name_raises(bd_test_compound_labeled):
    task = AssembleTask()
    comp = bd_test_compound_labeled
    env_size = (640, 640, 640)
    # Reference a non-existent label
    bad = {"mech_linked": (["a@solid", "c@solid"],)}
    with pytest.raises(AssertionError):
        task.perturb_initial_state(comp, env_size, linked_groups=bad)


@pytest.mark.parametrize(
    "linked",
    [None, {"mech_linked": (["a@solid", "b@solid"],)}],
)
def test_each_child_grounded_and_within_bounds(bd_test_compound_labeled, linked):
    np.random.seed(1)
    task = AssembleTask()
    env_size = (640, 640, 640)
    res = task.perturb_initial_state(
        bd_test_compound_labeled, env_size, linked_groups=linked
    )
    # Bounds and non-intersection
    assert (np.array(tuple(res.bounding_box().size)) < np.array(env_size)).all(), (
        "Compound AABB must be within env bounds"
    )
    filtered_intersection_check(res, assertion=True)
    # Each child grounded
    for child in res.children:
        assert np.isclose(child.bounding_box().min.Z, 0), "Child must be Z-grounded"


if __name__ == "__main__":
    debug_perturb_and_vis()

# TODO: assert that parts don't intersect in AssembleTask.perturb_initial_state()
