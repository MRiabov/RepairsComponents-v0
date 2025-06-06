"""Tests for the assembly task functionality."""

import numpy as np
from build123d import Box, Sphere, Cylinder, Compound, Pos
from repairs_components.processing.tasks import AssembleTask


def create_test_compound():
    """Create a test compound with various shapes."""
    shapes = [
        Box(10, 20, 5),  # Flat box
        Sphere(7.5).move(Pos(30, 0, 0)),  # Sphere
        Cylinder(5, 15).move(Pos(0, 30, 0)),  # Tall cylinder
        Box(15, 10, 8).move(Pos(30, 30, 0)),  # Medium box
        Box(25, 10, 3).move(Pos(15, 15, 0)),  # Wide flat box
    ]
    return Compound(children=shapes)


def test_perturb_initial_state():
    """Test that parts are properly disassembled within bounds."""
    task = AssembleTask()
    compound = create_test_compound()
    env_size = (640, 640, 640)  # 50cm x 50cm x 50cm environment

    # Run the disassembly
    new_compound = task.perturb_initial_state(compound, env_size)

    assert (
        np.array(new_compound.bounding_box().size.to_tuple()) < np.array(env_size)
    ).all(), "Parts are not within bounds."


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
    compound = create_test_compound()
    env_size = (64, 64, 64)  # 50cm x 50cm x 50cm environment

    # Run the disassembly
    new_compound = task.perturb_initial_state(compound, env_size)
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
    original_min = np.array(original_bbox.min.to_tuple())

    # Perturb the desired state
    env_size = (64, 64, 64)  # Large enough environment
    result = task.perturb_desired_state(box, env_size)

    # Verify the result
    result_bbox = result.bounding_box()
    result_min = np.array(result_bbox.min.to_tuple())
    result_max = np.array(result_bbox.max.to_tuple())

    # The box should be moved in XY and have Z-min at 0
    assert np.allclose(result_min[2], 0), f"Z-min should be 0, got {result_min[2]}"

    # The dimensions should be preserved
    assert np.allclose(result_max - result_min, [5, 5, 5]), (
        f"Box dimensions should be preserved as (5,5,5), got {result_max - result_min}"
    )

    # The XY position should be different from original
    assert not np.allclose(result_min[:2], original_min[:2]), (
        "XY position should be different after perturbation"
    )


if __name__ == "__main__":
    debug_perturb_and_vis()
