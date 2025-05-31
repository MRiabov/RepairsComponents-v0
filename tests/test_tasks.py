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
    env_size = (500, 500, 500)  # 50cm x 50cm x 50cm environment

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
    env_size = (500, 500, 500)  # 50cm x 50cm x 50cm environment

    # Run the disassembly
    new_compound = task.perturb_initial_state(compound, env_size)
    with bd.BuildPart() as vis_bounding_box:
        with bd.Locations((0, 0, env_size[2] / 2)):
            bd.Box(*env_size)
    vis_bounding_box.part.color = bd.Color(0.2, 0.2, 0.2, 0.2)
    show(new_compound, vis_bounding_box.part)


if __name__ == "__main__":
    debug_perturb_and_vis()
