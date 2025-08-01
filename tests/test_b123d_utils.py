import pytest
import numpy as np
from build123d import *
import build123d as bd

from repairs_components.geometry.b123d_utils import (
    fastener_hole,
    fastener_hole_joint_name,
    fastener_hole_info_from_joint_name,
    connect_fastener_to_joint,
    filtered_intersection_check,
)
from repairs_components.geometry.fasteners import Fastener


class TestFastenerHole:
    """Test the fastener_hole function with various scenarios."""

    def test_fastener_hole_with_depth(self):
        """Test fastener hole with specified depth of 6mm on top of a 10x10x10 box."""
        # Create a BuildPart with a Box(10,10,10)
        with BuildPart() as build_part:
            Box(10, 10, 10)

            # Add fastener hole on top face with depth of 6
            with Locations(build_part.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                _, hole_loc, joint = fastener_hole(
                    radius=2.5, depth=6.0, id=1, build_part=build_part
                )

        # Verify the location
        assert isinstance(hole_loc, Locations)
        hole_position = np.array(tuple(hole_loc.locations[0].position))
        assert np.allclose(hole_position, (0, 0, 5))

        # Verify the joint properties
        assert isinstance(joint, RigidJoint)
        assert joint.label == "fastener_hole_1#6.0#blind"

        # The fastener_loc should be at (0,0,5) relative to the top face
        # Since the box is 10 units tall and centered at origin, top face is at z=5
        # The hole location is relative to that face, so (0,0,0) relative to face = (0,0,5) absolute
        joint_position = np.array(tuple(joint.location.position))
        assert np.allclose(
            joint_position, (0, 0, 5)
        )  # This is relative to the face location

        # Verify it's not marked as through
        assert "blind" in joint.label
        assert "through" not in joint.label

        # Verify depth in joint label is 6
        _, depth, is_through = fastener_hole_info_from_joint_name(joint.label)
        assert depth == 6.0
        assert not is_through

    def test_fastener_hole_through_depth_none(self):
        """Test fastener hole with depth=None (through hole) on a 10x10x10 box."""
        # Create a BuildPart with a Box(10,10,10)
        with BuildPart() as build_part:
            Box(10, 10, 10)

            # Add through fastener hole on top face
            with Locations(build_part.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                _, hole_loc, joint = fastener_hole(
                    radius=2.5, depth=None, id=2, build_part=build_part
                )

        # Verify the location
        assert isinstance(hole_loc, Locations)
        hole_position = np.array(tuple(hole_loc.locations[0].position))
        assert np.allclose(hole_position, (0, 0, 5))

        # Verify the joint properties
        assert isinstance(joint, RigidJoint)

        # Verify it's marked as through
        assert "through" in joint.label
        assert "blind" not in joint.label

        # Verify exported depth in joint label is 10 (depth of the box)
        _, depth, is_through = fastener_hole_info_from_joint_name(joint.label)
        assert depth == 10.0  # Should be the full depth of the box
        assert is_through

    def test_fastener_hole_requires_build_part_for_through_hole(self):
        """Test that fastener_hole raises assertion error when depth=None but build_part=None."""
        with pytest.raises(
            AssertionError, match="build_part must be provided if depth is None"
        ):
            fastener_hole(radius=2.5, depth=None, id=3, build_part=None)

    def test_fastener_hole_positive_depth_assertion(self):
        """Test that fastener_hole raises assertion error for non-positive depth."""
        with pytest.raises(AssertionError, match="Depth must be positive"):
            fastener_hole(radius=2.5, depth=0, id=4)

        with pytest.raises(AssertionError, match="Depth must be positive"):
            fastener_hole(radius=2.5, depth=-1, id=5)


class TestFastenerHoleJointName:
    """Test the fastener_hole_joint_name function."""

    def test_blind_hole_joint_name(self):
        """Test joint name generation for blind holes."""
        name = fastener_hole_joint_name(id=1, connection_depth=6.0, is_through=False)
        assert name == "fastener_hole_1#6.0#blind"

    def test_through_hole_joint_name(self):
        """Test joint name generation for through holes."""
        name = fastener_hole_joint_name(id=2, connection_depth=10.0, is_through=True)
        assert name == "fastener_hole_2#10.0#through"

    def test_joint_name_with_float_depth(self):
        """Test joint name generation with float depth."""
        name = fastener_hole_joint_name(id=3, connection_depth=7.5, is_through=False)
        assert name == "fastener_hole_3#7.5#blind"

    def test_joint_name_depth_assertion(self):
        """Test that joint name generation raises assertion for invalid depth."""
        with pytest.raises(AssertionError, match="Depth must be a positive float"):
            fastener_hole_joint_name(id=1, connection_depth=0, is_through=False)

        with pytest.raises(AssertionError, match="Depth must be a positive float"):
            fastener_hole_joint_name(id=1, connection_depth=-1, is_through=False)

        with pytest.raises(AssertionError, match="Depth must be a positive float"):
            fastener_hole_joint_name(id=1, connection_depth=None, is_through=False)


class TestFastenerHoleInfoFromJointName:
    """Test the fastener_hole_info_from_joint_name function."""

    def test_parse_blind_hole_joint_name(self):
        """Test parsing blind hole joint name."""
        name = "fastener_hole_1#6.0#blind"
        id, depth, is_through = fastener_hole_info_from_joint_name(name)

        assert id == 1
        assert depth == 6.0
        assert not is_through

    def test_parse_through_hole_joint_name(self):
        """Test parsing through hole joint name."""
        name = "fastener_hole_2#10.0#through"
        id, depth, is_through = fastener_hole_info_from_joint_name(name)

        assert id == 2
        assert depth == 10.0
        assert is_through

    def test_parse_float_depth_joint_name(self):
        """Test parsing joint name with float depth."""
        name = "fastener_hole_3#7.5#blind"
        id, depth, is_through = fastener_hole_info_from_joint_name(name)

        assert id == 3
        assert depth == 7.5
        assert not is_through

    def test_parse_invalid_hole_type(self):
        """Test parsing joint name with invalid hole type."""
        name = "fastener_hole_1#6.0#invalid"
        with pytest.raises(AssertionError, match="Invalid hole type: invalid"):
            fastener_hole_info_from_joint_name(name)


class TestFastenerHoleJointNameRoundTrip:
    """Test that fastener_hole_joint_name and fastener_hole_info_from_joint_name are inverses."""

    def test_round_trip_blind_hole(self):
        """Test round trip for blind hole."""
        original_id = 1
        original_depth = 6.0
        original_is_through = False

        # Generate name
        name = fastener_hole_joint_name(
            original_id, original_depth, original_is_through
        )

        # Parse name back
        parsed_id, parsed_depth, parsed_is_through = fastener_hole_info_from_joint_name(
            name
        )

        # Verify they match
        assert parsed_id == original_id
        assert parsed_depth == original_depth
        assert parsed_is_through == original_is_through

    def test_round_trip_through_hole(self):
        """Test round trip for through hole."""
        original_id = 42
        original_depth = 15.75
        original_is_through = True

        # Generate name
        name = fastener_hole_joint_name(
            original_id, original_depth, original_is_through
        )

        # Parse name back
        parsed_id, parsed_depth, parsed_is_through = fastener_hole_info_from_joint_name(
            name
        )

        # Verify they match
        assert parsed_id == original_id
        assert parsed_depth == original_depth
        assert parsed_is_through == original_is_through


class TestFilteredIntersectionCheck:
    def test_has_intersection_flat(self):
        with BuildPart() as build_part:
            Box(10, 10, 10)
        with BuildPart() as build_part2:
            Box(10, 10, 10)
        build_part.part.label = "1@solid"
        build_part2.part.label = "2@solid"

        intersecting_compound = Compound(children=[build_part.part, build_part2.part])

        has_invalid_intersection, parts, intersect_volume = filtered_intersection_check(
            intersecting_compound, assertion=False
        )
        assert has_invalid_intersection
        assert parts == (build_part.part, build_part2.part)
        assert np.isclose(intersect_volume, build_part.part.volume, atol=1e-3)

    def test_no_intersection_flat(self):
        with BuildPart() as build_part:
            Box(10, 10, 10)
        with BuildPart() as build_part2:
            Box(10, 10, 10)

        build_part.part.label = "1@solid"
        build_part2.part.label = "2@solid"
        build_part2.part = build_part.part.located(Pos(50, 50, 50))  # further out.

        intersecting_compound = Compound(children=[build_part.part, build_part2.part])

        has_invalid_intersection, parts, intersect_volume = filtered_intersection_check(
            intersecting_compound, assertion=False
        )
        assert not has_invalid_intersection
        assert parts == (None, None)
        assert np.isclose(intersect_volume, 0, atol=1e-3)

    def test_filter_labels_in_compound(self):
        with BuildPart() as build_part:
            Box(10, 10, 10)

        with BuildPart() as ignored_intersecting_part:
            Box(10, 10, 10)

        with BuildPart() as build_part2:
            Box(10, 10, 10)

        build_part.part.label = "in_part@solid"
        ignored_intersecting_part.part.label = "ignored_part@ignored"
        build_part2.part.label = "other@solid"

        ignored_intersecting_part.part = ignored_intersecting_part.part.located(
            Pos(20, 20, 20)
        )
        build_part2.part = build_part2.part.located(Pos(20, 20, 20))

        compound_with_child = Compound(
            children=[build_part.part, ignored_intersecting_part.part],
            label="child_compound",
        )

        intersecting_compound = Compound(
            children=[compound_with_child, build_part2.part], label="top_compound"
        )

        has_invalid_intersection, parts, intersect_volume = filtered_intersection_check(
            intersecting_compound, ignored_labels=("ignored",), assertion=False
        )
        assert not has_invalid_intersection
        assert parts == (None, None)
        assert np.isclose(intersect_volume, 0, atol=1e-3)
        # note: I expect this to work, but for some reason, compound.do_children_intersect() intersects children with their children. Why, I don't know.
        # the test should pass otherwise.


class TestConnectFastenerToJoint:
    """Test the connect_fastener_to_joint function."""

    def test_connect_fastener_to_joint_basic(self):
        """Test connecting a fastener to a hole joint in a box."""
        # Create a BuildPart with a Box(10,10,10) and add a fastener hole
        with BuildPart() as build_part:
            Box(10, 10, 10)

            # Add fastener hole on top face with depth of 6
            with Locations(build_part.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                hole, hole_loc, hole_joint = fastener_hole(
                    radius=2.5, depth=6, id=1, build_part=build_part
                )
        build_part.part.label = "test_box@solid"  # Required label format

        # Create a fastener with proper label
        fastener = Fastener(length=15.0, diameter=5.0, initial_hole_id_a=1)
        fastener_geom = fastener.bd_geometry()

        # Move the box to a different position to test positioning
        moved_box = build_part.part.locate(Pos(10, 5, 0))  # move it.
        # debug #sanity check - should not edit parts label.
        assert moved_box.label == build_part.part.label
        # /debug

        # Get initial fastener position
        initial_fastener_pos = tuple(fastener_geom.center(CenterOf.BOUNDING_BOX))

        # Connect the fastener to the joint
        connect_fastener_to_joint(fastener_geom, hole_joint, "fastener_joint_a")

        # Verify the fastener has been moved to the hole position
        # The fastener should now be positioned relative to the hole joint location
        final_fastener_pos = tuple(fastener_geom.center(CenterOf.BOUNDING_BOX))

        # The fastener should have moved from its initial position
        assert not np.allclose(initial_fastener_pos, final_fastener_pos), (
            "Fastener position should have changed after connecting to joint"
        )

        # Verify the joint connection was made
        assert "fastener_joint_a" in fastener_geom.joints
        fastener_joint = fastener_geom.joints["fastener_joint_a"]
        assert fastener_joint.connected_to == hole_joint

    def test_connect_fastener_to_joint_assertions(self):
        """Test that connect_fastener_to_joint raises proper assertions."""
        # Create a basic fastener and hole setup
        with BuildPart() as build_part:
            Box(10, 10, 10)

            with Locations(build_part.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                hole, hole_loc, hole_joint = fastener_hole(
                    radius=2.5, depth=6, id=1, build_part=build_part
                )
        build_part.part.label = "test_box@solid"

        fastener = Fastener(length=15.0, diameter=5.0)
        fastener_geom = fastener.bd_geometry()

        # Test assertion for missing fastener label
        fastener_geom.label = None
        with pytest.raises(
            AssertionError,
            match="Connected fastener and joint parent labels are not set",
        ):
            connect_fastener_to_joint(fastener_geom, hole_joint, "fastener_joint_a")

        # Test assertion for wrong fastener label format
        fastener_geom.label = "wrong_label"
        with pytest.raises(
            AssertionError, match="Fastener label does not end with '@fastener'"
        ):
            connect_fastener_to_joint(fastener_geom, hole_joint, "fastener_joint_a")

        # Fix fastener label
        fastener_geom.label = "test_fastener@fastener"

        # Test assertion for wrong joint parent label format
        hole_joint.parent.label = "wrong_label"
        with pytest.raises(
            AssertionError,
            match="Joint parent label does not end with '@solid' or '@fixed_solid'",
        ):
            connect_fastener_to_joint(fastener_geom, hole_joint, "fastener_joint_a")

        # Fix joint parent label
        hole_joint.parent.label = "test_box@solid"

        # Test assertion for missing fastener joint
        with pytest.raises(
            AssertionError,
            match="Fastener joint name nonexistent_joint not found in fastener geometry",
        ):
            connect_fastener_to_joint(fastener_geom, hole_joint, "nonexistent_joint")

    def test_connect_fastener_to_joint_with_fixed_solid(self):
        """Test connecting fastener to joint with @fixed_solid label."""
        # Create a BuildPart with a Box(10,10,10) and add a fastener hole
        with BuildPart() as build_part:
            Box(10, 10, 10)

            # Add fastener hole on top face with depth of 6
            with Locations(build_part.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                hole, hole_loc, hole_joint = fastener_hole(
                    radius=2.5, depth=6, id=1, build_part=build_part
                )
        build_part.part.label = "test_box@fixed_solid"  # Test @fixed_solid label

        # Create a fastener with proper label
        fastener = Fastener(length=15.0, diameter=5.0)
        fastener_geom = fastener.bd_geometry()

        # This should not raise an assertion error with @fixed_solid label
        connect_fastener_to_joint(fastener_geom, hole_joint, "fastener_joint_a")

        # Verify the joint connection was made
        assert "fastener_joint_a" in fastener_geom.joints
        fastener_joint = fastener_geom.joints["fastener_joint_a"]
        assert fastener_joint.connected_to == hole_joint


if __name__ == "__main__":
    pytest.main([__file__])
