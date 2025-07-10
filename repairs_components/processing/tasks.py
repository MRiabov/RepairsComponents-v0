from abc import ABC, abstractmethod
from build123d import CenterOf, Compound, Location, Part, Pos, Axis
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class Task(ABC):
    "A task class that pertubs the geometry of a compound to create more diverse tasks."

    @abstractmethod
    def perturb_desired_state(
        self,
        compound: Compound,
        env_size=(640, 640, 640),  # mm
    ) -> Compound:
        """Perturb the desired state of the task. Only move in x and y directions, the minimum of Z axis should be kept at 0."""
        # Get the bounding box of the original compound
        bbox = compound.bounding_box()
        aabb_min = np.array(tuple(bbox.min))
        aabb_max = np.array(tuple(bbox.max))
        size = aabb_max - aabb_min
        assert np.all(size < np.array(env_size)), (
            f"Compound is too large for the environment. Size: {size}"
        )

        # Calculate the required margin to keep 5 units from each side
        margin = 50

        # Calculate min and max positions in XY plane
        min_xy = np.array([margin, margin])
        max_xy = np.array(
            [env_size[0] - size[0] - margin, env_size[1] - size[1] - margin]
        )

        # Ensure the part fits in the environment

        # Generate random position within the valid range
        target_xy = np.random.uniform(min_xy, max_xy)

        # Calculate the offset needed to move the compound
        # We want the new position to be at target_xy in XY plane
        # and have the bottom of the bounding box at Z=0
        offset = np.array(
            [
                target_xy[0] - aabb_min[0],  # Move to target X
                target_xy[1] - aabb_min[1],  # Move to target Y
                -aabb_min[2],  # Move up to make Z-min = 0
            ]
        )

        # print(f"Original bbox: min={aabb_min}, max={aabb_max}")
        # print(f"Target position: {target_xy}")
        # print(f"Moving by offset: {offset}")

        # Move the entire compound as a single unit
        result = compound.located(Location(offset))

        # # Debug: Check the resulting position
        # result_bbox = result.bounding_box()
        # print(f"Resulting bbox: min={result_bbox.min}, max={result_bbox.max}")

        # Verify the result meets our requirements
        result_bbox = result.bounding_box()
        result_min = np.array(tuple(result_bbox.min))
        result_max = np.array(tuple(result_bbox.max))

        # Check Z-min is at 0 (within floating point tolerance)
        assert np.allclose(result_min[2], 0, atol=1e-6), (
            f"Z-min should be 0, got {result_min[2]}"
        )
        # Check dimensions are preserved (within floating point tolerance)
        assert np.allclose(result_max - result_min, size, atol=1e-6), (
            f"Dimensions should be preserved as {size}, got {result_max - result_min}"
        )
        assert np.all(result_min >= 0), (
            f"All dimensions should be non-negative, got {result_min}"
        )
        assert np.all(result_max <= env_size), (
            f"All dimensions should be less than env_size, got {result_max}"
        )

        return result

    @abstractmethod
    def perturb_initial_state(
        self, compound: Compound, env_size: tuple[float, float, float]
    ) -> Compound:
        "Perturb initial state; return the compound and the new positions of bodies in the compound."
        raise NotImplementedError

    def unchanged_initial_state(
        self, compound: Compound, env_size: tuple[float, float, float]
    ) -> Compound:
        "Return the compound unchanged. This is useful to avoid recompilation times on genesis (so environments make a cache hit.)"
        return compound


class AssembleTask(Task):
    """A task class that pertubs the geometry of a compound to create more diverse tasks.

    This class in particular disassembles any Compound to separate parts and positions them on the flat surface.
    It also ensures stable orientation of prolonged parts, e.g. in a table, legs would lie flat on the ground."""

    def perturb_desired_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        return super().perturb_desired_state(compound, env_size)

    def _get_stable_orientation(self, aabb_size):
        """Determine the most stable orientation for a part based on its AABB dimensions.

        This method ensures that elongated parts are oriented with their longest dimension
        vertically for better stability during the disassembled state.

        Args:
            aabb_size: The size of the part's axis-aligned bounding box (x, y, z)

        Returns:
            tuple: (rotation_axis, angle), up_axis where:
                - rotation_axis: 3D vector (x, y, z) for the rotation axis
                - angle: Rotation angle in radians
                - up_axis: Indicates which axis is vertical (0=x, 1=y, 2=z)
        """
        dims = np.array(aabb_size)
        aspect_ratios = dims / np.mean(dims)

        if np.any(aspect_ratios > 1.3):
            up_axis = np.argmax(dims)
            if up_axis == 0:  # X is up
                return (0, 0, 1), np.pi / 2, up_axis  # Rotate 90° around Z to make X up
            elif up_axis == 1:  # Y is up
                return (1, 0, 0), np.pi / 2, up_axis  # Rotate 90° around X to make Y up
        return (0, 0, 1), 0.0, 2  # Default to Z-up, no rotation needed

    def _get_random_position(
        self,
        part_info,
        used_rectangles,
        bin_max_width,
        bin_max_height,
        safety_scale=0.9,
    ):
        """Find a random non-overlapping position for a part within the bin.

        Args:
            part_info: Dictionary containing part dimensions and other info
            used_rectangles: List of already placed rectangles (x1, y1, x2, y2)
            bin_max_width: Width of the packing area
            bin_max_height: Height of the packing area

        Returns:
            tuple: (x, y) position if found, None if no valid position found
        """
        w, h = part_info["proj_width"], part_info["proj_height"]
        max_attempts = 100

        # max width by safety scale
        bin_max_width *= safety_scale
        bin_max_height *= safety_scale
        # min width - 0 to 1-safety scale
        bin_min_width = bin_max_width * (1 - safety_scale)
        bin_min_height = bin_max_height * (1 - safety_scale)

        for _ in range(max_attempts):
            # Try random positions
            x = np.random.uniform(bin_min_width + w / 2, bin_max_width - w / 2)
            y = np.random.uniform(bin_min_height + h / 2, bin_max_height - h / 2)

            new_rect = (x, y, x + w, y + h)

            # Check for overlaps with existing rectangles
            overlap = False
            for placed in used_rectangles:
                if not (
                    new_rect[2] <= placed[0]  # New rect is to the left
                    or new_rect[0] >= placed[2]  # New rect is to the right
                    or new_rect[3] <= placed[1]  # New rect is above
                    or new_rect[1] >= placed[3]
                ):  # New rect is below
                    overlap = True
                    break

            if not overlap:
                return x, y

        return None  # No valid position found after max_attempts

    # NOTE!!! build123d provides a method to pack parts in 2D space without overlaps.
    # pack.pack
    # although it's not random.
    def _pack_2d(self, part_infos, bin_width, bin_height):
        """Randomly pack parts in 2D space without overlaps.

        This method tries to place each part at a random position within the bin
        while ensuring no overlaps with already placed parts. Parts are processed
        in random order to increase variation in the resulting layouts.

        Args:
            part_infos: List of part information dictionaries
            bin_width: Width of the packing area
            bin_height: Height of the packing area

        Returns:
            list: List of tuples (x, y, part_info) for placed parts
        """
        placed = []
        used_rectangles = []

        # Shuffle parts to get different layouts each time
        part_infos = part_infos.copy()
        np.random.shuffle(part_infos)

        for part_info in part_infos:
            pos = self._get_random_position(
                part_info, used_rectangles, bin_width, bin_height
            )
            if pos is not None:
                x, y = pos
                w, h = part_info["proj_width"], part_info["proj_height"]
                placed.append((x, y, part_info))
                used_rectangles.append((x, y, x + w, y + h))

        return placed

    def perturb_initial_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        """Randomly disassemble the compound into individual parts with non-overlapping positions.

        This method takes a compound object representing an assembled model and disassembles it
        by placing each part in a random but valid position within the environment. Parts are
        oriented for stability and positioned to avoid overlaps, creating a realistic disassembled
        state for robotic manipulation training.

        Args:
            compound: The assembled compound to disassemble
            initial_state: The current simulation state to modify
            env_size: Size of the environment (x, y, z) in millimeters

        Returns:
            RepairsSimState: The modified simulation state with disassembled parts
        """

        # NOTE: 100% definitely, this was from 0 to 640. I guess I shifted it to -320 to 320 somewhere later.
        # and it's ok to have it there.
        # for now: revert everything back to (0, 0, 0)-(640, 640, 640)
        # however I suspect this does not account for rotations. So when calculating for part position, it does not account for *new* part rotation
        # additionally, this does not respect fixed parts. Those should never be touched.

        env_size = np.array(env_size)
        # Safety margin: keep outer 10 % of XY space empty
        safety_scale = 0.9
        safe_env_size = env_size * safety_scale
        safe_env_min = np.array([-safe_env_size[0] / 2, -safe_env_size[1] / 2, 0])
        safe_env_max = np.array(
            [safe_env_size[0] / 2, safe_env_size[1] / 2, env_size[2]]
        )
        components = compound.children
        # note: components because both compounds and children.
        part_info = []

        preprocessed_components = []

        # remove all joints between all parts and recenter (preprocess)
        for component in components:
            for joint in component.joints.values():
                joint.connected_to = None
            # recenter all objects at their AABB center:
            # get current center
            center = np.array(tuple(component.center(CenterOf.BOUNDING_BOX)))
            component.moved(Pos(-center))  # move by negative center
            preprocessed_components.append(component)

        # Calculate AABBs and determine stable orientations for each part
        for component in preprocessed_components:
            # Get stable orientation (longest dimension up if significantly elongated)
            (rotation_axis, angle, up_axis) = self._get_stable_orientation(
                np.array(tuple(component.bounding_box().size))
            )

            # Apply transformation to part
            if abs(angle) > 1e-6:  # Only rotate if angle is not zero
                # Create an axis of rotation at the part's center
                rotation_axis_obj = Axis(
                    origin=component.center(), direction=rotation_axis
                )
                component = component.rotate(
                    rotation_axis_obj, angle * 180 / np.pi
                )  # Convert to degrees

            # FIXME: it aligns all parts in X axis at the moment.

            # put updated projection dimensions into part_info
            aabb = component.bounding_box()
            proj_width = aabb.size.X
            proj_height = aabb.size.Y
            aabb_min = np.array(tuple(aabb.min))
            aabb_size = np.array(tuple(aabb.max))

            part_info.append(
                {
                    "part": component,
                    "aabb_min": aabb_min,
                    "aabb_size": aabb_size,
                    "rotation_axis": rotation_axis,
                    "up_axis": up_axis,
                    "proj_width": proj_width,
                    "proj_height": proj_height,
                }
            )

        # Pack parts with random positions
        packed = self._pack_2d(part_info, safe_env_size[0], safe_env_size[1])

        # FIXME: pack2d makes overlapping shapes (note: this is written in plan for 2 days.md. See a list of bugs there.)

        assert packed, ValueError("Failed to find valid positions for all parts")

        new_parts = []
        # Position parts in the environment
        for x, y, info in packed:
            component = info["part"]

            # this is assuming the part is centered at the origin, which it should be. although not necessarily...
            pos = Pos(x, y, component.bounding_box().size.Z / 2)
            new_parts.append(component.located(pos))

        # Rebuild compound with new part positions
        new_compound = Compound(children=new_parts)
        assert np.isclose(new_compound.bounding_box().min.Z, 0), (
            f"New compound has non-zero minimum Z: {new_compound.bounding_box().min.Z}"
        )
        assert (
            np.array(new_compound.bounding_box().min.to_tuple()) >= safe_env_min
        ).all(), (
            f"New compound has minimum below safe environment minimum. AABB: {new_compound.bounding_box()}"
        )
        assert (
            np.array(new_compound.bounding_box().max.to_tuple()) <= env_size
        ).all(), (
            f"New compound has maximum above environment maximum. AABB: {new_compound.bounding_box()}"
        )

        return new_compound


# NOTE!: to apply absolute position to a part, use located. to apply relative position, use moved.


# Inverse of AssembleTask: disassemble for desired state, assemble for initial state
class DisassembleTask(Task):
    """Inverse of AssembleTask: initial state is the assembled desired state, and the desired state is disassembled desired state."""

    def perturb_initial_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        # Use AssembleTask's perturb_desired_state to create the assembled initial state
        return AssembleTask().perturb_desired_state(compound, env_size)

    def perturb_desired_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        # Use AssembleTask's perturb_initial_state to create the disassembled desired state
        return AssembleTask().perturb_initial_state(compound, env_size)


class ReplaceTask(Task):
    """A task type that marks a single part as "to be replaced" and puts the right copy next to the location."""

    # NOTE: this code was left unfinished.
    def perturb_initial_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        compound = super().perturb_desired_state(compound, env_size)
        parts = [part for part in compound.children if isinstance(part, Part)]
        random_part: Part = np.random.choice(parts)
        new_part = random_part.located(
            Location((10, 10, 10))
        )  # TODO: any position on the ground that is not occupied by another part.
        random_part.label = "part_to_replace"
        new_part.parent = compound
        return compound

    def perturb_desired_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        # Use AssembleTask's perturb_desired_state to create the disassembled desired state
        return AssembleTask().perturb_desired_state(
            compound, env_size
        )  # simply unchanged.


class InsertTask(Task):
    """A task type that marks a single part as "to be inserted", and detaches it from the compound."""

    def perturb_initial_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        compound = super().perturb_desired_state(compound, env_size)
        parts = [part for part in compound.children if isinstance(part, Part)]
        random_part: Part = np.random.choice(parts)
        random_part.joints = {}  # I hope this detaches it. to be tested.
        return compound

    def perturb_desired_state(
        self, compound: Compound, env_size=(640, 640, 640)
    ) -> Compound:
        # Use AssembleTask's perturb_desired_state to create the disassembled desired state
        return AssembleTask().perturb_desired_state(
            compound, env_size
        )  # simply unchanged.


class TaskTypes(IntEnum):
    "Enum necessary for persistence"

    # TODO: can be used at some point to generate more of certain type of tasks. Now it's equally random.
    # it would probably be done by user since it's a single torch.random.(...) function.

    ASSEMBLE = 0
    DISASSEMBLE = 1
    REPLACE = 2
    INSERT = 3
