from abc import ABC, abstractmethod
from build123d import Compound, Location, Part, Pos
import numpy as np


class Task(ABC):
    "A task class that pertubs the geometry of a compound to create more diverse tasks."

    @abstractmethod
    def perturb_desired_state(
        self,
        compound: Compound,
        env_size=(64, 64, 64),  # cm
    ) -> Compound:
        """Perturb the desired state of the task. Only move in x and y directions, the minimum of Z axis should be kept at 0."""
        # Get the bounding box of the original compound
        bbox = compound.bounding_box()
        aabb_min = np.array(bbox.min.to_tuple())
        aabb_max = np.array(bbox.max.to_tuple())
        size = aabb_max - aabb_min

        # Calculate the required margin to keep 5 units from each side
        margin = 5

        # Calculate min and max positions in XY plane
        min_xy = np.array([margin, margin])
        max_xy = np.array(
            [env_size[0] - size[0] - margin, env_size[1] - size[1] - margin]
        )

        # Ensure the part fits in the environment
        assert not np.any(max_xy < min_xy), ValueError(
            "Compound is too large for the environment."
        )

        # Generate random position within the valid range
        target_xy = np.random.uniform(min_xy, max_xy)

        # Calculate the offset needed to move the compound
        # We want the new position to be at target_xy in XY plane
        # and have the bottom of the bounding box at Z=0
        offset = np.array([
            target_xy[0] - aabb_min[0],  # Move to target X
            target_xy[1] - aabb_min[1],  # Move to target Y
            -aabb_min[2]                 # Move up to make Z-min = 0
        ])

        # print(f"Original bbox: min={aabb_min}, max={aabb_max}")
        # print(f"Target position: {target_xy}")
        # print(f"Moving by offset: {offset}")
        
        # Move the entire compound as a single unit
        result = compound.moved(Location(offset))

        # # Debug: Check the resulting position
        # result_bbox = result.bounding_box()
        # print(f"Resulting bbox: min={result_bbox.min}, max={result_bbox.max}")

        # Verify the result meets our requirements
        result_bbox = result.bounding_box()
        result_min = np.array(result_bbox.min.to_tuple())
        result_max = np.array(result_bbox.max.to_tuple())

        # Check Z-min is at 0 (within floating point tolerance)
        assert np.allclose(result_min[2], 0, atol=1e-6), (
            f"Z-min should be 0, got {result_min[2]}"
        )
        # Check dimensions are preserved (within floating point tolerance)
        assert np.allclose(result_max - result_min, size, atol=1e-6), (
            f"Dimensions should be preserved as {size}, got {result_max - result_min}"
        )

        return result

    @abstractmethod
    def perturb_initial_state(
        self, compound: Compound, env_size: tuple[float, float, float]
    ) -> Compound:
        "Perturb initial state; return the compound and the new positions of bodies in the compound."
        raise NotImplementedError


class AssembleTask(Task):
    """A task class that pertubs the geometry of a compound to create more diverse tasks.

    This class in particular disassembles any Compound to separate parts and positions them on the flat surface.
    It also ensures stable orientation of prolonged parts, e.g. in a table, legs would lie flat on the ground."""

    def perturb_desired_state(
        self, compound: Compound, env_size=(64, 64, 64)
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

    def _get_random_position(self, part_info, used_rectangles, bin_width, bin_height):
        """Find a random non-overlapping position for a part within the bin.

        Args:
            part_info: Dictionary containing part dimensions and other info
            used_rectangles: List of already placed rectangles (x1, y1, x2, y2)
            bin_width: Width of the packing area
            bin_height: Height of the packing area

        Returns:
            tuple: (x, y) position if found, None if no valid position found
        """
        w, h = part_info["proj_width"], part_info["proj_height"]
        max_attempts = 100

        for _ in range(max_attempts):
            # Try random positions
            x = np.random.uniform(0, bin_width - w)
            y = np.random.uniform(0, bin_height - h)

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
            list: List of tuples (x, y, part_info, w, h) for placed parts
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
                placed.append((x, y, part_info, w, h))
                used_rectangles.append((x, y, x + w, y + h))

        return placed

    def perturb_initial_state(
        self, compound: Compound, env_size=(64, 64, 64)
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
        # parts = compound.solids()
        parts = list(
            filter(lambda x: isinstance(x, Part), compound.descendants)
        )  # should filter only for solids, connectors, buttons, etc later.
        part_info = []

        # Calculate AABBs and determine stable orientations for each part
        for part in parts:
            bbox = part.bounding_box()
            aabb_min = np.array(bbox.min.to_tuple())
            aabb_max = np.array(bbox.max.to_tuple())
            aabb_size = aabb_max - aabb_min

            # Get stable orientation (longest dimension up if significantly elongated)
            (rotation_axis, angle, up_axis) = self._get_stable_orientation(aabb_size)

            # Project AABB onto ground plane based on up axis
            if up_axis == 0:  # X is up
                proj_width, proj_height = aabb_size[1], aabb_size[2]
            elif up_axis == 1:  # Y is up
                proj_width, proj_height = aabb_size[0], aabb_size[2]
            else:  # Z is up (default)
                proj_width, proj_height = aabb_size[0], aabb_size[1]

            part_info.append(
                {
                    "part": part,
                    "aabb_min": aabb_min,
                    "aabb_size": aabb_size,
                    "rotation_axis": rotation_axis,
                    "rotation_angle": angle,
                    "up_axis": up_axis,
                    "proj_width": proj_width,
                    "proj_height": proj_height,
                }
            )

        # Use environment size for packing area with some margin
        bin_width, bin_height = (
            env_size[0] * 0.8,
            env_size[1] * 0.8,
        )  # 80% of environment size

        # Pack parts with random positions
        packed = self._pack_2d(part_info, bin_width, bin_height)

        assert packed, ValueError("Failed to find valid positions for all parts")

        # Position parts in the environment
        for x, y, info, w, h in packed:
            aabb_min = info["aabb_min"]
            aabb_size = info["aabb_size"]
            rotation_axis = info["rotation_axis"]
            rotation_angle = info["rotation_angle"]
            up_axis = info["up_axis"]

            # Calculate center position in the bin with some random jitter
            jitter_x = np.random.uniform(-50, 50)  # ±50mm jitter
            jitter_y = np.random.uniform(-50, 50)

            # Calculate position ensuring parts stay within environment bounds
            half_env_x = env_size[0] / 2
            half_env_y = env_size[1] / 2

            # Calculate centered position with jitter, ensuring parts stay within bounds
            pos_x = max(
                -half_env_x + w / 2,
                min(half_env_x - w / 2, x + w / 2 - bin_width / 2 + jitter_x),
            )
            pos_y = max(
                -half_env_y + h / 2,
                min(half_env_y - h / 2, y + h / 2 - bin_height / 2 + jitter_y),
            )

            # Position on ground with proper height based on orientation
            # The height should be half the size in the up direction since we're centering
            if up_axis == 0:  # X is up
                height = aabb_size[0] / 2
                pos = Pos(pos_x, pos_y, height)
            elif up_axis == 1:  # Y is up
                height = aabb_size[1] / 2
                pos = Pos(pos_x, pos_y, height)
            else:  # Z is up (default)
                height = aabb_size[2] / 2
                pos = Pos(pos_x, pos_y, height)

            # Apply transformation to part
            if abs(rotation_angle) > 1e-6:  # Only rotate if angle is not zero
                from build123d import Axis

                # Create an axis of rotation at the part's center
                rotation_axis_obj = Axis(
                    origin=info["part"].center(), direction=rotation_axis
                )
                info["part"] = info["part"].rotate(
                    rotation_axis_obj, rotation_angle * 180 / np.pi
                )  # Convert to degrees
            info["part"] = info["part"].move(pos)

        # Rebuild compound with new part positions
        # Note: You'll need to implement this part based on your compound structure
        # For example: compound = Compound(children=[info['part'] for info in part_info])
        new_compound = Compound(children=[info["part"] for info in part_info])
        return new_compound
