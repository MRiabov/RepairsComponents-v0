from abc import ABC, abstractmethod
from training_utils.sim_state import RepairsSimState
from build123d import Compound, Pos
import numpy as np


class Task(ABC):
    @abstractmethod
    def perturb_desired_state(
        self,
        compound: Compound,
        desired_state: RepairsSimState,
        env_size=(640, 640, 640),
    ) -> RepairsSimState:
        """Perturb the desired state of the task."""

        bbox = compound.bounding_box()
        aabb_min = np.array(bbox.min.to_tuple())
        aabb_size = np.array(bbox.size.to_tuple())

        # Calculate the required margin to keep 5 units from each side
        margin = 5
        min_pos = margin - aabb_min  # Minimum position to move to maintain margin
        max_pos = np.array(env_size) - aabb_size - aabb_min - margin  # Maximum position

        assert (max_pos > 0).all(), "Compound is too large for the environment."

        # Generate random offset within the movable range
        random_offset = np.random.uniform(
            np.minimum(min_pos, max_pos), np.maximum(min_pos, max_pos)
        )

        # Move the compound by the random offset
        compound = compound.move(Pos(*random_offset))

    @abstractmethod
    def perturb_initial_state(
        self, compound: Compound, initial_state: RepairsSimState
    ) -> RepairsSimState:
        raise NotImplementedError


class AssembleTask(Task):
    def perturb_desired_state(
        self,
        compound: Compound,
        desired_state: RepairsSimState,
        env_size=(640, 640, 640),
    ) -> RepairsSimState:
        return super().perturb_desired_state(compound, desired_state, env_size)

    def _get_stable_orientation(self, aabb_size):
        """Determine the most stable orientation based on AABB dimensions."""
        dims = np.array(aabb_size)
        aspect_ratios = dims / np.mean(dims)

        if np.any(aspect_ratios > 1.3):
            up_axis = np.argmax(dims)
            if up_axis == 0:
                return (0, np.pi / 2, 0), up_axis
            elif up_axis == 1:
                return (np.pi / 2, 0, 0), up_axis
        return (0, 0, 0), 2  # Default to Z-up

    def _pack_2d(self, rectangles, bin_width, bin_height):
        """Simple 2D bin packing using bottom-left heuristic."""
        packed = []
        used_rectangles = []

        for w, h, part in rectangles:
            best_score = float("inf")
            best_pos = None

            for x in range(0, bin_width - int(w) + 1, 10):
                for y in range(0, bin_height - int(h) + 1, 10):
                    new_rect = (x, y, x + w, y + h)

                    if x + w > bin_width or y + h > bin_height:
                        continue

                    overlap = False
                    for placed in used_rectangles:
                        if not (
                            new_rect[2] <= placed[0]
                            or new_rect[0] >= placed[2]
                            or new_rect[3] <= placed[1]
                            or new_rect[1] >= placed[3]
                        ):
                            overlap = True
                            break

                    if not overlap:
                        score = x + y
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y)

            if best_pos is not None:
                x, y = best_pos
                packed.append((x, y, part, w, h))
                used_rectangles.append((x, y, x + w, y + h))

        return packed

    def perturb_initial_state(
        self, compound: Compound, initial_state: RepairsSimState
    ) -> RepairsSimState:
        """Randomly disassemble the compound into individual parts."""
        parts = compound.solids()
        part_info = []

        # Calculate AABBs and determine stable orientations
        for part in parts:
            bbox = part.bounding_box()
            aabb_min = np.array(bbox.min.to_tuple())
            aabb_max = np.array(bbox.max.to_tuple())
            aabb_size = aabb_max - aabb_min

            rotation, up_axis = self._get_stable_orientation(aabb_size)

            # Project AABB onto ground plane
            if up_axis == 0:  # X is up
                proj_width, proj_height = aabb_size[1], aabb_size[2]
            elif up_axis == 1:  # Y is up
                proj_width, proj_height = aabb_size[0], aabb_size[2]
            else:  # Z is up
                proj_width, proj_height = aabb_size[0], aabb_size[1]

            part_info.append(
                {
                    "part": part,
                    "aabb_min": aabb_min,
                    "aabb_size": aabb_size,
                    "rotation": rotation,
                    "up_axis": up_axis,
                    "proj_width": proj_width,
                    "proj_height": proj_height,
                }
            )

        # Sort by area (largest first)
        part_info.sort(key=lambda x: -(x["proj_width"] * x["proj_height"]))

        # Pack parts
        rectangles = [
            (info["proj_width"], info["proj_height"], info) for info in part_info
        ]

        bin_width, bin_height = 600, 600  # Adjust as needed
        packed = self._pack_2d(rectangles, bin_width, bin_height)

        # Position parts
        for x, y, info, w, h in packed:
            aabb_min = info["aabb_min"]
            aabb_size = info["aabb_size"]
            rotation = info["rotation"]
            up_axis = info["up_axis"]

            # Center in bin
            pos_x = x + w / 2 - bin_width / 2
            pos_y = y + h / 2 - bin_height / 2

            # Position on ground
            if up_axis == 0:  # X is up
                pos = Pos(pos_x, pos_y, aabb_min[0] + aabb_size[0] / 2)
            elif up_axis == 1:  # Y is up
                pos = Pos(pos_x, pos_y, aabb_min[1] + aabb_size[1] / 2)
            else:  # Z is up
                pos = Pos(pos_x, pos_y, aabb_min[2] + aabb_size[2] / 2)

            # Apply transformation
            info["part"] = info["part"].rotate(*rotation).move(pos)

        # Rebuild compound with new part positions
        # Note: You'll need to implement this part based on your compound structure
        # For example: compound = Compound(children=[info['part'] for info in part_info])

        return initial_state
