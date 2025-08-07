import copy
from abc import ABC, abstractmethod

import numpy as np
from build123d import Compound, Part

from repairs_components.geometry.b123d_utils import filtered_intersection_check
from repairs_components.geometry.base_env.tooling_stand_plate import render_and_save


class EnvSetup(ABC):
    """A setup for a single problem to solve. User needs to ONLY specify geometry and naming
    for parts in desired_state_geom.

    The initial geometry state would be then specified by `Task`s, and RepairsSimState
    will be created by `translation.translate_compound_to_sim_state`. The environment will be added by"""

    STANDARD_ENV_SIZE = (640, 640, 640)
    BOTTOM_CENTER_OF_PLATE = (0, STANDARD_ENV_SIZE[1] / 2 + 200, 200)

    @abstractmethod
    def desired_state_geom(self) -> Compound:
        """
        Set up the simulation environment's desired final state. Should be subclassed, but not called by others. Call get_desired_state instead.

        Args:
            sim: The simulation object to be set up.
        """
        raise NotImplementedError("Desired state geom must be implemented.")

    @property
    @abstractmethod
    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        """Return a dictionary of linked groups.
        keys:
        - mech_linked: rigidly constrained groups, useful for permanent constraints.
        """
        return {}

    def _debug_render(self, scene, camera_1, camera_2):
        render_and_save(scene, camera_1, camera_2)

    def validate(self):
        """Validate the environment setup."""
        geom = self.desired_state_geom()
        geom_intersect_check = copy.copy(geom)

        # check all parts being labeled
        assert all(
            (part.label and "@" in part.label) or len(part.children) > 0
            for part in geom.children
        ), f"All children must have labels. Currently have: {geom.children}"
        # remove connector defs from intersection check

        # DEBUG: TEMPORARILY removed intesection check because of underlying bug in build123d (or expected bug)
        # filtered_intersection_check(geom_intersect_check, assertion=True)
        # /debug.

        # check bounding box
        aabb = geom.bounding_box()
        assert (
            np.array(tuple(aabb.size)) <= np.array(self.STANDARD_ENV_SIZE)
        ).all(), (
            f"Compound must be within the environment. Current AABB size {aabb.size} with: {aabb.min} to {aabb.max}. Environment size: {self.STANDARD_ENV_SIZE}."
        )  # note: was min>=0 and max<=STANDARD_ENV_SIZE no point constraining myself though

        fastener_count = len(
            [part for part in geom.children if part.label.endswith("@fastener")]
        )
        assert fastener_count < 12, (
            f"At most 12 fasteners can be present. Currently: {fastener_count}"
        )  # supported by buffers.
        for part in geom.leaves:
            part_name, part_type = part.label.split("@", 2)
            supported_types = (
                "solid",
                "fixed_solid",
                "fastener",
                "connector",
                "button",
                "led",
                "switch",
                "connector_def",
            )
            assert part_type in supported_types, (
                f"Part type must be one of {supported_types}. Currently have: {part_type}."
            )
            if part_type == "fastener":
                a_populated = (
                    part.joints["fastener_joint_a"].connected_to
                    and part.joints["fastener_joint_a"].connected_to.parent is not None
                )
                b_populated = (
                    part.joints["fastener_joint_b"].connected_to
                    and part.joints["fastener_joint_b"].connected_to.parent is not None
                )
                assert a_populated or b_populated, (
                    "Fastener must be connected to a part."
                )
                if a_populated:
                    parent = part.joints["fastener_joint_a"].connected_to.parent
                    assert isinstance(parent, Part), (
                        "Fastener joint A which is marked as connected must be connected to a part."
                    )
                    assert parent.label, (
                        "Fastener joint A is connected to an unlabeled part."
                    )
                    assert parent.label.endswith(("@solid", "@fixed_solid")), (
                        f"Fastener joint A must be connected to a part of type solid. Current label: {parent.label}"
                    )
                else:
                    parent = part.joints["fastener_joint_b"].connected_to.parent
                    assert isinstance(parent, Part), (
                        "Fastener joint B which is marked as connected must be connected to a part."
                    )
                    assert parent.label, (
                        "Fastener joint B is connected to an unlabeled part."
                    )
                    assert parent.label.endswith(("@solid", "@fixed_solid")), (
                        f"Fastener joint B must be connected to a part of type solid. Current label: {parent.label}"
                    )

            elif part_type == "connector":
                assert part_name.endswith(("_male", "_female")), (
                    "Expected connector to end with male or female strings."
                )
        # if linked_groups are specified, validate.
        assert isinstance(self.linked_groups, dict), (
            f"Linked groups must be a dictionary, currently is {type(self.linked_groups)}"
        )  # dev note: if it is a method, don't forget to add @property
        if not self.linked_groups:
            return True
        mechanical_groups = self.linked_groups["mech_linked"]
        assert isinstance(mechanical_groups, tuple), (
            f"Linked groups must be a tuple, currently is {type(mechanical_groups)}"
        )
        for group in mechanical_groups:
            assert isinstance(group, list), (
                f"Linked groups must be a tuple, currently is {type(group)}"
            )
            assert all(isinstance(part_name, str) for part_name in group), (
                "Expected all part names to be strings."
            )
            geom_leaves_labels = [part.label for part in geom.leaves]
            for part_name in group:
                assert part_name in geom_leaves_labels, (
                    f"Part {part_name} not found in geometry."
                )

        return True
