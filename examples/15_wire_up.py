from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from ocp_vscode import *
from repairs_components.geometry.b123d_utils import fastener_hole
from repairs_components.geometry.connectors.models.europlug import Europlug
import numpy as np


class WireUp(EnvSetup):
    "An env with 8 connectors that need to be correctly wired into each hole."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        with BuildPart() as elec_panel:
            Box(60, 40, 100)
            with Locations(elec_panel.faces().filter_by(Axis.Y).sort_by(Axis.Y).first):
                hole_grid_locs = GridLocations(0, 30, 1, 4)
                with hole_grid_locs:
                    connector_hole = Box(5, 5, 5, mode=Mode.SUBTRACT)
                for i in range(4):
                    joint = RigidJoint(
                        f"always_{i}", joint_location=hole_grid_locs.locations[i]
                    )
        male_geoms = []
        female_geoms = []
        connect_positions = []
        for i in range(4):
            male_geom, connect_pos, female_geom, _ = Europlug(
                f"couple_{i}@connectors"
            ).bd_geometry((0, 0, 0), connected=True)
            female_geom.move(hole_grid_locs.locations[i])
            male_geom.move(hole_grid_locs.locations[i])


            male_geoms.append(male_geom)
            female_geoms.append(female_geom)
            connect_positions.append(connect_pos)
            joint = RigidJoint(f"always_{i}", to_part=female_geom)
            joint.connect_to(other=elec_panel.part.joints["always_" + str(i)])
            # note: if label name is "always", keep it even despite perturbations.
        return Compound(children=(male_geoms + female_geoms + [elec_panel.part]))

        # FIXME: no way to define that a connector would be constrained to other body.

    def linked_groups(self) -> list[tuple[str, ...]]:
        return [()]


show(WireUp().desired_state_geom())
