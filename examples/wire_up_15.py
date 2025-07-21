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
            Box(300, 80, 100)
            joints = []
            with Locations(
                elec_panel.faces().filter_by(Axis.Y).sort_by(Axis.Y).first.offset(-25)
            ):
                hole_grid_locs = GridLocations(0, 65, 1, 4)
                with hole_grid_locs:
                    connector_hole = Box(50, 50, 50, mode=Mode.SUBTRACT)

        male_geoms = []
        female_geoms = []
        connect_positions = []
        for i in range(4):
            male_geom, connect_pos, female_geom, _ = Europlug(i).bd_geometry(
                (0, 0, 0), connected=True
            )
            rotated_loc = Location(hole_grid_locs.locations[i].position, (0, 0, -90))

            female_geom.move(rotated_loc)
            male_geom.move(rotated_loc)
            # I've tested this and I'm quite sure it works, but this feels crazy.

            male_geoms.append(male_geom)
            female_geoms.append(female_geom)
            connect_positions.append(connect_pos)
            # joints[i].connect_to(other=female_geom.joints["native"])
            # note: if label name is "always", keep it even despite perturbations.
        elec_panel.part.label = "elec_panel@fixed_solid"
        return Compound(children=(male_geoms + female_geoms + [elec_panel.part]))

        # FIXME: no way to define that a connector would be constrained to other body.

    @property
    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        all_female_connectors = [
            f"{Europlug(i).get_name(i, False)}@connector" for i in range(4)
        ]
        return {"mech_linked": ([*all_female_connectors, "elec_panel@fixed_solid"],)}


if __name__ == "__main__":
    WireUp().validate()
    show(WireUp().desired_state_geom())
