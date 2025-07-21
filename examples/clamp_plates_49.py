import copy
from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole


class ClampPlates(EnvSetup):
    "Clamp two plates together with a fastener."

    def desired_state_geom(self) -> Compound:
        with BuildPart() as top_plate:
            Box(30, 20, 10)
            fillet(  # just a differentiator to see rotation.
                top_plate.faces().filter_by(Axis.X).sort_by(Axis.X).last.edges(),
                radius=1,
            )

            joints = []
            with Locations(top_plate.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                grid_locs = GridLocations(15, 0, 2, 1)
            for i, loc in enumerate(grid_locs.local_locations):
                with Locations(loc):
                    _hole, _loc, joint = fastener_hole(radius=3, depth=16, id=i)
                    joints.append(joint)

        bottom_plate = top_plate.part.moved(Location((0, 0, -5)))

        start_fastener = Fastener(
            False,
            initial_body_a="top_plate@solid",
            initial_body_b="bottom_plate@solid",
            b_depth=10,
        ).bd_geometry()
        fasteners = []
        for i, loc in enumerate(grid_locs.locations):
            moved_fastener = start_fastener.located(loc)
            moved_fastener.joints["fastener_joint_a"].connect_to(joints[i])
            moved_fastener.joints["fastener_joint_b"].connect_to(
                bottom_plate.joints[joints[i].label]
            )
            fasteners.append(moved_fastener)
        top_plate.part.label = "top_plate@solid"
        bottom_plate.label = "bottom_plate@solid"

        # don't add start_fastener (it's for copy only)
        return Compound(children=[top_plate.part, bottom_plate, *fasteners]).moved(
            Location((320, 320, 320))
        )

    @property
    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        return {}


if __name__ == "__main__":
    env_setup = ClampPlates()
    show(env_setup.desired_state_geom())
    env_setup.validate()
