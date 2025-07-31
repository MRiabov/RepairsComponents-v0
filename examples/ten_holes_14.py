import copy
from ocp_vscode import *
from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from build123d import *
from repairs_components.geometry.b123d_utils import fastener_hole


class TenHoles(EnvSetup):
    "Put 10 fasteners in 10 holes."

    # note: everything is created in mm.

    def desired_state_geom(self) -> Compound:
        with BuildPart() as base_box:
            Box(200, 200, 200)
            joints = []
            with Locations(base_box.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                grid_locs = GridLocations(15, 0, 10, 1)
            for i, loc in enumerate(grid_locs.local_locations):
                with Locations(loc):
                    _hole, _loc, joint = fastener_hole(radius=3, depth=17, id=i)
                    joints.append(joint)
            debug__ = ""

        start_fastener = Fastener(initial_hole_id_a=0).bd_geometry()
        fasteners = []
        for i, loc in enumerate(grid_locs.locations):
            moved_fastener = start_fastener.located(loc)
            #^NOTE: quite likely incorrect - for when the base body is moved.
            # The correct version is something like moved_fastener = fastener_geom.located(solid_with_hole.location*hole_joint.location).location
            # and then add the joint.


            # moved_fastener = copy.copy(start_fastener) # for whichever reason puts all fasteners one last hole.
            # Probably due to references...
            moved_fastener.joints["fastener_joint_a"].connect_to(joints[i])
            fasteners.append(moved_fastener)
        base_box.part.label = "base_box@fixed_solid"

        # don't add start_fastener (it's for copy only)
        return Compound(children=[base_box.part, *fasteners]).moved(
            Location((320, 320, 320))
        )

    @property
    def linked_groups(self) -> dict[str, tuple[list[str]]]:
        return {}


if __name__ == "__main__":
    env_setup = TenHoles()
    show(env_setup.desired_state_geom())
    env_setup.validate()
