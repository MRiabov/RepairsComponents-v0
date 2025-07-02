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
            Box(20, 20, 20)
            with Locations(base_box.faces().filter_by(Axis.Z).sort_by(Axis.Z).last):
                grid_locs = GridLocations(1.5, 0, 10, 1)
                with grid_locs:
                    _hole, _locs, joint1 = fastener_hole(radius=0.3, depth=1.6, id=0)

        fastener_, collision_detection_position = Fastener(
            False, initial_body_a="base_box@solid"
        ).bd_geometry()
        fasteners = []
        for i, loc in enumerate(grid_locs.locations):
            fastener = fastener_.moved(loc)
            fastener.joints["fastener_joint_a"].connect_to(joint1)
            fasteners.append(fastener)
        base_box.part.label = "base_box@solid"

        Compound(children=[base_box.part, *fasteners]).show_topology()

        return Compound(children=[base_box.part, *fasteners]).moved(
            Location((10, 10, 10))
        )


if __name__ == "__main__":
    env_setup = TenHoles()
    show(env_setup.desired_state_geom())
    env_setup.validate()
