from build123d import *
import ocp_vscode
from src.geometry.b123d_utils import add_fastener_markup
from build123d import Vector

# with BuildPart() as test:
#     with Locations((0, 0, 30)):
#         box = Box(10, 10, 10)
#         with Locations((0, 0, 5)):
#             fastener_hole = Hole(radius=3, depth=8)
# test_part = test.part
# test_part.label = "test"
# test_part.color = Color(0.5, 0.5, 0.5, 0.5)


# fastener_markup = add_fastener_markup([fastener_hole], test_part)


def fastener_hole(radius: float, depth: float):
    # make a hole
    fastener_hole1 = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, -radius))
    # tuple_pos=[loc.position.to_tuple() for loc in fastener_loc.locations]

    return fastener_hole1, fastener_loc


# NEXT: do the actual collision checking for fasteners in MuJoCo.
# *how about hot-swapping their position during runtime for being able to constrain anything?

# I also think bolts could be purely visual anyway. It will remove the middleman between the parts.
# next: fastener collision detection mechanism.
# heartbeat: 14.6.25: become highly technical, such that any major US corporation can easily hire you as ML researcher..


# ocp_vscode.show(test_part, fastener_markup)

collision_markers_part = Part()
with BuildPart() as test:
    with Locations((0, 0, 30)):
        box = Box(50, 20, 10)
        with Locations((0, 0, 5)):
            with GridLocations(8, 8, 5, 2):
                hole, fastener_locs = fastener_hole(radius=3, depth=5)

            additional_locs = GridLocations(20, 20, 20, 20).locations
    for loc in fastener_locs.locations + additional_locs:
        print(loc.position.to_tuple())

test_part = test.part
test_part.label = "test"
test_part.color = Color(0.5, 0.5, 0.5, 0.5)


ocp_vscode.show(test_part)
