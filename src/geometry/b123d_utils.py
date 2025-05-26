from build123d import *
import build123d as bd
import ocp_vscode


def fastener_hole(radius: float, depth: float):
    # make a hole
    fastener_hole1 = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, -radius))
    # tuple_pos=[loc.position.to_tuple() for loc in fastener_loc.locations]
    return fastener_hole1, fastener_loc


###debug:
