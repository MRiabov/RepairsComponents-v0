from build123d import *
from ocp_vscode import *
import ocp_vscode

# Round plug (male)
HANDLE_RADIUS = 6  # mm
HANDLE_LENGTH = 14  # mm
CONNECTOR_RADIUS = 2.5  # mm
CONNECTOR_LENGTH = 10  # mm
CENTER_HOLE_RADIUS = 0.6  # mm

with BuildPart() as round_plug:
    with Locations((0, 0, 0)):
        Cylinder(HANDLE_RADIUS, HANDLE_LENGTH)
    with Locations((0, 0, HANDLE_LENGTH / 2)):
        Cylinder(CONNECTOR_RADIUS, CONNECTOR_LENGTH)
        Hole(CENTER_HOLE_RADIUS, CONNECTOR_LENGTH + 1)

# Mother connector (female socket)
SOCKET_OUTER_RADIUS = 6.2  # mm, slight tolerance
SOCKET_LENGTH = 10  # mm
SOCKET_INNER_RADIUS = CONNECTOR_RADIUS + 0.2  # mm, small clearance
SOCKET_HOLE_RADIUS = CENTER_HOLE_RADIUS + 0.3  # mm

with BuildPart() as round_socket:
    with Locations((0, 0, 0)):
        Cylinder(SOCKET_OUTER_RADIUS, SOCKET_LENGTH)
        Hole(SOCKET_INNER_RADIUS, SOCKET_LENGTH)
        Hole(SOCKET_HOLE_RADIUS, SOCKET_LENGTH + 1)


sockets = Compound(
    [
        round_plug.part.moved(Location((10, 0, 0))),
        round_socket.part.moved(Location((-10, 0, 0))),
    ]
)
export_gltf(
    round_plug.part, "geom_exports/electronics/connectors/round_laptop_male.gltf"
)
export_gltf(
    round_socket.part, "geom_exports/electronics/connectors/round_laptop_female.gltf"
)


ocp_vscode.show(sockets)
