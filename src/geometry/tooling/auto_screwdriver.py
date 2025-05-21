from build123d import *  # noqa: F403

# noqa: F405
import ocp_vscode
import build123d

# An automatic end effector for pick-and-place screwdriver.
BODY_DIAMETER = 10
BODY_HEIGHT = 15
BOTTOM_HOLE_DIAMETER = 1
BOTTOM_FILLET_RADIUS = 4

END_EFFECTOR_ATTACHMENT_HOLE_DIAMETER = 5
END_EFFECTOR_ATTACHMENT_HOLE_DEPTH = 3


with BuildPart() as auto_screwdriver:
    body = Cylinder(BODY_DIAMETER / 2, BODY_HEIGHT)
    lower_face = body.faces().sort_by(Axis.Z)[0]
    fillet(lower_face.edges(), radius=BOTTOM_FILLET_RADIUS)
    with Locations(lower_face):
        Hole(BOTTOM_HOLE_DIAMETER / 2, 3)
    with Locations(body.faces().sort_by(Axis.Z)[-1]):
        Hole(
            END_EFFECTOR_ATTACHMENT_HOLE_DIAMETER / 2,
            END_EFFECTOR_ATTACHMENT_HOLE_DEPTH,
        )


export_gltf(auto_screwdriver.part, "geom_exports/auto_screwdriver.gltf")
ocp_vscode.show(auto_screwdriver.part)
