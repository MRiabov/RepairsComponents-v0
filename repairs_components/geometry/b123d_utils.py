from build123d import *
import build123d as bd
import ocp_vscode
from pathlib import Path
import trimesh
import os


def fastener_hole(radius: float, depth: float | None, id: int):
    """
    Create a fastener hole with a specified radius and depth, and optionally attach a revolute joint.
    It creates a collision point with which a prospective fastener can intersect and allows for a joint.

    Args:
        radius (float): The radius of the hole in mm.
        depth (float | None): The depth of the hole in mm. If None, the hole is through.
        id (int): The unique id of the hole.

    Returns:
        Tuple: A tuple containing the created Hole object and its location.
    """

    # make a hole
    fastener_hole = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, 0))
    # (0, 0, -radius) -radius was a bug I understand.
    if depth is None:
        depth = fastener_hole.vertices().sort_by(Axis.Z).first.Z
        depth = -depth  # if "-", it is through

    joint = RigidJoint(
        label=f"fastener_hole_{id}#d{depth}", joint_location=fastener_loc.locations[0]
    )

    return fastener_hole, fastener_loc, joint  # TODO - add joint axis?


def fastener_hole_joint_name(id: int, depth: float | None):
    return f"fastener_hole_{id}#d{depth}"
    # note: new convention: "#" in labels means parameter.


def fastener_hole_info_from_joint_name(name: str):
    id = int(name.split("_")[1].split("#d")[0])
    depth = float(name.split("#d")[1])
    return id, depth


def export_obj(part: Part, obj_path: Path, glb_path: Path | None = None) -> Path:
    """
    Convert a GLB file to OBJ format using trimesh.

    Args:
        glb_path: Path to the input GLB file
        obj_path: Path to save the output OBJ file. If None, replaces .glb with .obj

    Returns:
        Path to the created OBJ file
    """
    if glb_path is None:
        glb_path = obj_path.with_suffix(".glb")
    if part.children:
        print(
            "Warning, exporting a compound with children in `export_obj`. Is this expected?"
        )
    # Create parent directories if they don't exist
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    export_successful = export_gltf(part, str(glb_path), binary=True)
    assert export_successful, "Failed to export GLB file"

    # Load the GLB file in trimesh and export
    mesh = trimesh.load(glb_path, file_type="glb")
    mesh.export(obj_path, file_type="obj")


###debug utils:


def get_all_joints(compound: Compound):
    return [
        joint.connected_to for c in compound.children for joint in c.joints.values()
    ]


def compute_inertial(part: Part, density=1):
    assert part.center(CenterOf.BOUNDING_BOX) == (0, 0, 0), "Part must be centered"
    return part.matrix_of_inertia * density


# todo: put this everywhere where necessary
def recenter_part(part: Part):
    "Return a recentered to origin copy of self."
    assert isinstance(part, Part), "Part must be a Part object"
    center = part.center(CenterOf.BOUNDING_BOX)
    return part.moved(Pos(-center))
