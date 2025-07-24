from build123d import *
import build123d as bd
import ocp_vscode
from pathlib import Path
import trimesh
import os
import numpy as np


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
    is_through = depth is None
    if depth is None:
        depth = fastener_hole.vertices().sort_by(Axis.Z).first.Z
        # TODO: check if this is absolute or relative.
    else:
        assert depth > 0, "Depth must be positive."

    joint = RigidJoint(
        label=fastener_hole_joint_name(id, depth, is_through),
        joint_location=fastener_loc.locations[0],
    )

    return fastener_hole, fastener_loc, joint  # TODO - add joint axis?


def fastener_hole_joint_name(id: int, depth: float, is_through: bool):
    assert depth is not None and depth > 0, "Depth must be a positive float."
    hole_type = "through" if is_through else "blind"
    return f"fastener_hole_{id}#{depth}#{hole_type}"
    # note: new convention: "#" in labels means parameter. param.


def fastener_hole_info_from_joint_name(name: str):
    parts = name.split("#")
    id = int(parts[0].split("_")[1])
    depth_str, hole_type = parts[1], parts[2]
    depth = float(depth_str)
    assert hole_type in ("through", "blind"), f"Invalid hole type: {hole_type}"
    is_through = hole_type == "through"
    return id, depth, is_through


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


def filtered_intersection_check(
    compound: Compound, filter_labels=("connector_def",), assertion=True
):
    any_intersect, parts, intersect_volume = compound.do_children_intersect()
    # Check if there's any intersection that's not just between connector_defs
    has_invalid_intersection = any_intersect and not all(
        any(
            child.label.endswith(filter_labels)
            and np.isclose(intersect_volume, child.volume)
            for child in part.children
        )
        if part.children
        else False
        for part in parts
    )
    if assertion:
        assert not has_invalid_intersection, (
            f"Non-connector parts intersect. Intersecting parts: {[(part.label, part.volume) for part in parts]}. "
            f"Intersecting volume: {intersect_volume}."
        )
    return has_invalid_intersection, parts, intersect_volume
