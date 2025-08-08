from build123d import *
import build123d as bd
import ocp_vscode
from pathlib import Path
import trimesh
import os
import numpy as np


def fastener_hole(
    radius: float, depth: float | None, id: int, build_part: BuildPart | None = None
):
    """
    Create a fastener hole with a specified radius and depth, and optionally attach a revolute joint.
    It creates a collision point with which a prospective fastener can intersect and allows for a joint.

    Args:
        radius (float): The radius of the hole in mm.
        depth (float | None): The depth of the hole in mm. If None, the hole is through.
        id (int): The unique id of the hole.
        build_part: context of the build_part. If depth is None, necessary for checking of actual depth.

    Returns:
        Tuple: A tuple containing the created Hole object and its location.
    """
    if depth is None:
        # depth = fastener_hole.vertices().sort_by(Axis.Z).first.Z
        # TODO: check if this is absolute or relative.
        assert build_part is not None, (
            "build_part must be provided if depth is None (hole is through)"
        )
        # export depth - this is the depth we are using to calculate connection depth.
        export_depth = np.linalg.norm(
            np.array(
                tuple(
                    Axis.Z.intersect(build_part.part).vertices().sort_by(Axis.Z).first
                )
            )
            - np.array((tuple(Locations((0, 0, 0)).locations[0].position))),
        )  # the distance between the starting pos of the hole and it's end.
    else:
        assert depth > 0, "Depth must be positive."
        export_depth = depth

    # make a hole
    fastener_hole = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, 0))
    # (0, 0, -radius) -radius was a bug I understand.
    is_through = depth is None

    joint = RigidJoint(
        label=fastener_hole_joint_name(id, export_depth, is_through),
        joint_location=fastener_loc.locations[0],
    )

    return fastener_hole, fastener_loc, joint  # TODO - add joint axis?


def fastener_hole_joint_name(id: int, connection_depth: float, is_through: bool):
    assert connection_depth is not None and connection_depth > 0, (
        "Depth must be a positive float."
    )
    hole_type = "through" if is_through else "blind"
    return f"fastener_hole_{id}#{connection_depth}#{hole_type}"
    # note: new convention: "#" in labels means parameter. param.


def fastener_hole_info_from_joint_name(name: str):
    parts = name.split("#")
    id = int(parts[0].split("_")[2])
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


def connect_fastener_to_joint(
    fastener_geom, hole_joint, fastener_joint_name: str = "fastener_joint_a"
):
    assert fastener_geom.label and hole_joint.parent.label, (
        f"Connected fastener and joint parent labels are not set. Fastener: {fastener_geom.label}, joint parent: {hole_joint.parent.label}"
    )
    assert fastener_geom.label.endswith("@fastener"), (
        f"Fastener label does not end with '@fastener'. Fastener: {fastener_geom.label}"
    )
    assert hole_joint.parent.label.endswith(("@solid", "@fixed_solid")), (
        f"Joint parent label does not end with '@solid' or '@fixed_solid'. Parent: {hole_joint.parent.label}"
    )
    assert fastener_joint_name in fastener_geom.joints, (
        f"Fastener joint name {fastener_joint_name} not found in fastener geometry."
    )
    fastener_geom = fastener_geom.locate(
        hole_joint.parent.global_location
        * hole_joint.relative_location  # local location? to test.
    )  # must do relocation and connection.
    fastener_geom.joints[fastener_joint_name].connect_to(hole_joint)
    # TODO: assert that hole depth is not too big.


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
    compound: Compound, ignored_labels=("connector_def",), assertion=True
):
    assert len(compound.children) > 1, (
        "filtered_intersection_check requires at least two children in compound"
    )
    any_intersect, intersecting_components, intersect_volume = (
        compound.do_children_intersect()
    )
    # Check if there's any intersection that's not just between connector_defs

    # Logic: any_intersect is true if *any* children intersect. Then check if
    # *all* intersecting components are ignored labels with the same volume as the
    # intersecting volume. If so, then there's no invalid intersection.
    has_invalid_intersection = any_intersect and not all(
        any(
            child.label.endswith(ignored_labels)
            and np.isclose(intersect_volume, child.volume)
            for child in component.children
        )
        if component.children
        else False
        for component in intersecting_components
    )

    if assertion:
        assert not has_invalid_intersection, (
            f"Non-connector parts intersect. Intersecting parts: {[(part.label, part.volume) for part in intersecting_components]}. "
            f"Intersecting volume: {intersect_volume}."
        )
    return has_invalid_intersection, intersecting_components, intersect_volume
