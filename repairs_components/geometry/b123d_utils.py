from build123d import *
import build123d as bd
import ocp_vscode


def fastener_hole(radius: float, depth: float, id: int):
    """
    Create a fastener hole with a specified radius and depth, and optionally attach a revolute joint.
    It creates a collision point with which a prospective fastener can intersect and allows for a joint.

    Args:
        radius (float): The radius of the hole.
        depth (float): The depth of the hole.
        id (int): The unique id of the hole.

    Returns:
        Tuple: A tuple containing the created Hole object and its location.
    """
    if radius > 2.5 or depth > 4:
        print("Warning: radius > 2.5 or depth > 4. Is this for centimeters??")

    # make a hole
    fastener_hole1 = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, -radius))
    # tuple_pos=[loc.position.to_tuple() for loc in fastener_loc.locations]
    joint_axis = Axis.Z
    joint = RigidJoint(label=f"fastener_hole_{id}")

    return fastener_hole1, fastener_loc, joint  # TODO - add joint axis?


###debug:
