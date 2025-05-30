from build123d import *
import build123d as bd
import ocp_vscode


def fastener_hole(radius: float, depth: float, joint_name: str = "fastener_hole1"):
    """
    Create a fastener hole with a specified radius and depth, and optionally attach a revolute joint.
    It creates a collision point with which a prospective fastener can intersect and allows for a joint.

    Args:
        radius (float): The radius of the hole.
        depth (float): The depth of the hole.
        joint_name (str): The name of the revolute joint to attach. Name it as "to_{other_part_name}_{id}.

    Returns:
        Tuple: A tuple containing the created Hole object and its location.
    """
    # make a hole
    fastener_hole1 = Hole(radius=radius, depth=depth)
    fastener_loc = Locations((0, 0, -radius))
    # tuple_pos=[loc.position.to_tuple() for loc in fastener_loc.locations]
    joint_axis = Axis.Z
    RevoluteJoint(joint_name, axis=Axis.Z, to_part=None)

    return fastener_hole1, fastener_loc


###debug:
