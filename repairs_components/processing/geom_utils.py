import torch
import torch.nn.functional as F


def get_connector_pos(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    rel_connector_pos: torch.Tensor,
):
    """
    Get the position of a connector relative to its parent. Used both in translation from compound to sim state and in screwdriver offset.
    Note: expects batched inputs of rel_connector_pos, ndim=2.
    Returns:
        torch.Tensor: The position of the connector relative to its parent. [B, 3]
    """
    return (
        parent_pos
        + rel_connector_pos
        + 2
        * torch.cross(
            parent_quat[..., 1:],
            torch.cross(parent_quat[..., 1:], rel_connector_pos, dim=-1)
            + parent_quat[..., 0:1] * rel_connector_pos,
            dim=-1,
        )
    )


def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Quaternion multiplication, q1 ⊗ q2.
    Inputs: [..., 4] tensors where each quaternion is [w, x, y, z]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    q = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    if normalize:
        q = F.normalize(q, dim=-1)
    return q


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Returns the conjugate of a quaternion [w, x, y, z]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_angle_diff_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Returns the angle in degrees between rotations represented by q1 and q2."""
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    q_rel = quat_multiply(quat_conjugate(q1), q2)
    w = torch.clamp(torch.abs(q_rel[..., 0]), -1.0, 1.0)  # safe acos
    angle_rad = 2.0 * torch.acos(w)
    return torch.rad2deg(angle_rad)


def are_quats_within_angle(
    q1: torch.Tensor, q2: torch.Tensor, max_angle_deg: float | torch.Tensor
) -> torch.Tensor:
    """Returns True where angular distance between q1 and q2 is ≤ max_angle_deg."""
    q1 = sanitize_quaternion(q1)
    q2 = sanitize_quaternion(q2)
    return quat_angle_diff_deg(q1, q2) <= max_angle_deg


def sanitize_quaternion(q: torch.Tensor | tuple, atol: float = 1e-4) -> torch.Tensor:
    """
    Assert that a quaternion tensor is valid and normalized.

    Args:
        q (torch.Tensor): Tensor of shape (..., 4), representing quaternion(s).
        atol (float): Tolerance for unit-norm check.

    Returns:
        torch.Tensor: The same tensor, optionally normalized for further use.
    """
    if isinstance(q, tuple):
        q = torch.tensor(q, dtype=torch.float)
    # Shape check
    assert q.shape[-1] == 4, (
        f"Expected last dimension to be 4 (quaternion), got {q.shape}"
    )

    # Check not out of range values.
    assert torch.all(torch.isfinite(q)), "Quaternion contains NaN or Inf"

    # Unit norm check
    norm = torch.linalg.norm(q, dim=-1)
    norm_error = (norm - 1).abs()
    max_error = norm_error.max().item()
    assert torch.all(norm_error <= atol), (
        f"Quaternion norm deviates from 1 by up to {max_error}, tolerance is {atol}"
    )

    return q


def euler_deg_to_quat_wxyz(euler_deg: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles in degrees (roll, pitch, yaw) to a quaternion in WXYZ format.

    Args:
        euler_deg: Rotation around x, y, z-axes in degrees (tensor of shape [..., 3]).

    Returns:
        torch.Tensor of shape (..., 4), representing the quaternion [w, x, y, z].
    """
    assert euler_deg.shape[-1] == 3, (
        f"Expected last dimension to be 3 (euler angles), got {euler_deg.shape}"
    )
    roll = torch.deg2rad(euler_deg[..., 0])
    pitch = torch.deg2rad(euler_deg[..., 1])
    yaw = torch.deg2rad(euler_deg[..., 2])

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.stack([w, x, y, z], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True)


def quaternion_delta(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    """Returns the quaternion that rotates *q_from* into *q_to* (element-wise)."""
    return quat_multiply(q_to, quat_conjugate(q_from))
