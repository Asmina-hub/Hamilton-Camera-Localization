"""Small rigid-transform helpers for editing /tf_static and the first pose."""
import math
import numpy as np

NS = 'constructor0'


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(tr):
    q, p = tr.rotation, tr.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = p.x, p.y, p.z
    return T


def perturb_camera(msg, p):
    """Rotate / shift camera `p['cam']` in a decoded TFMessage (base_link axes, about the camera centre).

    Only the last link of the chain base_link -> ... -> camera_<cam> is edited. Returns True if found."""
    child = f'{NS}/camera_{p["cam"]}'
    links = {t.child_frame_id: t for t in msg.transforms}
    if child not in links:
        return False
    # T_base_parent of the camera's parent frame
    parent = links[child].header.frame_id
    T_bp, f = np.eye(4), parent
    while f != f'{NS}/base_link':
        if f not in links:            # chain not complete in this message: cannot edit safely
            return False
        T_bp = _T(links[f].transform) @ T_bp
        f = links[f].header.frame_id
    T_bc = T_bp @ _T(links[child].transform)
    T_new = np.eye(4)
    T_new[:3, :3] = rpy(math.radians(p['droll']), math.radians(p['dpitch']), math.radians(p['dyaw'])) @ T_bc[:3, :3]
    T_new[:3, 3] = T_bc[:3, 3] + [p['dx'], p['dy'], p['dz']]
    T_pc = np.linalg.inv(T_bp) @ T_new
    tr = links[child].transform
    tr.translation.x, tr.translation.y, tr.translation.z = (float(v) for v in T_pc[:3, 3])
    tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = (float(v) for v in R_to_quat(T_pc[:3, :3]))
    return True


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def set_pose_2d(tr, x, y, yaw):
    """Set x, y, yaw of a geometry_msgs/Transform, keeping its roll and pitch."""
    q = tr.rotation
    R = quat_to_R(q.x, q.y, q.z, q.w)
    dyaw = yaw - yaw_of(q)
    Rn = rpy(0, 0, dyaw) @ R
    tr.translation.x, tr.translation.y = float(x), float(y)
    q.x, q.y, q.z, q.w = (float(v) for v in R_to_quat(Rn))
