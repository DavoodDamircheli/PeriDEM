import numpy as np




# ----------------------------
# Helpers
# ----------------------------
def peridynamic_timestep(E, rho, dx, alpha=0.2):
    """
    Compute stable peridynamic time step Δt based on CFL-type condition.
    
    Parameters
    ----------
    E : float
        Young's modulus (Pa)
    rho : float
        Density (kg/m^3)
    dx : float
        Particle spacing (m)
    alpha : float, optional
        Stability (CFL) factor, typically between 0.2–0.5
        
    Returns
    -------
    dt : float
        Stable time step (s)
    c : float
        Wave speed (m/s)
    """
    c = (E / rho) ** 0.5  # wave speed
    dt = alpha * dx / c
    return dt, c
# --- rotation utils (no recentering; rotate about origin) --- #
def euler_to_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx

def rotation_matrix_from_quaternion(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def detect_rotation_matrix(rot_vec):
    r = np.asarray(rot_vec).flatten()
    if r.size == 3:
        rx, ry, rz = r
        return euler_to_matrix(rx, ry, rz)
    if r.size == 4:
        return rotation_matrix_from_quaternion(r)
    if r.size == 9:
        return r.reshape(3, 3)
    raise ValueError(f"Unsupported rotation vector length {r.size}")

def rotate_about_origin(P, R):
    """Rotate Nx3 points about the origin (no centroid shift)."""
    return np.asarray(P) @ R.T

def circumscribed_radius(P):
    """
    Quick radius proxy: max distance from centroid.
    Good enough to verify scaling is in the right ballpark.
    """
    P = np.asarray(P)
    c = P.mean(axis=0)
    return np.max(np.linalg.norm(P - c, axis=1))


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_indices(spec: str):
    """
    Accepts "0,1,2" or "0:10" or "0:10:2".
    """
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            a, b = parts
            return list(range(a, b))
        elif len(parts) == 3:
            a, b, s = parts
            return list(range(a, b, s))
        else:
            raise ValueError(f"Bad index slice: {spec}")
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    if spec == "":
        return []
    return [int(spec)]



def rotate_about_origin(pts, R):
    """Return points rotated about the origin by 3x3 matrix R."""
    P = np.asarray(pts)
    R = np.asarray(R).reshape(3,3)
    return P @ R.T   # (N,3) x (3,3)^T

def rotate_about_center(pts, R, center):
    """Return points rotated about `center` by 3x3 matrix R."""
    P = np.asarray(pts)
    R = np.asarray(R).reshape(3,3)
    c = np.asarray(center, dtype=P.dtype).reshape(1,3)
    return (P - c) @ R.T + c


def scale_for_overlap(pts, s, ct=None):
    """
    Shrink a SINGLE particle about its own center so its centroid stays put.
    Use when you want the particle smaller without moving it.
    """
    pts = np.asarray(pts)
    c = np.asarray(ct) if ct is not None else pts.mean(axis=0)
    return (pts - c) * float(s) + c


def scale_compact(pts, s, c_global):
    """
    Shrink the WHOLE cluster coherently: scale every particle about
    the SAME global center c_global so inter-particle spacing shrinks too.
    """
    pts = np.asarray(pts)
    return (pts - np.asarray(c_global)) * float(s) + np.asarray(c_global)
def centroid(A): 
    return A.mean(axis=0)
