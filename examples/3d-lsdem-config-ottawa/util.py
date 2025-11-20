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
def rebuild_connectivity(p, material, dim=3, nci=True, parallel=True):
    from scipy.spatial import cKDTree
    pos_d = p.pos[:, :dim]
    tree  = cKDTree(pos_d)

    # horizon must match the final scale
    horizon = float(material.delta)
    if horizon <= 0:
        raise ValueError(f"Invalid horizon: {horizon}")

    cand_pairs = list(tree.query_pairs(horizon))  # [(i,j),...]

    if nci:
        # your existing dispatch logic
        if dim == 3:
            dispatcher = {
                'kalthoff':       p.single_bond_3d_kalthoff,
                'plus3d':         p.single_bond_3d_plus3d,
                'hollow_sphere':  p.single_bond_3d_sphere,
            }
            worker = dispatcher.get(p.name, p.single_bond_3d)
        else:
            dispatcher = {
                'plate_2d_circle_void':         p.single_bond_2d_plate_circle_void,
                'plate_2d_vnotch_circle_void':  p.single_bond_2d_plate_vnotch_circle_void,
                'plate_2d_notch_1':             p.single_bond_2d_plate_2d_notch_1,
                'plate_2d_notch_2':             p.single_bond_2d_plate_2d_notch_2,
            }
            worker = dispatcher.get(p.name, p.test_single_bond)

        if parallel:
            from multiprocessing import get_context
            with get_context("fork").Pool() as pool:
                results = pool.map(worker, cand_pairs)
            bonds = [q for q in results if q is not None]
        else:
            bonds = [q for q in cand_pairs if worker(q) is not None]
    else:
        bonds = cand_pairs

    # always store as (M,2)
    p.NArr = np.asarray(sorted(bonds), dtype=np.int64).reshape(-1, 2)
import numpy as np

def safe_rotation_matrix_from_quat(q):
    # q = [w, x, y, z] or [x, y, z, w]? adjust as needed!
    # Below assumes [w, x, y, z]. If yours is [x,y,z,w], reorder first.
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must have 4 components")

    # normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # classic Hamilton formula
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=float)

    # project to SO(3) (fix tiny numerical drift or bad input)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # enforce det=+1
        U[:, -1] *= -1
        R = U @ Vt
    return R
def centroid(P):
    return np.mean(P, axis=0)

def rotate_about_center(P, R, c):
    return (P - c) @ R.T + c  # or R@(P-c).T + c[:,None]

def place_particle_points(P, R, ct):
    c_loc = centroid(P)
    # rigid rotation about local centroid
    P = rotate_about_center(P, R, c_loc)
    # translate to desired center
    P = P + (ct - c_loc)
    return P
def scale_local(P, s, center):
    return (P - center) * s + center


