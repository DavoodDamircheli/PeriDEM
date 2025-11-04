#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import argparse

# project imports
sys.path.append(os.getcwd())
from shape_dict import Shape
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup
import material_dict


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
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

def print_container_info(wall):
    # Axis-aligned rectangular box (Wall3d with x/y/z mins & maxes)
    if all(hasattr(wall, a) for a in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        Lx = wall.x_max - wall.x_min
        Ly = wall.y_max - wall.y_min
        Lz = wall.z_max - wall.z_min
        cx = 0.5 * (wall.x_min + wall.x_max)
        cy = 0.5 * (wall.y_min + wall.y_max)
        cz = 0.5 * (wall.z_min + wall.z_max)
        r_bottom = 0.5 * min(Lx, Ly)  # largest circle that fits on the bottom rectangle (x–y)

        print(f"[container] box size: Lx={Lx:.6g}, Ly={Ly:.6g}, Lz={Lz:.6g}")
        print(f"[container] center: ({cx:.6g}, {cy:.6g}, {cz:.6g})")
        print(f"[container] largest inscribed circle on bottom (x–y) radius: r={r_bottom:.6g}")

    # Cylindrical wall (if your wall exposes center & radius)
    elif all(hasattr(wall, a) for a in ("cyl_center_x", "cyl_center_y", "cyl_radius")):
        cx, cy = wall.cyl_center_x, wall.cyl_center_y
        if all(hasattr(wall, a) for a in ("z_min", "z_max")):
            H = wall.z_max - wall.z_min
            cz = 0.5 * (wall.z_min + wall.z_max)
            print(f"[container] cylinder: radius={wall.cyl_radius:.6g}, height={H:.6g}")
            print(f"[container] center: ({cx:.6g}, {cy:.6g}, {cz:.6g})")
        else:
            print(f"[container] cylinder: radius={wall.cyl_radius:.6g}")
            print(f"[container] center (x, y): ({cx:.6g}, {cy:.6g})")

        # On a cylinder, the biggest circle on the bottom is the cylinder itself
        print(f"[container] largest inscribed circle on bottom (x–y) radius: r={wall.cyl_radius:.6g}")

    else:
        raise AttributeError("Unknown wall type: expected box (x_min..z_max) or cylinder (cyl_center_*, cyl_radius).")





def estimate_h_from_msh(msh_path, sample=5000):
    import meshio, numpy as np
    m = meshio.read(msh_path)
    X = np.asarray(m.points, float)  # in meters
    if X.shape[0] > sample:
        idxs = np.random.default_rng(0).choice(X.shape[0], size=sample, replace=False)
        X = X[idxs]
    dmins = []
    for i in range(X.shape[0]):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        dmins.append(d.min())
    dmins = np.array(dmins)
    return float(np.mean(dmins))
    



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

# --- args (your schema + S) --- #
parser = argparse.ArgumentParser(description='Setup generator (no Open3D)')
parser.add_argument('--N', type=int, default=1, help='number of particles to include (0..N-1)')

parser.add_argument('--particle_rad', type=float, default=40, help='radius of each particle')
parser.add_argument('--vel_val', type=float, default=5e2, help='initial velocity')
parser.add_argument('--acc_val', type=float, default=-10, help='initial acceleration')

parser.add_argument('--G_scale', type=float, default=0.5, help='shear modulus scaling')
parser.add_argument('--Gnot_scale', type=float, default=1e-4, help='shear modulus scaling')
parser.add_argument('--K_scale', type=float, default=0.5, help='bulk modulus scaling')
parser.add_argument('--rho_scale', type=float, default=1, help='density scaling')

parser.add_argument('--meshsize_factor', type=float, default=4, help='meshsize factor vs radius')
parser.add_argument('--delta_factor', type=float, default=3, help='delta factor vs radius')
parser.add_argument('--contact_rad_factor', type=float, default=4, help='contact radius factor vs radius')
parser.add_argument("--gravity", type=float, default=-1e4) #


parser.add_argument('--datapath', type=str, default='/home/davood/projects/ls-shapes',
                    help='path to input positions/rotations')
parser.add_argument('--path', type=str, default='examples_output/', help='output directory')
parser.add_argument('--noplot', action='store_true', help='disable plot')
parser.add_argument('--novis', action='store_true', help='disable visualization')
parser.add_argument('--S', type=float, default=1, help='global scale for coordinates & length params')
args = parser.parse_args()

# --- load positions & rotations --- #
positions = np.loadtxt(os.path.join(args.datapath, 'Input', 'positions_image.dat'))
rotations = np.loadtxt(os.path.join(args.datapath, 'Input', 'rotations_image.dat'))

# --- params --- #
N = int(args.N)
rad = 400e-4#------------------------>for scaling mesh
# radius = args.particle_rad
# meshsize = radius / args.meshsize_factor
# delta = radius / args.delta_factor
# contact_radius = radius / args.contact_rad_factor
S = float(args.S)

MESH_ROOT = '/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/grain-mesh-meshsize-10-voxel-7'
alpha = 3.015
beta = 4
rep_msh = os.path.join(MESH_ROOT, f"mesh_0.msh")
h_mean = estimate_h_from_msh(rep_msh)

h_mean *=S

rad_mesh=rad/60
delta = alpha *rad_mesh 
contact_radius = beta *rad_mesh 
#
# contact_radius *= S
# delta *= S



material = material_dict.ottawa_sand(delta)


# Material properties
a = 0.2    # stability factor

dt, c = peridynamic_timestep(material.E, material.rho, h_mean, a)

print(f"space meshsize  Δx = {h_mean:.2e} m")
print(f"Wave speed c = {c:.2f} m/s")
print(f"Stable time step Δt = {dt:.2e} s")


# --- meshes (strict) --- #
mesh_files = []
missing = []
for ind in range(N):
    path = os.path.join(MESH_ROOT, f'mesh_{ind}.msh')
    (mesh_files if os.path.exists(path) else missing).append(path)

print(f"[info] requested N = {N}")
print(f"[info] found {len(mesh_files)} mesh files")
if missing:
    for m in missing: print("   [missing]", m)
    raise FileNotFoundError(f"Missing {len(missing)} mesh files; cannot continue.")

if len(positions) < N or len(rotations) < N:
    raise ValueError(f"positions/rotations rows < N (pos={len(positions)}, rot={len(rotations)}, N={N}).")

# --- build shapes --- #
SL = ShapeList()
for path in mesh_files:
    SL.append(shape=Shape(P=None, nonconvex_interceptor=None, msh_file=path, scale_mesh_to=float(rad),centroid_origin=True),
              count=1, meshsize=1, material=material)

print("[info] generating particle meshes ...")
particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius,
                             plot_mesh=False, shapes_in_parallel=False, keep_mesh=True)
print(f"[info] generation done. shapes created = {len(particles)}")





# --- rotate about ORIGIN, then translate by absolute index 'ind' --- #
for i, part_group in enumerate(particles):
    ind = i  # meshes are 0..N-1; positions/rotations indexed by absolute mesh index
    p = part_group[0]
    R = detect_rotation_matrix(rotations[ind])
    pts = rotate_about_origin(p.pos, R)   # <— key difference vs previous: NO centroid recenter
    pts = pts + positions[ind]
    p.pos = pts

# --- global scale S --- #
for i in range(len(particles)):
    particles[i][0].pos = np.asarray(particles[i][0].pos) * S


# --- Add gravity/extforce ---
gvec = np.array([0.0, 0.0, float(args.gravity)], dtype=float)
for i in range(N):
    prt = particles[i][0]
    prt.acc += gvec
    prt.extforce += [0.0, 0.0, float(args.gravity) * prt.material.rho]



# --- tight bounding box AFTER scaling --- #
all_pts = np.concatenate([np.asarray(particles[i][0].pos) for i in range(len(particles))], axis=0)
lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
buff = contact_radius * 1.1
x_min, y_min, z_min = lo - buff
x_max, y_max, z_max = hi + buff

# --- wall & contact (your formulas) --- #
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)

print_container_info(wall)

normal_stiffness = material.cnot / contact_radius
damping_ratio = 0.8
friction_coefficient = 0.8
contact = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

# --- experiment --- #
exp = Experiment(particles, wall, contact)

# --- save --- #
os.makedirs(args.path, exist_ok=True)
setup_file = os.path.join(args.path, 'setup.h5')
print(f"[info] saving setup to {setup_file}")
exp.save(setup_file)

# --- plot --- #

if not args.noplot:
    try:
        png = os.path.join(args.path, 'setup.png')
        print("[info] rendering setup.png ...")
        # keep it simple; scatter only to avoid occlusion hiding one grain
        plot3d_setup(particles, dotsize=5, wall=wall, show_plot=False,
                     delta=delta, contact_radius=contact_radius,
                     save_filename=png, noscatter=False, show_particle_index=True,
                     trisurf=False)
        print("[info] saved", png)
    except Exception as e:
        print("[warn] plot3d_setup failed:", e)

print("[done]")
