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


# ---------- rotation utilities ---------- #
def euler_to_matrix(rx, ry, rz):
    """Return rotation matrix from XYZ Euler angles (radians)."""
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
    """Convert quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])


def detect_rotation_matrix(rot_vec):
    """Detect rotation representation and return 3×3 matrix."""
    rot_vec = np.asarray(rot_vec).flatten()
    if rot_vec.size == 3:
        rx, ry, rz = rot_vec
        return euler_to_matrix(rx, ry, rz)
    elif rot_vec.size == 4:
        return rotation_matrix_from_quaternion(rot_vec)
    elif rot_vec.size == 9:
        return rot_vec.reshape(3, 3)
    else:
        raise ValueError(f"Unsupported rotation vector length {rot_vec.size}")


def rotate_points(P, R):
    """Rotate Nx3 points P by 3×3 matrix R about centroid."""
    C = P.mean(axis=0)
    return (P - C) @ R.T + C


# ---------- argument parser ---------- #
parser = argparse.ArgumentParser(description='Setup generator (no Open3D)')
parser.add_argument('--N', type=int, help='number of particles to include (0..N-1)', default=2)

parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=5e2)
parser.add_argument('--acc_val', type=float, help='initial acceleration', default=-10)

parser.add_argument('--G_scale', type=float, help='shear modulus scaling', default=0.5)
parser.add_argument('--Gnot_scale', type=float, help='shear modulus scaling', default=1e-4)
parser.add_argument('--K_scale', type=float, help='bulk modulus scaling', default=0.5)
parser.add_argument('--rho_scale', type=float, help='density scaling', default=1)

parser.add_argument('--meshsize_factor', type=float, help='meshsize factor compared to radius', default=4)
parser.add_argument('--delta_factor', type=float, help='delta factor compared to radius', default=3)
parser.add_argument('--contact_rad_factor', type=float, help='contact radius factor compared to radius', default=4)

parser.add_argument('--datapath', type=str, help='path to input positions/rotations', default='/home/davood/projects/ls-shapes')
parser.add_argument('--path', type=str, help='output directory', default='examples_output/')
parser.add_argument('--noplot', action='store_true', help='disable plot')
parser.add_argument('--novis', action='store_true', help='disable visualization')
args = parser.parse_args()

args.N = 2
ind = args.N
# ---------- load positions & rotations ---------- #
positions = np.loadtxt(os.path.join(args.datapath, 'Input', 'positions_image.dat'))
rotations = np.loadtxt(os.path.join(args.datapath, 'Input', 'rotations_image.dat'))

num_particles = int(args.N)
radius = args.particle_rad
meshsize = radius / args.meshsize_factor
delta = radius / args.delta_factor
contact_radius = radius / args.contact_rad_factor

# ---------- material ---------- #
material = material_dict.ottawa_sand(delta)

# ---------- shape list ---------- #
SL = ShapeList()
for ind in range(num_particles):
    msh_file = f'/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/grain-mesh-meshsize-10-voxel-7/mesh_{ind}.msh'
    if not os.path.exists(msh_file):
        print(f"[warn] Missing mesh file {msh_file}, skipping")
        continue
    sh = Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)
    SL.append(shape=sh, count=1, meshsize=meshsize, material=material)


    particles = SL.generate_mesh(
        dimension=3,
        contact_radius=contact_radius,
        plot_mesh=False,
        plot_node_text=False,
        shapes_in_parallel=False,
        keep_mesh=True
    )
print("Time taken to generate all shapes done.")

# ---------- apply rotations & translations ---------- #
for i, part_group in enumerate(particles):
    if i >= len(positions) or i >= len(rotations):
        print(f"[warn] No rotation/position for particle {i}, skipping transform")
        continue
    p = part_group[0]
    R = detect_rotation_matrix(rotations[i])
    p.pos = rotate_points(np.asarray(p.pos), R)
    p.pos += positions[i]

# ---------- wall & experiment ---------- #
# (simplify to a tall cylinder around all particles)
S = 1e-3  
S = 1  
ind_range = range(args.N)
# 1) Scale particle coordinates
for i, ind in enumerate(ind_range):
    p = particles[i][0]
    p.pos = np.asarray(p.pos) * S   # ensure NumPy, not O3D Vector
# 2) Scale all other length-like params
contact_radius *= S
delta *= S

# 3) Recompute tight bounding box *after* scaling
x_ini, y_ini, z_ini = particles[0][0].pos[0]
x_min = x_max = x_ini
y_min = y_max = y_ini
z_min = z_max = z_ini

for i, ind in enumerate(ind_range):
    P = particles[i][0].pos
    x_min = min(x_min, np.min(P[:,0])); x_max = max(x_max, np.max(P[:,0]))
    y_min = min(y_min, np.min(P[:,1])); y_max = max(y_max, np.max(P[:,1]))
    z_min = min(z_min, np.min(P[:,2])); z_max = max(z_max, np.max(P[:,2]))

buff = contact_radius * 1.1
x_min -= buff; x_max += buff
y_min -= buff; y_max += buff
z_min -= buff; z_max += buff

wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
#######################################################################


# contact properties
normal_stiffness = material.cnot / contact_radius
damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

#######################################################################
exp = Experiment(particles, wall, contact)

# ---------- save setup ---------- #
os.makedirs(args.path, exist_ok=True)
setup_file = os.path.join(args.path, 'setup.h5')
print(f"Saving setup to {setup_file}")
exp.save(setup_file)
args.noplot=0
# ---------- optional plot ---------- #
if not args.noplot:
    try:
        print("Rendering setup.png ...")
        plot3d_setup(
            particles,
            dotsize=10,
            wall=wall,
            show_plot=False,
            delta=delta,
            contact_radius=contact_radius,
            save_filename=os.path.join(args.path, 'setup.png')
        )
        print("Saved setup.png")
    except Exception as e:
        print("[warn] plot3d_setup failed:", e)

print("Done.")
