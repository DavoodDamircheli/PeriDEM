import numpy as np
import math

# import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
# from genmesh import genmesh
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup, GridDist
from arrangements import get_incenter_mesh_loc

from shape_params import Param
import argparse
# Instantiate the parser
# import pdb
# pdb.set_trace()




# # ----------------- helpers -----------------
# def max_fit_along(L, gap, spacing):
#     """Max integer N so that N points in [gap, L-gap] with min spacing `spacing` fit."""
#     usable = L - 2*gap
#     if usable < 0:
#         return 0
#     if usable == 0:
#         return 1
#     # N points require (N-1)*spacing <= usable  -> N <= floor(usable/spacing) + 1
#     return int(usable // spacing) + 1
#
# def positions_along(n, L, gap):
#     """
#     Return n coordinates in [gap, L-gap].
#     - n == 1: place at center (clamped between gaps).
#     - n > 1: evenly spaced between gap and L-gap.
#     """
#     if n <= 0:
#         return np.array([], dtype=float)
#     lo, hi = gap, L - gap
#     if hi <= lo:
#         raise ValueError(f"Box length {L:g} too small for required gap {gap:g} on both sides.")
#     if n == 1:
#         c = 0.5 * L
#         return np.array([min(max(c, lo), hi)], dtype=float)
#     return np.linspace(lo, hi, n, dtype=float)
#
#-------------------------------------
# ----------------- helper functions -----------------
def max_fit_along(L, wall_gap, center_spacing, rad):
    """
    Max number of centers (equally spaced) that fit along length L
    with:
      - first center at wall_gap + rad
      - last  center at L - (wall_gap + rad)
      - constant step = center_spacing (>= 2*rad)
    """
    if L <= 0 or center_spacing <= 0:
        return 0

    # available span for center locations (from first to last center position)
    avail = L - 2*(wall_gap + rad)

    # If even a single grain doesn't fit with wall clearance, return 0
    if avail < -1e-15:
        return 0

    # If we can place exactly one center, that's valid if L >= 2*(wall_gap + rad)
    if avail < center_spacing - 1e-15:
        # one center fits if there is room for a center anywhere between the walls
        return 1 if L >= 2*(wall_gap + rad) - 1e-15 else 0

    # Otherwise, 1 + floor(avail / center_spacing)
    return 1 + int(math.floor((avail + 1e-15) / center_spacing))


def positions_along(n, L, wall_gap, center_spacing, rad):
    """
    Compute n center coordinates along [0, L] given:
      - first at wall_gap + rad
      - step center_spacing
    Assumes n is feasible (i.e., computed by max_fit_along or capacity check).
    """
    if n <= 0:
        return np.array([], dtype=float)
    x0 = wall_gap + rad
    xs = x0 + np.arange(n, dtype=float) * center_spacing
    # Safety clamp (due to float eps) so last center never violates wall clearance
    xmax = L - (wall_gap + rad)
    if len(xs) > 0:
        xs[-1] = min(xs[-1], xmax)
    return xs


parser = argparse.ArgumentParser(description='Optional app description')

# Optional argument
# parser.add_argument('--shape', type=str, help='shape of each particle', default='small_disk')
parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--L', type=float, help='half-length of container', default=100e-3)
parser.add_argument('--wallh_ratio', type=float, help='half-length of wall height vs wall width', default=2)
parser.add_argument('--nx', type=int, help='number of particles in x dir', default=10)
parser.add_argument('--ny', type=int, help='number of particles in y dir', default=10)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=-20)
parser.add_argument('--acc_val', type=float, help='initial velocity', default=-10)
parser.add_argument('--shape', type=str, help='shape of grain', default='grains')
#parser.add_argument('--msh_path', type=str, help='shape of grain', default='/home/davood/projects/beta_perigrain_v2/grain-data/test5grains')
parser.add_argument('--msh_path', type=str, help='shape of grain', default='/home/davood/projects/beta_perigrain_v2/grain-data')

parser.add_argument('--G_scale', type=float, help='shear modulus scaling', default=0.5)
parser.add_argument('--Gnot_scale', type=float, help='shear modulus scaling', default=1e-4)
parser.add_argument('--K_scale', type=float, help='bulk modulus scaling', default=0.5)
parser.add_argument('--rho_scale', type=float, help='density scaling', default=1)

parser.add_argument('--meshsize_factor', type=float, help='meshsize factor compared to radius', default=50)
parser.add_argument('--delta_factor', type=float, help='delta factor compared to radius', default=3)
parser.add_argument('--contact_rad_factor', type=float, help='contact radius factor compared to radius', default=5)

parser.add_argument('--setup_file', type=str, help='output setup directory', default='data/hdf5/all.h5')
parser.add_argument('--plot', action='store_true', help='whether to show plot or not')
# finish parsing
args = parser.parse_args()
args.plot=1
print('plot', args.plot)
print('saving experiment setup to', args.setup_file)

""" Two particles colliding in 3D
"""

# Common knobs
mf = 20  # meshsize factor (your existing code)
delta_factor      = getattr(args, "delta_factor", 3.0)
contact_rad_factor = getattr(args, "contact_rad_factor", 3.0)

# # --- sizes & spacing (unchanged parts above) ---
rad = 600e-6                    # meters (sphere radius)
delta = rad/args.delta_factor
meshsize = rad/mf
contact_radius = rad/args.contact_rad_factor
# gap = rad + contact_radius                 # margin to each wall and half the center spacing
# min_center_spacing = 2.0 * (rad + contact_radius)  # no-overlap spacing
# import numpy as np
#
# ----------------- user-fixed box -----------------
Lx = 4e-3   # meters
Ly = 4e-3
Lz = 5e-3

# ----------------- material / peridynamics params -----------------
# Choose one:
# mode = "counts"  # you set n_x, n_y, n_z
mode = "radius"     # you set rad; will auto-compute n_x, n_y, n_z

# === Mode 1: specify counts and radius ===
# rad = 600e-6; n_x, n_y, n_z = 2, 2, 4

# ----------------- geometry knobs -----------------
mode = "radius"                 # "counts" or "radius"
surf_gap = 1.0 * contact_radius          # <-- KNOB: surface-to-surface clearance between neighbors


wall_gap = 1.0 * surf_gap            # <-- KNOB: clearance between wall and grain *surface*

# Container (set these elsewhere if you already have them)
# Lx, Ly, Lz = 4e-3, 4e-3, 4e-3   # example: 4 mm each

# ----------------- derive spacings -----------------
center_spacing = 1.0*rad + surf_gap    # REQUIRED center-to-center distance

# ----------------- choose counts -----------------
if mode == "counts":
    # YOU set these:
    n_x, n_y, n_z = 2, 2, 4  # example

    # capacity checks
    cap_x = max_fit_along(Lx, wall_gap, center_spacing, rad)
    cap_y = max_fit_along(Ly, wall_gap, center_spacing, rad)
    cap_z = max_fit_along(Lz, wall_gap, center_spacing, rad)

    if (n_x > cap_x) or (n_y > cap_y) or (n_z > cap_z):
        raise ValueError(
            "Requested grid does not fit without overlaps.\n"
            f"  Requested: n_x={n_x}, n_y={n_y}, n_z={n_z}\n"
            f"  Max that fit: n_x≤{cap_x}, n_y≤{cap_y}, n_z≤{cap_z}\n"
            "  Adjust Lx,Ly,Lz or n_* or rad/surf_gap/wall_gap."
        )

elif mode == "radius":
    # auto-compute max counts that fit
    n_x = max_fit_along(Lx, wall_gap, center_spacing, rad)
    n_y = max_fit_along(Ly, wall_gap, center_spacing, rad)
    n_z = max_fit_along(Lz, wall_gap, center_spacing, rad)
    if min(n_x, n_y, n_z) <= 0:
        raise ValueError(
            "No particle fits with the given radius/gaps.\n"
            f"Lx={Lx:g}, Ly={Ly:g}, Lz={Lz:g}, center_spacing={center_spacing:g}, "
            f"wall_gap={wall_gap:g}, rad={rad:g}"
        )
else:
    raise ValueError("mode must be 'counts' or 'radius'.")

# ----------------- build positions -----------------
x = positions_along(n_x, Lx, wall_gap, center_spacing, rad)
y = positions_along(n_y, Ly, wall_gap, center_spacing, rad)
z = positions_along(n_z, Lz, wall_gap, center_spacing, rad)

X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
shifts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
cnt = shifts.shape[0]

print(f"Placed {cnt} grains: n_x={n_x}, n_y={n_y}, n_z={n_z}")
print(f"center_spacing = {center_spacing:g} (2*rad + surf_gap)")
print(f"first center (x): {x[0]:.6g}, last center (x): {x[-1]:.6g}  | domain [0,{Lx:g}]")

# ----------------- peridem material / mesh (unchanged) -----------------
delta = rad / delta_factor
meshsize = rad / mf

SL = ShapeList()
material = material_dict.peridem_3d(delta)
material.print()

# Walls: [0,Lx]×[0,Ly]×[0,Lz]
wall = Wall3d(1, 0.0, 0.0, 0.0, Lx, Ly, Lz)


print(f"Mode: {mode}")
print(f"Placed {cnt} grains: n_x={n_x}, n_y={n_y}, n_z={n_z}")
# print(f"min center spacing ≥ {min_center_spacing:.4e}, gap={gap:.4e}")
# print(f"x in [{x.min():.4e}, {x.max():.4e}] within [0, {Lx:.4e}]")
# print(f"y in [{y.min():.4e}, {y.max():.4e}] within [0, {Ly:.4e}]")
# print(f"z in [{z.min():.4e}, {z.max():.4e}] within [0, {Lz:.4e}]")
#











#------------------max count is 400--------------
count = n_x*n_y*n_z
print("total count is ",count)
if args.shape == 'grains':
    for i in range(1,count+1):
        print(i)
        ind = int(i)
        msh_file = args.msh_path + '/mesh_' + str(ind) + '.msh'
        shape = shape_dict.Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file, scale_mesh_to=rad, centroid_origin=True)
        SL.append(shape=shape, count=1, meshsize=meshsize, material=material,plot_shape=False)
else:
    SL.append(shape=shape, count=count, meshsize=meshsize, material=material,plot_shape=False)



particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius, plot_node_text=False, plot_shape=False, plot_mesh=False)

a1 = list(range(len(particles)))
a2= len(particles)
print('a1 is', a1) 
print('a2 is', a2) 

for i in range(count):
    # particles[0][i].scale(rad/1e-3)
    if args.shape == 'grains':
        part = particles[i][0]
    else:
        part = particles[0][i]
    part.shift(shifts[i])

normal_stiffness = material_dict.peridem_3d(contact_radius).cnot / contact_radius


damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)
args.plot=1
if args.plot:
    plot3d_setup(particles, dotsize=15, wall=wall, show_particle_index=True, delta=delta, contact_radius=contact_radius, save_filename='setup.png')

exp = Experiment(particles, wall, contact)

#######################################################################

# save the data
print('saving experiment setup to', args.setup_file)
exp.save(args.setup_file)
