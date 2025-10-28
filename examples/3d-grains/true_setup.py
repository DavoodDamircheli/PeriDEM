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





# --- variable-radius "squeeze" helpers ---

def try_add_edge_axis(L, wall_gap, surf_gap, r_prev, x_last, r_min, r_max):
    """
    Try to place ONE extra center on the + side with a (possibly) smaller radius r_edge in [r_min, r_max].
    We place that extra center at x_edge = L - (wall_gap + r_edge).
    Feasibility condition vs last existing center:
        x_edge - x_last >= r_prev + r_edge + surf_gap
    which gives:
        r_edge <= (L - wall_gap - x_last - surf_gap - r_prev)/2
    Returns (ok, r_edge, x_edge)
    """
    rhs = (L - wall_gap - x_last - surf_gap - r_prev) * 0.5
    if rhs < r_min:  # even the smallest allowed grain wouldn't fit
        return (False, None, None)
    r_edge = min(r_max, max(r_min, rhs))
    x_edge = L - (wall_gap + r_edge)
    return (True, r_edge, x_edge)


def build_axis_positions_with_edges(L, wall_gap, surf_gap, r_base, n_base, center_spacing):
    """
    Returns:
      xs   : list of center positions along this axis
      rcol : list of radii (same length as xs), possibly with a smaller last element
    """
    # base (uniform) positions, all radius=r_base
    xs = positions_along(n_base, L, wall_gap, center_spacing, r_base).tolist()
    rcol = [r_base] * len(xs)

    # Consider one extra at the +side using smaller radius if it fits
    r_min = 420e-6  # 0.42 mm
    r_max = 840e-6  # 0.84 mm

    if len(xs) > 0:
        x_last = xs[-1]
        r_prev = rcol[-1]
        ok, r_edge, x_edge = try_add_edge_axis(L, wall_gap, surf_gap, r_prev, x_last, r_min, r_max)
        if ok:
            xs.append(x_edge)
            rcol.append(r_edge)
    return xs, rcol


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

# ----------------- geometry knobs -----------------
mode = "radius"                                   # "counts" or "radius"
surf_gap = .02 * rad                   # surface-to-surface clearance between neighboring grains
wall_gap = 1.0 * surf_gap                         # clearance from wall surface to grain surface

# ----------------- derive spacings -----------------
center_spacing = 2.0*rad + surf_gap               # <<<< FIXED: must be 2*rad + surf_gap

# ----------------- choose counts for the *base* grid -----------------
if mode == "counts":
    n_x, n_y, n_z = 2, 2, 4  # example
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




# ----------------- build positions (base grid + optional edge squeeze) -----------------
x_list, r_x = build_axis_positions_with_edges(Lx, wall_gap, surf_gap, rad, n_x, center_spacing)
y_list, r_y = build_axis_positions_with_edges(Ly, wall_gap, surf_gap, rad, n_y, center_spacing)
z_list, r_z = build_axis_positions_with_edges(Lz, wall_gap, surf_gap, rad, n_z, center_spacing)

x = np.array(x_list, dtype=float)
y = np.array(y_list, dtype=float)
z = np.array(z_list, dtype=float)

# 3-D grid of centers
X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
shifts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

# Per-axis radii to 3-D: take the *most restrictive* (smallest) to ensure clearances
RX, RY, RZ = np.meshgrid(np.array(r_x), np.array(r_y), np.array(r_z), indexing="xy")
radii = np.minimum(np.minimum(RX, RY), RZ).ravel()

cnt = shifts.shape[0]
print(f"Placed {cnt} grains: "
      f"n_x(base)={n_x} (+{len(r_x)-n_x}), n_y(base)={n_y} (+{len(r_y)-n_y}), n_z(base)={n_z} (+{len(r_z)-n_z})")
print(f"center_spacing = {center_spacing:g} (2*rad + surf_gap)")
print(f"x in [{x.min():.6g}, {x.max():.6g}] within [0,{Lx:g}]")




# ----------------- peridem material / mesh (unchanged) -----------------
delta = rad / delta_factor
meshsize = rad / mf

SL = ShapeList()
material = material_dict.ottawa_sand(delta)
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
#------------------ build ShapeList with per-particle scaling --------------
count = cnt
print("total count is ", count)

if args.shape == 'grains':
    # pre-append one Shape per particle, but scale each to its own radius
    for i in range(count):
        ind = (i % 400) + 1   # reuse your library up to 400 variants; wrap if needed
        msh_file = os.path.join(args.msh_path, f"mesh_{ind}.msh")
        # scale to the *individual* radius
        shape = shape_dict.Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file,
                                 scale_mesh_to=float(radii[i]), centroid_origin=True)
        SL.append(shape=shape, count=1, meshsize=meshsize, material=material, plot_shape=False)
else:
    # single primitive shape repeated with per-particle scaling (if your ShapeList supports it)
    SL.append(shape=shape, count=count, meshsize=meshsize, material=material, plot_shape=False)

particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius,
                             plot_node_text=False, plot_shape=False, plot_mesh=False)

gval = -5e4
gvec = np.array([0.0, 0.0, gval], dtype=float)

# shift them into place
for i in range(count):
    part = particles[i][0] if args.shape == 'grains' else particles[0][i]
    part.shift(shifts[i])
    # ensure .acc exists and is a vector
    if not hasattr(part, "acc") or part.acc is None:
          part.acc = np.zeros(3, dtype=float)

    part.acc += gvec

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
