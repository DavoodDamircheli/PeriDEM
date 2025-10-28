import numpy as np
import math
import sys, os
import argparse

# Project modules (must be available on PYTHONPATH or in CWD)
sys.path.append(os.getcwd())
import shape_dict, material_dict
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup
from shape_params import Param

# ----------------- helpers: feasibility along an axis -----------------
def max_fit_along(L, wall_gap, center_spacing, rad):
    """
    Max number of centers (equally spaced) that fit along length L with:
      - first center at wall_gap + rad
      - last  center at L - (wall_gap + rad)
      - constant step = center_spacing (>= 2*rad)
    """
    if L <= 0 or center_spacing <= 0:
        return 0

    avail = L - 2*(wall_gap + rad)

    # If even a single grain doesn't fit with wall clearance, return 0
    if avail < -1e-15:
        return 0

    # Single center case
    if avail < center_spacing - 1e-15:
        return 1 if L >= 2*(wall_gap + rad) - 1e-15 else 0

    # Otherwise, 1 + floor(avail / center_spacing)
    return 1 + int(math.floor((avail + 1e-15) / center_spacing))


def positions_along(n, L, wall_gap, center_spacing, rad):
    """
    Compute n center coordinates along [0, L] given:
      - first at wall_gap + rad
      - step center_spacing
    Assumes n was computed by max_fit_along (feasible).
    """
    if n <= 0:
        return np.array([], dtype=float)
    x0 = wall_gap + rad
    xs = x0 + np.arange(n, dtype=float) * center_spacing
    # Clamp final center to keep wall clearance (float safety)
    xmax = L - (wall_gap + rad)
    xs[-1] = min(xs[-1], xmax)
    return xs


# ----------------- NEW: greedy compact 1D packer for x-rows -----------------
def pack_line_x(L, wall_gap, surf_gap, r_min, r_max, start_big):
    """
    Greedy 1D pack along [0,L] with wall clearance and surface gaps.
    Rules:
      - If start_big:   place r_max first, then fill with r_min; at end try an r_max if it fits.
      - If not start_big: place r_min first, then r_min; at end try an r_max if it fits.
    Returns:
      xs : list of x centers
      rs : list of radii (same length)
    """
    xs, rs = [], []

    # Choose first radius
    r0 = r_max if start_big else r_min

    # Place first sphere at left with wall clearance
    x0 = wall_gap + r0
    if x0 > L - (wall_gap + r0) + 1e-15:
        return xs, rs  # nothing fits
    xs.append(float(x0))
    rs.append(float(r0))

    # Fill with r_min spheres as far as they fit
    while True:
        r_prev = rs[-1]
        r_next = r_min
        x_next = xs[-1] + (r_prev + r_next + surf_gap)
        # right wall clearance for next
        if x_next > L - (wall_gap + r_next) + 1e-15:
            break
        xs.append(float(x_next))
        rs.append(float(r_next))

    # Try to squeeze a big one at the end if possible
    if len(xs) > 0:
        x_req = xs[-1] + (rs[-1] + r_max + surf_gap)
        x_max = L - (wall_gap + r_max)
        if x_req <= x_max + 1e-15:
            xs.append(float(x_req))
            rs.append(float(r_max))

    return xs, rs


parser = argparse.ArgumentParser(description='Greedy compact packing of grains in a 3D box.')
# Container (default 4x4x5 mm)
parser.add_argument('--Lx', type=float, default=4e-3, help='Container length in x [m]')
parser.add_argument('--Ly', type=float, default=4e-3, help='Container length in y [m]')
parser.add_argument('--Lz', type=float, default=5e-3, help='Container length in z [m]')

# Radius range and gaps
parser.add_argument('--r_min', type=float, default=420e-6, help='Minimum radius [m] (e.g., 0.42 mm)')
parser.add_argument('--r_max', type=float, default=840e-6, help='Maximum radius [m] (e.g., 0.84 mm)')
parser.add_argument('--surf_gap_factor', type=float, default=0.01,
                    help='Surface gap as a fraction of r_ref (default 0.01*r_ref)')
parser.add_argument('--wall_gap_factor', type=float, default=0.5,
                    help='Wall clearance as a fraction of surf_gap (default 0.5*surf_gap)')

# Peridynamics / numerics
parser.add_argument('--delta_factor', type=float, default=3.0, help='delta = r_ref / delta_factor')
parser.add_argument('--meshsize_factor', type=float, default=20.0, help='meshsize = r_ref / meshsize_factor')
parser.add_argument('--contact_rad_factor', type=float, default=5.0, help='contact radius = r_ref / contact_rad_factor')

# Grain shape controls
parser.add_argument('--shape', type=str, default='grains', help='grain mode (grains uses mesh library)')
parser.add_argument('--msh_path', type=str, default='/home/davood/projects/beta_perigrain_v2/grain-data',
                    help='Directory of mesh_#.msh files')

# Physics
parser.add_argument('--gravity', type=float, default=-5e4, help='Gravitational acceleration in z [m/s^2] (project convention)')
parser.add_argument('--damping_ratio', type=float, default=0.8, help='Contact damping ratio')
parser.add_argument('--friction_coeff', type=float, default=0.8, help='Contact friction coefficient')

# IO
parser.add_argument('--setup_file', type=str, default='data/hdf5/all.h5', help='Output HDF5 path')
parser.add_argument('--plot', action='store_true', help='Save setup.png rendering')

args = parser.parse_args()

# ----------------- sizes -----------------
Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
r_min, r_max = args.r_min, args.r_max

# Reference radius for numerical scales (conservative choice)
r_ref = r_min

surf_gap = args.surf_gap_factor * r_ref
wall_gap = args.wall_gap_factor * surf_gap

delta          = r_ref / args.delta_factor
meshsize       = r_ref / args.meshsize_factor
contact_radius = r_ref / args.contact_rad_factor

# ----------------- build rows (y,z packed with r_min) -----------------
center_spacing_min = 2.0 * r_min + surf_gap
ny = max_fit_along(Ly, wall_gap, center_spacing_min, r_min)
nz = max_fit_along(Lz, wall_gap, center_spacing_min, r_min)
if min(ny, nz) <= 0:
    raise ValueError("No (y,z) row fits with r_min—enlarge Ly/Lz or reduce gaps. "
                     f"(Ly={Ly:g}, Lz={Lz:g}, r_min={r_min:g}, surf_gap={surf_gap:g}, wall_gap={wall_gap:g})")

ys = positions_along(ny, Ly, wall_gap, center_spacing_min, r_min)
zs = positions_along(nz, Lz, wall_gap, center_spacing_min, r_min)

all_centers = []
all_radii   = []

for j, yj in enumerate(ys):
    for k, zk in enumerate(zs):
        start_big = ((j + k) % 2 == 0)  # alternate rows
        xs, rs = pack_line_x(Lx, wall_gap, surf_gap, r_min, r_max, start_big)
        for xi, ri in zip(xs, rs):
            all_centers.append([xi, yj, zk])
            all_radii.append(ri)

shifts = np.asarray(all_centers, dtype=float)
radii  = np.asarray(all_radii, dtype=float)
cnt    = len(radii)

print(f"[packer] Built {cnt} particles in a {ny}x{nz} grid of rows (y*z={ny*nz}).")
print(f"[packer] r_min={r_min:g}, r_max={r_max:g}, surf_gap={surf_gap:g}, wall_gap={wall_gap:g}")

# ----------------- material, wall, shapes -----------------
SL = ShapeList()
material = material_dict.ottawa_sand(delta)
material.print()

wall = Wall3d(1, 0.0, 0.0, 0.0, Lx, Ly, Lz)

# Build one Shape per particle, scaling each to its radius
if args.shape == 'grains':
    for i in range(cnt):
        ind = (i % 400) + 1   # reuse your library up to 400 variants; wrap if needed
        msh_file = os.path.join(args.msh_path, f"mesh_{ind}.msh")
        shape = shape_dict.Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file,
                                 scale_mesh_to=float(radii[i]), centroid_origin=True)
        SL.append(shape=shape, count=1, meshsize=meshsize, material=material, plot_shape=False)
else:
    # Fallback: single primitive repeated with per-particle scaling (if supported)
    shape = shape_dict.Shape(P=Param.sphere(r_ref), nonconvex_interceptor=None, msh_file=None,
                             scale_mesh_to=float(r_ref), centroid_origin=True)
    SL.append(shape=shape, count=cnt, meshsize=meshsize, material=material, plot_shape=False)

particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius,
                             plot_node_text=False, plot_shape=False, plot_mesh=False)

# ----------------- gravity / external forces -----------------
gvec = np.array([0.0, 0.0, float(args.gravity)], dtype=float)

for i in range(cnt):
    part = particles[i][0] if args.shape == 'grains' else particles[0][i]
    # shift to the packed center location
    part.shift(shifts[i])

    if not hasattr(part, "acc") or part.acc is None:
        part.acc = np.zeros(3, dtype=float)

    part.acc += gvec
    # Project convention used previously: extforce = g * rho (units per your engine)
    part.extforce += [0, 0, float(args.gravity) * part.material.rho]

# ----------------- contact model -----------------
normal_stiffness = material_dict.peridem_3d(contact_radius).cnot / contact_radius
contact  = Contact(contact_radius, normal_stiffness, float(args.damping_ratio), float(args.friction_coeff))
args.plot=1
# ----------------- plot & save -----------------
if args.plot:
    print("Rendering setup.png ...")
    plot3d_setup(particles, dotsize=15, wall=wall, show_particle_index=True,
                 delta=delta, contact_radius=contact_radius, save_filename='setup.png')

exp = Experiment(particles, wall, contact)
os.makedirs(os.path.dirname(args.setup_file), exist_ok=True)
print('Saving experiment setup to', args.setup_file)
exp.save(args.setup_file)


