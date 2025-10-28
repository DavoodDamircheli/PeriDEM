import numpy as np
import math
import sys, os
import argparse

# Project modules (must be available on PYTHONPATH or in CWD)
sys.path.append(os.getcwd())
import shape_dict, material_dict
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup
from shape_params import Param

EPS = 1e-15  # numeric safety margin

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
    if avail < -EPS:
        return 0

    # Single center case
    if avail < center_spacing - EPS:
        return 1 if L >= 2*(wall_gap + rad) - EPS else 0

    # Otherwise, 1 + floor(avail / center_spacing)
    return 1 + int(math.floor((avail + EPS) / center_spacing))


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


def feasible_edge_radius(L, wall_gap, surf_gap, r_prev, x_prev, r_min_allowed, r_max_allowed):
    """
    For placing an extra sphere at the +edge with center x_edge = L - (wall_gap + r_edge),
    feasibility vs previous sphere (x_prev, r_prev) requires:
        x_edge - x_prev >= r_prev + r_edge + surf_gap
    ->  r_edge <= (L - wall_gap - x_prev - surf_gap - r_prev)/2
    Returns (ok, r_edge, x_edge), allowing r_edge in [r_min_allowed, r_max_allowed].
    """
    rhs = 0.5*(L - wall_gap - x_prev - surf_gap - r_prev)
    if rhs < r_min_allowed - EPS:
        return (False, None, None)
    r_edge = min(r_max_allowed, max(r_min_allowed, rhs))
    x_edge = L - (wall_gap + r_edge)
    # Final safety clamp: ensure both wall clearance and neighbor gap hold
    if (x_edge - x_prev) + EPS < (r_prev + r_edge + surf_gap):
        return (False, None, None)
    if x_edge > L - (wall_gap + r_edge) + EPS:
        return (False, None, None)
    return (True, r_edge, x_edge)


# ----------------- greedy compact 1D packer for x-rows -----------------
def pack_line_x(L, wall_gap, surf_gap, r_min, r_max, start_big, r_floor):
    """
    Greedy 1D pack along [0,L] with wall clearance and surface gaps.
    Improvements:
      - First sphere radius is the LARGEST feasible (<= r_max if start_big else <= r_min but >= r_floor).
      - Fill with r_min next as far as they fit.
      - At the end, squeeze the LARGEST feasible radius in [r_floor, r_max].
    Returns:
      xs : list of x centers
      rs : list of radii (same length)
    """
    xs, rs = [], []

    # Choose first radius: try desired (r_max or r_min), reduce if needed to fit
    r0_target = r_max if start_big else r_min
    r0_upper  = r0_target
    r0_lower  = r_floor
    # Feasibility for a single sphere near left wall: center x0 = wall_gap + r0 must also satisfy x0 <= L - (wall_gap + r0)
    # i.e., 2*(wall_gap + r0) <= L => r0 <= L/2 - wall_gap
    r0_cap = max(0.0, 0.5*L - wall_gap)
    r0 = min(r0_upper, max(r0_lower, r0_cap))
    if r0 <= r_floor + EPS and r0_cap < r_floor - EPS:
        # nothing fits
        return xs, rs
    x0 = wall_gap + r0
    xs.append(float(x0))
    rs.append(float(r0))

    # Fill with r_min spheres as far as they fit (r_min >= r_floor by design; if r_min < r_floor, clamp)
    r_min_eff = max(r_min, r_floor)
    while True:
        r_prev = rs[-1]
        r_next = r_min_eff
        x_next = xs[-1] + (r_prev + r_next + surf_gap)
        if x_next > L - (wall_gap + r_next) + EPS:
            break
        xs.append(float(x_next))
        rs.append(float(r_next))

    # Try to squeeze the largest possible end sphere in [r_floor, r_max]
    if len(xs) > 0:
        ok, r_edge, x_edge = feasible_edge_radius(
            L, wall_gap, surf_gap, rs[-1], xs[-1], r_floor, r_max
        )
        if ok:
            xs.append(float(x_edge))
            rs.append(float(r_edge))

    # Final robust clamp: remove any sphere that violates boundaries or neighbor spacing (paranoid safeguard)
    keep_xs, keep_rs = [], []
    for i, (xc, rc) in enumerate(zip(xs, rs)):
        left_ok  = xc >= wall_gap + rc - EPS
        right_ok = xc <= L - (wall_gap + rc) + EPS
        neigh_ok = True
        if i > 0:
            xp, rp = xs[i-1], rs[i-1]
            neigh_ok = (xc - xp) + EPS >= (rp + rc + surf_gap)
        if left_ok and right_ok and neigh_ok:
            keep_xs.append(xc); keep_rs.append(rc)
    return keep_xs, keep_rs


def main():
    parser = argparse.ArgumentParser(description='Greedy compact packing of grains in a 3D box, with end squeezing.')
    # Container (default 4x4x5 mm)
    parser.add_argument('--Lx', type=float, default=2e-3, help='Container length in x [m]')
    parser.add_argument('--Ly', type=float, default=2e-3, help='Container length in y [m]')
    parser.add_argument('--Lz', type=float, default=2e-3, help='Container length in z [m]')

    # Radius range and gaps
    parser.add_argument('--r_min', type=float, default=420e-6, help='Minimum radius [m] (e.g., 0.42 mm)')
    parser.add_argument('--r_max', type=float, default=840e-6, help='Maximum radius [m] (e.g., 0.84 mm)')
    parser.add_argument('--r_floor_factor', type=float, default=0.05,
                        help='Smallest allowed radius as a fraction of r_min for edge squeezes (default 0.05*r_min)')
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
    #r_ref = r_min
    r_ref = 600e-6 
    mesh_ref=600e-4
    surf_gap = args.surf_gap_factor * r_ref
    wall_gap = args.wall_gap_factor * surf_gap

    delta          = mesh_ref / args.delta_factor
    meshsize       = mesh_ref / args.meshsize_factor
    contact_radius = mesh_ref / args.contact_rad_factor

    # ----------------- build rows (y,z packed with r_min) -----------------
    center_spacing_min = 2.0 * r_min + surf_gap
    ny = max_fit_along(Ly, wall_gap, center_spacing_min, r_min)
    nz = max_fit_along(Lz, wall_gap, center_spacing_min, r_min)
    if min(ny, nz) <= 0:
        raise ValueError("No (y,z) row fits with r_min—enlarge Ly/Lz or reduce gaps. "
                         f"(Ly={Ly:g}, Lz={Lz:g}, r_min={r_min:g}, surf_gap={surf_gap:g}, wall_gap={wall_gap:g})")

    ys = positions_along(ny, Ly, wall_gap, center_spacing_min, r_min)
    zs = positions_along(nz, Lz, wall_gap, center_spacing_min, r_min)

    # ------------- optional: add a Z cap layer under top wall, any radius down to r_floor -------------
    r_floor = max(args.r_floor_factor * r_min, 1e-9)
    if len(zs) > 0:
        z_last = zs[-1]
        r_last = r_min  # rows are built with r_min in z
        # compute the largest feasible z-edge radius
        rhs = 0.5*(Lz - wall_gap - z_last - surf_gap - r_last)
        if rhs >= r_floor - EPS:
            r_cap = min(r_max, max(r_floor, rhs))
            z_cap = Lz - (wall_gap + r_cap)
            # safety check with neighbor spacing and wall
            if (z_cap - z_last) + EPS >= (r_last + r_cap + surf_gap) and z_cap <= Lz - (wall_gap + r_cap) + EPS:
                zs = np.append(zs, z_cap)
                # Note: we remember per-row z radii later; here we only need centers. We'll assign radii per particle.

    all_centers = []
    all_radii   = []

    # We'll keep a parallel list of z radii per layer: base layers have r_min; cap (last, if added) has r_cap.
    z_radii_per_layer = [r_min]*len(zs)
    if len(zs) >= 2:
        # If a cap was added, it must be the last element (since we appended)
        # Detect if last gap < center_spacing_min implying it's a cap:
        if (len(zs) >= 2) and ((zs[-1] - zs[-2]) < (r_min + r_min + surf_gap) + 1e-12):
            # Estimate r_cap from wall gap
            r_est = Lz - wall_gap - zs[-1]
            z_radii_per_layer[-1] = r_est

    # Build rows
    for j, yj in enumerate(ys):
        for kz, zk in enumerate(zs):
            start_big = ((j + kz) % 2 == 0)  # alternate rows
            xs, rs = pack_line_x(Lx, wall_gap, surf_gap, r_min, r_max, start_big, r_floor)
            # record this row's particles (x varies; y=const; z=const)
            rz_layer = z_radii_per_layer[kz]
            for xi, ri_x in zip(xs, rs):
                # The actual sphere radius must respect the most restrictive of x-row choice (ri_x) and z-layer (rz_layer).
                ri = min(ri_x, rz_layer)
                # Final boundary safety: clamp center so it remains legal after taking min radius
                xi = min(max(xi, wall_gap + ri), Lx - (wall_gap + ri))
                yi = yj
                zi = min(max(zk, wall_gap + ri), Lz - (wall_gap + ri))
                all_centers.append([xi, yi, zi])
                all_radii.append(ri)

    shifts = np.asarray(all_centers, dtype=float)
    radii  = np.asarray(all_radii, dtype=float)
    cnt    = len(radii)

    print(f"[packer] Built {cnt} particles in a {ny}x{len(zs)} grid of rows (y*z={ny*len(zs)}).")
    print(f"[packer] r_min={r_min:g}, r_max={r_max:g}, r_floor={r_floor:g}, surf_gap={surf_gap:g}, wall_gap={wall_gap:g}")

    # ----------------- material, wall, shapes -----------------
    SL = ShapeList()
    material = material_dict.ottawa_sand(delta)
    material.print()

    wall = Wall3d(1, 0.0, 0.0, 0.0, Lx, Ly, Lz)

    # Build one Shape per particle, scaling each to its radius
    if cnt == 0:
        raise RuntimeError("No particles generated. Check dimensions/gaps.")

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


if __name__ == "__main__":
    main()
