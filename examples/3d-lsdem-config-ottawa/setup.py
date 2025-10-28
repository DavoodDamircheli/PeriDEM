#!/usr/bin/env python3
import os, sys, argparse
import numpy as np

sys.path.append(os.getcwd())
import shape_dict, material_dict
from shape_dict import Shape
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup
import open3d as o3d


def mm_to_m(x_mm: float) -> float:
    return 1e-3 * float(x_mm)


def load_quaternions(path):
    if not os.path.exists(path):
        return np.array([[0,0,0,1]], dtype=float)
    Q = np.loadtxt(path)
    if Q.ndim == 1:
        Q = Q[None, :]
    if Q.shape[1] != 4:
        raise ValueError(f"Expected rotations as Nx4 quaternions; got shape {Q.shape}")
    return Q


def quat_to_R(q):
    q1, q2, q3, q4 = [float(x) for x in q]
    return np.array([
        [-q1**2 + q2**2 - q3**2 + q4**2, -2*(q1*q2 - q3*q4),        2*(q2*q3 + q1*q4)],
        [-2*(q1*q2 + q3*q4),              q1**2 - q2**2 - q3**2 + q4**2, -2*(q1*q3 - q2*q4)],
        [ 2*(q2*q3 - q1*q4),             -2*(q1*q3 + q2*q4),       -q1**2 - q2**2 + q3**2 + q4**2],
    ], dtype=float)


def apply_rotation_about_centroid(prt, R):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(prt.pos, dtype=np.float64))
    c = pcd.get_center()
    pcd.translate(-c)
    pcd.rotate(R)
    pcd.translate(c)
    prt.pos = np.asarray(pcd.points, dtype=np.float64)


def compute_geometry_bbox(particles):
    mn = np.array([+np.inf, +np.inf, +np.inf], dtype=float)
    mx = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for row in particles:
        prt = row[0]
        P = np.asarray(prt.pos, dtype=np.float64)
        mn = np.minimum(mn, P.min(axis=0))
        mx = np.maximum(mx, P.max(axis=0))
    return mn, mx


def rigid_scale_translate_geometry(particles, scale, center_from, center_to):
    for row in particles:
        prt = row[0]
        prt.shift(-center_from)
        prt.scale(scale)
        prt.shift(center_to)


def pairwise_nn_mean(X):
    N = X.shape[0]
    Dmin = []
    for i in range(N):
        d = np.linalg.norm(X[i] - X, axis=1)
        d[i] = np.inf
        Dmin.append(d.min())
    return float(np.mean(Dmin))


def main():
    ap = argparse.ArgumentParser(description="Pack image-arranged particles into a target box with shared orientation; bottom/corner pin; auto center-spacing.")
    # I/O
    ap.add_argument("--datapath", type=str, default="/home/davood/projects/ls-shapes", help="Path with Input/positions_image.dat (and rotations_image.dat optional).")
    ap.add_argument("--mesh_dir", type=str, default="/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/grain-mesh-meshsize-10-voxel-7", help="Directory containing mesh_{i}.msh files.")
    ap.add_argument("--path", type=str, required=True, help="Output directory ($dir). setup.h5 and setup.png saved here.")
    ap.add_argument("--N", type=int, default=200, help="Number of particles to use.")
    ap.add_argument("--indices_start", type=int, default=0, help="Start index into positions/rotations files.")
    # Target box in mm (numeric args are mm)
    ap.add_argument("--nx", type=float, default=4.0, help="mm")
    ap.add_argument("--ny", type=float, default=4.0, help="mm")
    ap.add_argument("--nz", type=float, default=5.0, help="mm")
    ap.add_argument("--padding_mm", type=float, default=0.0, help="Gap to walls on each side (mm).")
    ap.add_argument("--scale_tweak", type=float, default=0.995, help="Safety factor (<1) to avoid touching walls after squeeze.")
    ap.add_argument("--pin_to", type=str, choices=["center","corner","bottom"], default="bottom",
                    help="Placement: center (box center), corner (xmin,ymin,zmin = padding), bottom (z=min=padding, x/y centered).")
    # Orientation / positions
    ap.add_argument("--positions_in_mm", action="store_true", help="Interpret positions_image.dat as mm (converted to m).")
    ap.add_argument("--use_input_rotations", action="store_true", help="Use per-particle rotations from rotations_image.dat.")
    ap.add_argument("--shared_quaternion", type=float, nargs=4, metavar=("q1","q2","q3","q4"), default=[0,0,0,1],
                    help="Shared orientation quaternion when not using per-particle rotations.")
    # Center-spacing control
    ap.add_argument("--centers_scale_mode", type=str, choices=["auto","fixed","none"], default="auto",
                    help="Auto: scale centers so mean NN ≈ beta*avg_diameter. Fixed: multiply by centers_scale_factor. None: use raw centers.")
    ap.add_argument("--centers_scale_factor", type=float, default=1.0, help="Used if centers_scale_mode=fixed.")
    ap.add_argument("--beta_spacing", type=float, default=1.01, help="Target mean NN distance as beta*avg_diameter (auto mode).")
    # Mechanics / contact
    ap.add_argument("--delta_factor", type=float, default=3.0)
    ap.add_argument("--contact_rad_factor", type=float, default=4.0)
    ap.add_argument("--gravity", type=float, default=-1e4)
    ap.add_argument("--damping_ratio", type=float, default=0.8)
    ap.add_argument("--friction_coefficient", type=float, default=0.8)
    # Plot/vis
    ap.add_argument("--noplot", action="store_true")
    ap.add_argument("--novis", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.path, exist_ok=True)

    # Load positions & rotations
    pos_path = os.path.join(args.datapath, "Input", "positions_image.dat")
    rot_path = os.path.join(args.datapath, "Input", "rotations_image.dat")
    positions = np.loadtxt(pos_path)
    if positions.ndim == 1:
        positions = positions[None, :]
    if positions.shape[1] != 3:
        raise ValueError("positions_image.dat must be Nx3")
    if args.positions_in_mm:
        positions = positions * 1e-3  # mm -> m

    rotations = load_quaternions(rot_path)

    s0, N = int(args.indices_start), int(args.N)
    idx = np.arange(s0, s0+N, dtype=int)
    if idx[-1] >= positions.shape[0]:
        raise IndexError("Requested indices exceed positions length.")

    # Build particles
    SL = ShapeList()
    rad = 40.0  # representative
    delta = rad / args.delta_factor
    contact_radius = rad / args.contact_rad_factor
    material = material_dict.ottawa_sand(delta)

    for ind in idx:
        msh_file = os.path.join(args.mesh_dir, f"mesh_{ind}.msh")
        if not os.path.exists(msh_file):
            raise FileNotFoundError(f"Missing mesh file {msh_file}")
        sh = Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)
        SL.append(shape=sh, count=1, meshsize=1, material=material)

    particles = SL.generate_mesh(
        dimension=3,
        contact_radius=contact_radius,
        plot_mesh=False,
        plot_node_text=False,
        shapes_in_parallel=False,
        keep_mesh=True
    )

    # Orientation
    if args.use_input_rotations:
        for i, ind in enumerate(idx):
            R = quat_to_R(rotations[ind % rotations.shape[0]])
            apply_rotation_about_centroid(particles[i][0], R)
    else:
        Rshared = quat_to_R(np.array(args.shared_quaternion, dtype=float))
        for i in range(N):
            apply_rotation_about_centroid(particles[i][0], Rshared)

    # Estimate average particle diameter from geometry bboxes
    diams = []
    for i in range(N):
        P = np.asarray(particles[i][0].pos, dtype=np.float64)
        sz = P.max(axis=0) - P.min(axis=0)
        diams.append(np.max(sz))
    avg_diam = float(np.mean(diams))

    # Select and pre-scale centers to prevent overlap
    C_raw = positions[idx]
    C0 = C_raw - C_raw.mean(axis=0)  # zero-mean for stable scaling
    if args.centers_scale_mode == "auto":
        nn = pairwise_nn_mean(C0)
        if nn <= 1e-12:
            scale_c = 1.0
        else:
            target_nn = args.beta_spacing * avg_diam
            scale_c = max(1.0, target_nn / nn)
        print(f"[centers] auto scale: meanNN={nn:.6e}, avg_diam={avg_diam:.6e}, beta={args.beta_spacing:.3f} -> scale_c={scale_c:.6f}")
    elif args.centers_scale_mode == "fixed":
        scale_c = float(args.centers_scale_factor)
        print(f"[centers] fixed scale: scale_c={scale_c}")
    else:
        scale_c = 1.0
        print("[centers] no scaling.")

    C = C0 * scale_c

    # Place by scaled centers (preserve relative spacing)
    for i in range(N):
        prt = particles[i][0]
        P = np.asarray(prt.pos, dtype=np.float64)
        c = P.mean(axis=0)
        prt.shift(-c)
        prt.shift(C[i])

    # Gravity/external
    gvec = np.array([0.0, 0.0, float(args.gravity)], dtype=float)
    for i in range(N):
        prt = particles[i][0]
        prt.acc += gvec
        prt.extforce += [0.0, 0.0, float(args.gravity) * prt.material.rho]

    # Geometry bbox and cluster center
    gmin0, gmax0 = compute_geometry_bbox(particles)
    gsize0 = gmax0 - gmin0
    gctr0 = 0.5 * (gmin0 + gmax0)

    # Target and inner (m)
    target = np.array([mm_to_m(args.nx), mm_to_m(args.ny), mm_to_m(args.nz)], dtype=float)
    pad = mm_to_m(args.padding_mm)
    inner = np.maximum(target - 2.0*pad, 1e-12)

    raw_scale = np.min(inner / np.maximum(gsize0, 1e-12))
    scale = args.scale_tweak * raw_scale

    # Placement
    pad_vec = np.array([pad, pad, pad], dtype=float)
    if args.pin_to == "center":
        target_center = 0.5 * target
    elif args.pin_to == "corner":
        target_center = pad_vec - scale * (gmin0 - gctr0)
    else:  # bottom
        cx = 0.5 * target[0]
        cy = 0.5 * target[1]
        cz = pad - scale * (gmin0[2] - gctr0[2])
        target_center = np.array([cx, cy, cz], dtype=float)

    # Apply geometry scale+translate
    rigid_scale_translate_geometry(particles, scale=scale, center_from=gctr0, center_to=target_center)

    # Diagnostics
    gmin, gmax = compute_geometry_bbox(particles)
    print("=== SQUEEZE REPORT ===")
    print("Orig bbox:", gmin0, gmax0, "size:", gsize0)
    print("Target (m):", target, "inner:", inner, "scale:", scale, "(raw:", raw_scale, ")")
    print("New bbox:", gmin, gmax, "size:", gmax - gmin)
    print("Margins (min->0):", gmin - 0.0, "Margins (max->target):", target - gmax)

    # Contact & walls
    normal_stiffness = material.cnot / contact_radius
    contact = Contact(contact_radius, normal_stiffness, args.damping_ratio, args.friction_coefficient)
    wall = Wall3d(1, 0.0, 0.0, 0.0, *target.tolist())

    exp = Experiment(particles, wall, contact)

    # Optional plot
    if not args.noplot:
        try:
            plot3d_setup(
                particles,
                dotsize=12,
                wall=wall,
                show_particle_index=False,
                delta=delta,
                contact_radius=contact_radius,
                trisurf=True,
                trisurf_transparent=False,
                trisurf_linewidth=0,
                trisurf_alpha=0.6,
                noscatter=True
            )
        except Exception as e:
            print("[warn] plot3d_setup failed:", e)

    # Minimal Open3D preview
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=not args.novis)
        # particles
        for i in range(N):
            prt = particles[i][0]
            P = np.asarray(prt.pos, dtype=np.float64)
            edges = getattr(prt, "bdry_edges", None)
            if edges is None:
                continue
            edges = np.asarray(edges)
            if edges.ndim == 2 and edges.shape[1] == 3:
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(P),
                    triangles=o3d.utility.Vector3iVector(edges.astype(np.int32))
                )
                mesh.compute_vertex_normals()
                vis.add_geometry(mesh)
            elif edges.ndim == 2 and edges.shape[1] == 2:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(P)
                ls.lines  = o3d.utility.Vector2iVector(edges.astype(np.int32))
                vis.add_geometry(ls)
        # box
        x_max, y_max, z_max = target.tolist()
        pts_box = np.array([[0,0,0],[x_max,0,0],[0,y_max,0],[x_max,y_max,0],
                            [0,0,z_max],[x_max,0,z_max],[0,y_max,z_max],[x_max,y_max,z_max]], dtype=float)
        lines_box = np.array([[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]], dtype=int)
        wall_ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts_box),
                                       lines=o3d.utility.Vector2iVector(lines_box))
        vis.add_geometry(wall_ls)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(os.path.join(args.path, "setup.png"))
        if not args.novis:
            vis.run()
        vis.destroy_window()
    except Exception as e:
        print("[warn] Open3D preview failed:", e)

    # Save HDF5
    out_h5 = os.path.join(args.path, "setup.h5")
    print("Saving experiment to:", out_h5)
    exp.save(out_h5)
    print("Done.")


if __name__ == "__main__":
    main()
