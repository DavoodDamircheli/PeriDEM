#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script (no S scaling) — size is driven only by --rad_scale.

Usage example:
  python setup.py \
    --datapath /path/to/positions-and-rotations \
    --mesh_dir /path/to/mesh_dir \
    --ind 0,1,2 \
    --path /tmp/run-001 \
    --rad_scale 0.04 \
    --savefig
"""

import os
import sys
import argparse
import numpy as np
import h5py

# ====== adjust these imports to your project structure ======
# ============================================================
# Project modules (must be available on PYTHONPATH or in CWD)
sys.path.append(os.getcwd())
import shape_dict, material_dict
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup
from shape_params import Param
from shape_dict import Shape 
from util import *

#-----Debug------
import pdb
#pdb.set_trace()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Pack pre-arranged particles into a target box; size driven by --rad_scale only.")
    # I/O
    ap.add_argument("--datapath", type=str, default='/home/davood/projects/ls-shapes',  help="Path containing positions_image.dat and rotations_image.dat.")
    ap.add_argument("--mesh_dir", type=str, default = '/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/grain-mesh-meshsize-10-voxel-7' , help="Directory with mesh_{i}.msh files.")
    ap.add_argument("--N", type=int, default=1,
                help="Number of particles to generate (overrides --ind if given).")
    ap.add_argument("--ind", type=str, default="3", help="Indices to load, e.g. '0,1,2' or '0:10' or '0:10:2'.")
    ap.add_argument("--path", type=str, default='examples_output', help="Output directory; setup.h5 and optional setup.png saved here.")

    # Geometry / physics
    ap.add_argument("--rad_scale", type=float, default=.040, help="Target circumscribed radius per particle (meters).")
    ap.add_argument("--rad_mesh_ratio", type=float, default=1.0/60.0, help="rad_mesh = rad_scale * rad_mesh_ratio.")
    ap.add_argument("--alpha", type=float, default=3.015, help="delta = alpha * rad_mesh.")
    ap.add_argument("--beta", type=float, default=4.0, help="contact_radius = beta * rad_mesh.")
    ap.add_argument("--cfl_a", type=float, default=0.2, help="CFL safety factor for peridynamic time step.")
    ap.add_argument("--wall_buffer_factor", type=float, default=1.0, help="Wall buffer multiplier on contact_radius.")
    ap.add_argument("--material_model", type=str, default="ottawa_sand", help="Key in material_dict.*")
    ap.add_argument('--plot', action='store_true', help='disable plot')
    args = ap.parse_args()
    # Resolve indices
    idx = list(range(args.N))


    # Load placement data
    #positions, rotations = load_positions_rotations(args.datapath)
    # --- load positions & rotations --- #
    positions = np.loadtxt(os.path.join(args.datapath, 'Input', 'positions_image.dat'))
    rotations = np.loadtxt(os.path.join(args.datapath, 'Input', 'rotations_image.dat'))

    # --- Set fundamental geometric scales from rad_scale ---
    rad_scale = 1 
    rad_mesh = rad_scale * float(args.rad_mesh_ratio)
    delta = float(args.alpha) * rad_mesh
    contact_radius = float(args.beta) * rad_mesh

    # Material
    material=material_dict.ottawa_sand(delta) 

    # Time step (choose a representative spatial scale; tie it to rad_mesh or a proxy)
    # If you have a project-specific estimate_h_from_msh, replace h_mean here.
    h_mean = rad_mesh
    dt, wavespeed = peridynamic_timestep(material.E, material.rho, h_mean, args.cfl_a)

    # --- Construct particle shapes with target radius = rad_scale ---
    SL = ShapeList()
    for ind in idx:
        msh_file = os.path.join(args.mesh_dir, f"mesh_{ind}.msh")
        if not os.path.exists(msh_file):
            raise FileNotFoundError(f"Missing mesh file {msh_file}")
        sh = Shape(
            P=None,
            nonconvex_interceptor=None,
            msh_file=msh_file,
        )
        SL.append(shape=sh, count=1, meshsize=1, material=material)

    particles = SL.generate_mesh(
        dimension=3,
        contact_radius=contact_radius,
        plot_mesh=False,
        plot_node_text=False,
        shapes_in_parallel=False,
        keep_mesh=True
    )
    scale_overlap=1e0
    
    scale_global=1e0
    c_global = np.mean([positions[i] for i in idx], axis=0)
    # --- Rotate & translate each particle to its target placement ---
    

    # for k, ind in enumerate(idx):
    #     p = particles[k][0]
    #     R = detect_rotation_matrix(rotations[ind])
    #     ct = positions[ind]
    #     
    #     
    #     
    #     pts = p.pos
    #     # rotate around centroid; local spin; rigid rotation
    #     pts = rotate_about_center(pts, R, ct) 
    #     # scale  around centroid
    #     pts = scale_for_overlap(pts,scale_overlap)
    #     
    #     # shrink the whole configuration toward c_global (compact shrink)
    #     pts = scale_compact(pts,scale_global,c_global)
    #     shift =  c_global + scale_global * (positions[ind] - c_global) 
    #     p.shift(shift)
    #     #pts = pts + positions[ind]  # translate
    #     #p.pos = np.asarray(pts)
    #     
    #------------------------------------------------------------
    scale_global = 1e-3
    scale_overlap=1e-1 
    for k, ind in enumerate(idx):
        p  = particles[k][0]
        R  = detect_rotation_matrix(rotations[ind])
        ct = positions[ind]           # pre-compact desired center
        S  = scale_global
        g  = c_global

        # 1) start with raw points
        pts = p.pos
        print("here") 
        print(pts)
        # 2) rotate about the particle's current centroid (not ct)
        c_loc = ct 
        pts = rotate_about_center(pts, R, ct)
        print("here") 
        print(pts) 
        #
        # # 3) local shrink (should keep centroid fixed if implemented as (P-c)*s + c)
        pts = scale_for_overlap(pts, scale_overlap,ct)
        print("here") 
        print(pts) 
        #
        # # 4) place at the pre-compact center ct
        c_loc = centroid(pts)
        pts += (ct - c_loc)
        # # #
        # # 5) global compact scaling about c_global
        pts = scale_compact(pts, S, g)
        #
        # # 6) after compact, the desired final center is:
        d_final = g + S * (ct - g)
        #
        # # current center after step 5:
        c_now = centroid(pts)
        p.pos = pts
        print("here")
        print(p.pos)
        # 7) translate by a vector to hit the exact final center
        t = d_final -0* c_now
        p.shift(t)          # shift expects a delta vector
        # or p.pos = pts + 
        args.gravity=-1e4
        gvec = np.array([0.0, 0.0, float(args.gravity)], dtype=float)



        p.acc += gvec
            # Project convention used previously: extforce = g * rho (units per your engine)
        p.extforce += [0, 0, float(args.gravity) * p.material.rho]



    #------------------------------------------------------------
    # --- Build tight bounding box and wall with a small buffer ---
    all_pts = np.concatenate([np.asarray(particles[i][0].pos) for i in range(len(particles))], axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    buff = contact_radius * float(args.wall_buffer_factor)

    x_min, y_min, z_min = lo - buff
    x_max, y_max, z_max = hi + buff
    # --- wall & contact (your formulas) --- #
    wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)

    normal_stiffness = material.cnot / contact_radius
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)
    print(contact_radius)
    print(delta)
    exp = Experiment(particles, wall, contact)
    #######################################################################
    args.plot=1
    if  args.plot:
        #plot3d_setup(particles, dotsize=15, wall=wall,show_plot=True, show_particle_index=False, delta=delta, contact_radius=contact_radius, trisurf=True, trisurf_transparent=False, trisurf_linewidth=0,trisurf_alpha=0.6, noscatter=True)
        plot3d_setup(particles, dotsize=15, wall=wall,show_plot=True,  delta=delta, contact_radius=contact_radius )


    #######################################################################

    # args.plot=1 
    # if args.plot:
    #     try:
    #         print("[info] rendering setup.png ...")
    #         # keep it simple; scatter only to avoid occlusion hiding one grain
    #         plot3d_setup(particles, dotsize=15, wall=wall, show_plot=True,
    #                      delta=delta, contact_radius=contact_radius,
    #                      save_filename='setup.png')
    #         print("[info] saved", png)
    #     except Exception as e:
    #         print("[warn] plot3d_setup failed:", e)
    #
    os.makedirs(os.path.dirname(args.setup_file), exist_ok=True)
    print('Saving experiment setup to', args.setup_file)
    exp.save(args.setup_file)


    #--- Final echo for logs ---
    print("[summary]")
    print(f"  particles:        {len(idx)}")
    print(f"  rad_scale:        {rad_scale:.6g} m")
    print(f"  rad_mesh:         {rad_mesh:.6g} m")
    print(f"  delta:            {delta:.6g} m")
    print(f"  contact_radius:   {contact_radius:.6g} m")
    print(f"  wall:             xmin={x_min:.6g}, ymin={y_min:.6g}, zmin={z_min:.6g}, "
          f"xmax={x_max:.6g}, ymax={y_max:.6g}, zmax={z_max:.6g}")
    print(f"  dt (CFL):         {dt:.6g} s   (wavespeed~{wavespeed:.6g} m/s)")
    print("done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

