import numpy as np
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
parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
# parser.add_argument('--shape', type=str, help='shape of each particle', default='small_disk')
#parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--particle_size', type=float, help='size of each particle a=b=c', default=1e-3)
parser.add_argument('--L', type=float, help='half-length of container', default=100e-3)
parser.add_argument('--wallh_ratio', type=float, help='half-length of wall height vs wall width', default=2)
parser.add_argument('--nx', type=int, help='number of particles in x dir', default=10)
parser.add_argument('--ny', type=int, help='number of particles in y dir', default=10)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=-20)
parser.add_argument('--acc_val', type=float, help='initial velocity', default=-10)

parser.add_argument('--G_scale', type=float, help='shear modulus scaling', default=0.9)
parser.add_argument('--Gnot_scale', type=float, help='shear modulus scaling', default=1e-4)
parser.add_argument('--K_scale', type=float, help='bulk modulus scaling', default=0.5)
parser.add_argument('--rho_scale', type=float, help='density scaling', default=1)

parser.add_argument('--meshsize_factor', type=float, help='meshsize factor compared to radius', default=10)
parser.add_argument('--delta_factor', type=float, help='delta factor compared to radius', default=3.015)
parser.add_argument('--contact_rad_factor', type=float, help='contact radius factor compared to radius', default=7)

parser.add_argument('--setup_file', type=str, help='output setup directory', default='data/hdf5/all.h5')
parser.add_argument('--plot', action='store_true', help='whether to show plot or not')
# finish parsing
args = parser.parse_args()

print('plot', args.plot)
print('saving experiment setup to', args.setup_file)

""" Two particles colliding in 3D
"""

lx = 2e-4  
ly = 2e-4 
lz = 2e-4 
rad = np.min([lx,ly,lz])
rad=rad/2

## Original
# delta = rad/2
# meshsize = rad/4
# contact_radius = rad/3;    # conserves momentum better (than delta/3)
delta = rad/args.delta_factor
# meshsize = rad/4
meshsize = rad/args.meshsize_factor
contact_radius = rad/args.contact_rad_factor    # conserves momentum better (than delta/3)

SL = ShapeList()

#shape=shape_dict.plus3d(l=lx, meshsize=meshsize)
# shape=shape_dict.cube_3d(lx=lx, ly=ly, lz=lz, meshsize=meshsize)
shape=shape_dict.sphere(rad=lx/2, meshsize=meshsize)

#shape=shape_dict.plus3d_longHand(l=lx,l2=3*lx, meshsize=meshsize)
#material = material_dict.ottawa_sand(delta)
material = material_dict.ottawa_sand_old2(delta)
material.print()

#material = material_dict.kalthoff3d(delta)

SL.append(shape=shape, count=1, meshsize=meshsize, material=material)
# SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.disk_w_hole_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.plus_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

material.print()

particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius,plot_mesh=False, plot_node_text=False)

# apply transformation
#particles[0][0].rotate3d('y', 1*np.pi/2)
# particles[0][1].rotate3d('y', 1*-np.pi/2)
# particles[0][1].shift([0, 0, rad+contact_radius*1.1])
# particles[0][1].shift([-0*lx, 2*lx, -1.1*lx])
# #particles[0][1].shift([-0*lx, 3.2*lx, -2.1*lx])

#particles[0][1].shift([0, 0, 0])
# Initial data
# particles[0][0].movable = 1 
# particles[0][0].vel += [0, 0, -args.vel_val]
# #particles[0][1].vel += [0, 0, args.vel_val]
# particles[0][1].acc += [0, 0, -16e4]
# particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]
# wall info

L = 1e-4+0.5*contact_radius
L1 = 1e-4
L2 = 1e-4+2.01*contact_radius
x_min = -L2
y_min = -L2
z_min = -L1
x_max = L2
y_max = L2
z_max = L
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
# wall = Wall3d(0)
sheet = particles[0][0]
clamp = 0
if clamp:
    for i in range(len(sheet.pos)):
        if (np.abs(sheet.pos[i,2] - (z_min)) < delta/2):
                sheet.clamped_nodes.append(i)
    print(sheet.clamped_nodes)


# contact properties
# normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));

# normal_stiffness = material.cnot / contact_radius
normal_stiffness = material_dict.ottawa_sand_old2(contact_radius).cnot / contact_radius



damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

plot3d_setup(particles, dotsize=35, wall=wall, show_particle_index=True, delta=delta, contact_radius=contact_radius)

exp = Experiment(particles, wall, contact)

#######################################################################
#--- Final echo for logs ---
print("[summary]")
#print(f"  particles:        {len(idx)}")
print(f"  rad:              {rad:.6g} m")
print(f"  meshsize:         {meshsize:.6g} m")
print(f"  delta:            {delta:.6g} m")
print(f"  normal_stiffness:            {normal_stiffness:.6g} ")
print(f"  contact_radius:   {contact_radius:.6g} m")
print(f"  wall:             xmin={x_min:.6g}, ymin={y_min:.6g}, zmin={z_min:.6g}, "
      f"xmax={x_max:.6g}, ymax={y_max:.6g}, zmax={z_max:.6g}")
print("done.")


# save the data
print('saving experiment setup to', args.setup_file)
exp.save(args.setup_file)
