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
import pdb
#pdb.set_trace()
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
# parser.add_argument('--shape', type=str, help='shape of each particle', default='small_disk')
parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--L', type=float, help='half-length of container', default=100e-3)
parser.add_argument('--wallh_ratio', type=float, help='half-length of wall height vs wall width', default=2)
parser.add_argument('--nx', type=int, help='number of particles in x dir', default=10)
parser.add_argument('--ny', type=int, help='number of particles in y dir', default=10)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=-5e2)
parser.add_argument('--acc_val', type=float, help='initial velocity', default=-10)
parser.add_argument('--shape', type=str, help='shape of grain', default='grains')
#parser.add_argument('--msh_path', type=str, help='shape of grain', default='/home/davood/projects/beta_perigrain_v2/grain-data/test5grains')
parser.add_argument('--msh_path', type=str, help='shape of grain', default='/media/davood/093c4011-b7d0-4917-a86a-7c2fb7e4c748/project_data/grain-mesh-meshsize-10-voxel-7')

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

print('plot', args.plot)
print('saving experiment setup to', args.setup_file)

""" Two particles colliding in 3D
"""
mf = 30
#mf=args.meshsize_factor
print('meshsize factor is ', mf)

rad = 1e-3 
#rad = 1 
#n_x, n_y, n_z = (8, 8,8)
#n_x, n_y, n_z = (9, 9,9)
#n_x, n_y, n_z = (10,10,10)
#n_x, n_y, n_z = (8, 5, 10)
#n_x, n_y, n_z = (7, 7, 7)
#n_x, n_y, n_z = (4, 4, 4)
#n_x, n_y, n_z = (2,2,2)
#n_x, n_y, n_z = (1,1,4)
#n_x, n_y, n_z = (2,5,10)

n_x, n_y, n_z = (1,1,1)
#n_x, n_y, n_z = (15,15,10)
#n_x, n_y, n_z = (20,20,10)

#n_x, n_y, n_z = (5,5,5)
print("rad is ",rad)
delta = rad/args.delta_factor
meshsize = rad/mf
print('delta is ', delta)
print('meshsize is', meshsize)


args.contact_rad_factor=4 

contact_radius = rad/args.contact_rad_factor    # conserves momentum better (than delta/3)

L = n_x * (rad+contact_radius) 
Ly = n_y * (rad+contact_radius) 
Lz = n_z * (rad+contact_radius) 

gap = rad + contact_radius
gap_scale = 1.05
gap = gap_scale * gap
x = np.linspace(-L+gap, L-gap, n_x)
y = np.linspace(-Ly+gap, Ly-gap, n_y)
z = np.linspace(-Lz+gap, Lz-gap, n_z)
X, Y, Z = np.meshgrid(x, y, z)
xf = X.flatten()
yf = Y.flatten()
zf = Z.flatten()
shifts = np.c_[xf, yf, zf]

SL = ShapeList()

material = material_dict.peridem_3d(delta)

material.print()

x_min = -L
y_min = -Ly
z_min = -Lz
x_max = L
y_max = Ly
z_max = Lz
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
cnt = 1 
max_files = 400

# Calculate the total number of exclusions
exclusions = [356, 377]
total_exclusions = (cnt // max_files) * len(exclusions) + sum(1 for ex in exclusions if ex <= cnt % max_files)

if args.shape == 'grains':
    for i in range(1, cnt + 1):
        current_file_index = i % max_files
        if current_file_index == 0:
            current_file_index = max_files
        if current_file_index in exclusions:
            continue
        msh_file = f"{args.msh_path}/mesh_{current_file_index}.msh"
        print(f"Reading file: {msh_file}")  # Debug statement
        shape = shape_dict.Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file, scale_mesh_to=rad, centroid_origin=True)
        SL.append(shape=shape, count=1, meshsize=meshsize, material=material, plot_shape=False)
else:
    SL.append(shape=shape, count=cnt, meshsize=meshsize, material=material, plot_shape=False)

particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius, plot_node_text=False, plot_shape=False, plot_mesh=False)

# Adjust the for loop to handle arbitrary cnt values larger than 400

#------------------------------------------------------------------------
settle_gravity=1

if settle_gravity==1:
    g_val = -5e4
    for i in range(cnt - total_exclusions):
        particles[i][0].shift(shifts[i])
        particles[i][0].acc += [0, 0,g_val]
        particles[i][0].extforce += [0,0, g_val * particles[i][0].material.rho]
else:
    for i in range(cnt - total_exclusions):
        particles[i][0].shift(shifts[i])
#------------------------------------------------------------------------
a1 = list(range(len(particles)))
a2= len(particles)


normal_stiffness = material_dict.peridem_3d(contact_radius).cnot / contact_radius


damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)
args.plot=1
if args.plot:
    plot3d_setup(particles, dotsize=15, wall=wall,show_plot=True, show_particle_index=True, delta=delta, contact_radius=contact_radius, save_filename='setup.png')

exp = Experiment(particles, wall, contact)

#######################################################################

# save the data
print('saving experiment setup to', args.setup_file)
exp.save(args.setup_file)
