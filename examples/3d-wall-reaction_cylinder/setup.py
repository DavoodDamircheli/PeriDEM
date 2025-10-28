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
parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--L', type=float, help='half-length of container', default=100e-3)
parser.add_argument('--wallh_ratio', type=float, help='half-length of wall height vs wall width', default=2)
parser.add_argument('--nx', type=int, help='number of particles in x dir', default=10)
parser.add_argument('--ny', type=int, help='number of particles in y dir', default=10)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=-15)
parser.add_argument('--acc_val', type=float, help='initial velocity', default=-10)
parser.add_argument('--shape', type=str, help='shape of grain', default='sphere_small_3d')

parser.add_argument('--G_scale', type=float, help='shear modulus scaling', default=0.5)
parser.add_argument('--Gnot_scale', type=float, help='shear modulus scaling', default=1e-4)
parser.add_argument('--K_scale', type=float, help='bulk modulus scaling', default=0.5)
parser.add_argument('--rho_scale', type=float, help='density scaling', default=1)

parser.add_argument('--meshsize_factor', type=float, help='meshsize factor compared to radius', default=50)
parser.add_argument('--delta_factor', type=float, help='delta factor compared to radius', default=5)
parser.add_argument('--contact_rad_factor', type=float, help='contact radius factor compared to radius', default=5)

parser.add_argument('--setup_file', type=str, help='output setup directory', default='data/hdf5/all.h5')
parser.add_argument('--plot', action='store_true', help='whether to show plot or not')
# finish parsing
args = parser.parse_args()

print('plot', args.plot)
print('saving experiment setup to', args.setup_file)

""" Two particles colliding in 3D
"""

rad = 1e-3
# n_x, n_y, n_z = (5, 5, 5)
# n_x, n_y, n_z = (2, 2, 2)
n_x, n_y, n_z = (1, 1, 1)

delta = rad/args.delta_factor
meshsize = rad/args.meshsize_factor
contact_radius = rad/args.contact_rad_factor    # conserves momentum better (than delta/3)

# L = n_x * (rad + contact_radius)
# L = 2 * n_x * (rad + contact_radius)
L = 2e-3
# L = n_x * (rad + contact_radius)
# L = n_x * (rad + contact_radius) * 2

gap = rad + contact_radius
x = np.linspace(-L+gap, L-gap, n_x)
X, Y, Z = np.meshgrid(x, x, x)
xf = X.flatten()
yf = Y.flatten()
zf = Z.flatten()
shifts = np.c_[xf, yf, zf]

SL = ShapeList()

if args.shape == 'sphere_small_3d':
    # shape = shape_dict.sphere_3d(rad=rad, meshsize=meshsize)
    shape = shape_dict.sphere_small_3d()
elif args.shape == 'disk_w_hole_3d':
    shape = shape_dict.disk_w_hole_3d()
elif args.shape == 'plus_small_3d':
    shape = shape_dict.plus_small_3d()
elif args.shape == 'plus_coarse_small_3d':
    shape = shape_dict.plus_coarse_small_3d()
else:
    print('Wrong shape specified')

# ## lsdem shapes
# ls_mesh_path = '/ddnA/work/debdeep/msh/'
# # ind = 37
# msh_file = 'ls_mesh_path' + '/mesh_' + str(ind) + '.msh'
# shape = shape_dict.Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


## Scale here to avoid scaling delta also (before computing bond narr)
# def scale(self, scale):
#     self.pos *= scale
#     # update the volume
#     self.vol *= (scale**self.dim)

# SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.disk_w_hole_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.plus_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

material = material_dict.peridem_3d(delta)

material.print()


# # apply transformation
# particles[0][0].rotate3d('z', np.pi/2)
# particles[0][1].rotate3d('z', -np.pi/2)
# particles[0][1].shift([0, 0, 2*rad+contact_radius*1.1])
# # particles[0][1].shift([0, 0, 2.6e-3])

# Initial data
# particles[0][0].vel += [0, 0, args.vel_val]
# particles[0][1].acc += [0, 0, -16e4]
# particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]

# wall info

# L = 4e-3
x_min = -L
y_min = -L
z_min = -L
x_max = L
y_max = L
z_max = L
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
count = n_x*n_y*n_z

# lx = np.linspace(-L,L, n_x)
# ly = np.linspace(-L,L, n_y)
# lz = np.linspace(-L,L, n_z)
#
# incenter_pos = np.meshgrid(lx, ly, lz)
# print('incenter_pos', incenter_pos)
#
SL.append(shape=shape, count=count, meshsize=meshsize, material=material)
particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius, plot_node_text=False, plot_shape=False, plot_mesh=False)


# vv = 1e-3 + contact_radius
# shifts = [ [-vv, -vv, -vv],
#         [-vv, -vv, vv],
#         [-vv, vv, -vv],
#         [-vv, vv, vv],
#         [vv, -vv, -vv],
#         [vv, -vv, vv],
#         [vv, vv, -vv],
#         [vv, vv, vv]]



for i in range(count):
    # particles[0][i].scale(rad/1e-3)
    part = particles[0][i]
    # part.shift(shifts[i])

particles[0][0].vel += [0,  args.vel_val, 0]

# ## Apply transformation
# seed(1)
# for i in range(count):
#     # particles[0][i].scale(scaling_list[i])
#     # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
#     # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))
#     ## generate random rotation
#     # particles[0][i].rotate(0 + (random() * (2*np.pi - 0)) )
#
#     particles[0][i].rotate3d('x',0 + (random() * (2*np.pi - 0)))
#     particles[0][i].rotate3d('y',0 + (random() * (2*np.pi - 0)))
#     particles[0][i].rotate3d('z',0 + (random() * (2*np.pi - 0)))
#
#     shift = incenter_pos[i]
#     print('shift', shift)
#     particles[0][i].shift(shift)



# g_val = -5e3
# v_val = -1e1
# ## Initial data
# for i in range(count):
#
#     particles[0][i].vel += [0, 0, v_val]
#     particles[0][i].acc += [0, 0, g_val]
#     particles[0][i].extforce += [0, 0, g_val * particles[0][i].material.rho]




# contact properties
# normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));

# normal_stiffness = material.cnot / contact_radius
normal_stiffness = material_dict.peridem_3d(contact_radius).cnot / contact_radius

# normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

if args.plot:
    plot3d_setup(particles, dotsize=15, wall=wall, show_particle_index=True, delta=delta, contact_radius=contact_radius, save_filename=args.setup_file+'setup.png')

exp = Experiment(particles, wall, contact)

#######################################################################

# save the data
print('saving experiment setup to', args.setup_file)
exp.save(args.setup_file)
