import numpy as np
# import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from shape_dict import Shape
# from genmesh import genmesh
from exp_dict import ShapeList, Wall3d, Contact, Experiment, plot3d_setup, GridDist
from arrangements import get_incenter_mesh_loc

from shape_params import Param

import open3d as o3d

import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
# parser.add_argument('--shape', type=str, help='shape of each particle', default='small_disk')
parser.add_argument('--particle_rad', type=float, help='radius of each particle', default=8e-3)
parser.add_argument('--vel_val', type=float, help='initial velocity', default=5e2)
parser.add_argument('--acc_val', type=float, help='initial velocity', default=-10)

parser.add_argument('--G_scale', type=float, help='shear modulus scaling', default=0.5)
parser.add_argument('--Gnot_scale', type=float, help='shear modulus scaling', default=1e-4)
parser.add_argument('--K_scale', type=float, help='bulk modulus scaling', default=0.5)
parser.add_argument('--rho_scale', type=float, help='density scaling', default=1)

parser.add_argument('--meshsize_factor', type=float, help='meshsize factor compared to radius', default=4)
parser.add_argument('--delta_factor', type=float, help='delta factor compared to radius', default=3)
parser.add_argument('--contact_rad_factor', type=float, help='contact radius factor compared to radius', default=4)

# parser.add_argument('--setup_file', type=str, help='output setup directory', default='data/hdf5/all.h5')
parser.add_argument('--datapath', type=str, help='output setup directory', default='/home/debdeep/ls-shapes')
parser.add_argument('--path', type=str, help='output setup directory', default='examples_output/')
parser.add_argument('--noplot', action='store_true', help='whether to show plot or not')
parser.add_argument('--novis', action='store_true', help='whether to show plot or not in open3d')
# finish parsing
args = parser.parse_args()


""" Two particles colliding in 3D
"""

rad = 40
delta = rad/args.delta_factor
contact_radius = rad/args.contact_rad_factor    # conserves momentum better (than delta/3)
material = material_dict.kalthoff3d(delta, K_scale=1, Gnot_scale=5e9)

SL = ShapeList()


args.datapath = "/home/debdeep/ls-shapes"
positions = np.loadtxt(args.datapath + '/Input/positions_image.dat')
rotations = np.loadtxt(args.datapath + '/Input/rotations_image.dat')

N = 4
# N = 10
# N = 100
# N = 50
# N = 50
# N = 403

# N1 = 1
# N2 = 403
# N1 = 1
# N2 = 1

ind_range = range(N)
# ind_range = range(N1, N2)
# ind_range = range(0, 403, 1)
# ind_range = range(0, 403, 5)


# for ind in range(N):
for ind in ind_range:
    msh_file = args.datapath + '/msh/mesh_' + str(ind) + '.msh'
    shape = Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)
    SL.append(shape=shape, count=1, meshsize=1, material=material)

# material.print()

particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius, plot_mesh=False, plot_node_text=False, shapes_in_parallel=True, keep_mesh=True)

# for ind in range(N):
for i,ind in enumerate(ind_range):
    particle = particles[i][0]

    # particle.scale(scale)
    quat = rotations[ind]

    q1 =  quat[0]
    q2 =  quat[1]
    q3 =  quat[2]
    q4 =  quat[3]
    RLSDEM = [[-q1**2+q2**2-q3**2+q4**2 , -2*(q1*q2-q3*q4)    ,  2*(q2*q3+q1*q4)],
              [-2*(q1*q2+q3*q4)     , q1**2-q2**2-q3**2+q4**2 ,  -2*(q1*q3-q2*q4)],
              [2*(q2*q3-q1*q4)      , -2*(q1*q3+q2*q4)    ,  -q1**2-q2**2+q3**2+q4**2]]

    # centroid = np.mean(particle.pos, axis=0)
    # print('mean position before', centroid)
    # particle.shift(-centroid)
    # print('mean position at zero', np.mean(particle.pos, axis=0))
    # particle.trasnform_bymat(RLSDEM, quiet=True) # incorrect, since we need 3x3 matrix, not 4x4
    # borrowing open3d's own rotation method for our particles, which is known to work on lsdem-shapes

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(particle.pos)

    # scaling the shapes to avoid collision
    # scale = 0.8
    scale = 1.0
    pcd.scale(scale, center=pcd.get_center())

    pcd.rotate(RLSDEM)
    particle.pos = pcd.points

    # print('mean position after', np.mean(particle.pos, axis=0))
    shift = positions[ind]
    particle.shift(shift)

    particle.vel += [0, 0, args.vel_val]
    # particle.acc += [0, 0, -16e4]
    # particle.extforce += [0, 0, -16e4 * particles[0][1].material.rho]


#######################################################################


# contact properties
normal_stiffness = material.cnot / contact_radius
damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

#######################################################################
# wall info: tight bounding box
# L = 0
x_ini = particles[0][0].pos[0,0]
y_ini = particles[0][0].pos[0,1]
z_ini = particles[0][0].pos[0,2]

x_min = x_ini
y_min = y_ini 
z_min = z_ini
x_max = x_ini
y_max = y_ini
z_max = z_ini

buff = contact_radius * 1.1
# buff = 0

for i,ind in enumerate(ind_range):
    # L = max(L, np.amax(np.abs(particles[i][0].pos)))
    x_min = min(x_min, np.min(particles[i][0].pos[:,0]))
    x_max = max(x_max, np.max(particles[i][0].pos[:,0]))
    y_min = min(y_min, np.min(particles[i][0].pos[:,1]))
    y_max = max(y_max, np.max(particles[i][0].pos[:,1]))
    z_min = min(z_min, np.min(particles[i][0].pos[:,2]))
    z_max = max(z_max, np.max(particles[i][0].pos[:,2]))

print('Tight bounding box', [x_min, x_max], [y_min, y_max], [z_min, z_max])

x_min -= buff
x_max += buff
y_min -= buff
y_max += buff
z_min -= buff
z_max += buff

# L = 10 * rad
# x_min = -L
# y_min = -L
# z_min = -L
# x_max = L
# y_max = L
# z_max = L
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
# wall = Wall3d(0)

# open3d box
points = [
[x_min, y_min, z_min],
[x_max, y_min, z_min],
[x_min, y_max, z_min],
[x_max, y_max, z_min],
[x_min, y_min, z_max],
[x_max, y_min, z_max],
[x_min, y_max, z_max],
[x_max, y_max, z_max],
]
lines = [
[0, 1],
[0, 2],
[1, 3],
[2, 3],
[4, 5],
[4, 6],
[5, 7],
[6, 7],
[0, 4],
[1, 5],
[2, 6],
[3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
wall_line_set = o3d.geometry.LineSet(
points=o3d.utility.Vector3dVector(points),
lines=o3d.utility.Vector2iVector(lines),
)
wall_line_set.colors = o3d.utility.Vector3dVector(colors)
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points = o3d.utility.Vector3dVector(points)
point_cloud2.paint_uniform_color([0, 1, 0])

# o3d.visualization.draw_geometries([line_set, point_cloud2])

exp = Experiment(particles, wall, contact)


#######################################################################

if not args.noplot:
    plot3d_setup(particles, dotsize=15, wall=wall, show_particle_index=False, delta=delta, contact_radius=contact_radius, trisurf=True, trisurf_transparent=False, trisurf_linewidth=0,trisurf_alpha=0.6, noscatter=True)


#######################################################################

vis = o3d.visualization.Visualizer()
if args.novis:
    vis.create_window(visible=False)
else:
    vis.create_window(visible=True)   # turn on for interactive
vis.create_window()

# for ind in range(N):
for i,ind in enumerate(ind_range):
    particle = particles[i][0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(particle.pos[particle.edge_nodes])

    # vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(particle.pos), o3d.cpu.pybind.utility.Vector3iVector(particle.bdry_edges))

    # adds a dash of reflections
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)
    # vis.update_geometry(mesh)

vis.add_geometry(wall_line_set)
# vis.update_geometry(wall_line_set)

## better lighting
opt = vis.get_render_option()
opt.show_coordinate_frame = True
opt.light_on = True
# opt.mesh_show_wireframe = True

vis.poll_events()
vis.update_renderer()
figname = args.path + '/setup.png'
vis.capture_screen_image(figname)

if not args.novis:
    vis.run()     # turn on for interactive
vis.destroy_window()

#######################################################################

# save the data
setup_file = args.path + '/setup.h5'
print('saving experiment setup to', setup_file)
exp.save(setup_file)
