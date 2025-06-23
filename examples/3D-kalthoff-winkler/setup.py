
import numpy as np
# import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
# from genmesh import genmesh
from exp_dict import ShapeList, Wall, Wall3d,  Contact, Experiment, plot3d_setup, GridDist, plot_nodes
from arrangements import get_incenter_mesh_loc

from shape_params import Param
import argparse
import pdb
#pdb.set_trace()
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
parser.add_argument('--setup_file', type=str, help='output setup directory', default='data/hdf5/all.h5')
parser.add_argument('--delta_frac', type=float, help='fraction of radius to be equal to delta', default=2)
parser.add_argument('--plot', action='store_true', help='whether plot or not')
parser.add_argument('--meshsize', action='store_true', help='whether plot or not')
# finish parsing
args = parser.parse_args()

print('plot', args.plot)
print('saving experiment setup to', args.setup_file)

#-------------size of the elements of the exprement------
#--------kalthoff plate
imx = 0e-3
r1 = 25e-3#25e-3
h1 = 100e-3#100e-3

lx = 200e-3
ly = 100e-3
lz = 9e-3
lyshift = -0e-3
notch_lenght = 50e-3
notch_angle = 2 
notch_w = 1.5e-3  #????????????????????????????????????????????? 
notch_dist_corner = 73.5e-3#735e-4
ncor = notch_dist_corner
#----- we need some error for convex tol
radianA = np.deg2rad(notch_angle)
rate_cnx = 10000
l_trig = 2 * notch_lenght *np.tan(radianA/2)
convex_tol = (lx-(2*(l_trig+ncor)))/rate_cnx
     


convex_tol = 3e-4
dist_impactor = 7e-3
#--------------end of expremint's element---------------
v_val = -32
msh_fac = 10# 40 
meshsize = lx/msh_fac 
delta_frac = 4.0       #args.delta_frac
delta = meshsize *delta_frac
#contact_radius = (lx/100) * 4
print("\n Dx is \n"+ str(meshsize))
print("\n Delta is \n"+ str(delta)+"\n lx is \n"+str(lx))
contact_radius = lx/25
clamp = True 
prescribe = False
stiffness = 'constant-value'#'delta-dependent'
stif_scale = 5
Gnot_scale = 1#2.12
vol = lx*ly*lz 
mass = 8000*vol
rho_scale = 1.6/mass
print("\n rho_scale\n"+str(rho_scale))
#rho_scale = 1.5
impact_rho_scaling = 1.0  #1.2
g_val = 0
#---------------end of the wall---------------------
# Create a list of shapes

SL = ShapeList()
# append each shape with own scaling
#------------------------plate with nothches------------------
#--------- access to ther dimension of structures in the expremnets

#shape1 = shape_dict.kalthoff_3d_plate_beta0(lx,ly,lz, notch_lenght,notch_angle, notch_dist_corner,meshsize)
#shape1 = shape_dict.kalthoff_3d_plate_beta0(lx,ly,lz, notch_lenght,notch_angle, notch_dist_corner,convex_tol,meshsize)
shape1 = shape_dict.kalthoff_3d_plate_beta(lx,ly,lz, notch_lenght,notch_w, notch_dist_corner,meshsize)
#--------Material properties
material=material_dict.kalthoff3d(delta, Gnot_scale=Gnot_scale, rho_scale=rho_scale)

#material=material_dict.kalthoff3d_beta(delta, Gnot_scale=Gnot_scale, rho_scale=rho_scale)
material.print()
#------------------bringing plate into the exprements--------
#-----append different particles and generate mesh!!!
SL.append(shape=shape1, count=1, meshsize=meshsize, material=material,plot_shape=False)

#if not prescribe:
    #-----------------lets define and bring the impactor to the exprement 
impactor_material=material_dict.kalthoff3d(delta, Gnot_scale=Gnot_scale, rho_scale=impact_rho_scaling)
    #------------- appending the second shape=plank=impacor to the exprement-------------
#shape2 = shape_dict.cube_3d(imx,imx,imx,meshsize)
shape2 = shape_dict.cylinder(r = r1,h = h1,meshsize = meshsize)
#shape2=shape_dict.kalthoff_impactor_3d(lx,ly,lz,imx,imy,imz,meshsize=meshsize)
SL.append(shape=shape2, count=1, meshsize=meshsize, material=impactor_material,plot_shape=False)
#-------------------- generate the mesh for each and all  shapes that are in SL (exprement)

particles = SL.generate_mesh(dimension = 3, contact_radius = contact_radius, plot_mesh=False,plot_shape=False)
###################plot_nodes(particles)

#particles = SL.generate_mesh(dimension = 3, contact_radius = contact_radius, plot_mesh=True, plot_shape=False, shapes_in_parallel=False)


particles[0][0].rotate3d('x',1.5* np.pi)
#particles[0][0].rotate3d('y', np.pi/2)
particles[0][0].shift([-lx/2,0, lyshift])  #???????????????????????? just add 0,
sheet = particles[0][0]
particles[1][0].shift([0, 0,0.01*lx+contact_radius*1.1])  #???????????????????????? just add 0,


m = 250e-3
#wall3d = Wall3d(1,-m,-m,-m,m,m,m)
wall3d = Wall3d(1,-m,-m,-m,m,m,m)
wall3d.print()

dist = lx/3+25e-3 
s = ly
a = 1.5e-3
'''
if clamp:
    for i in range(len(sheet.pos)):
        if (np.abs(sheet.pos[i,1] - (s/2)) < delta):
            if (np.abs(sheet.pos[i,0] - (0)) > dist/2+a):
                sheet.clamped_nodes.append(i)
    print("clamp nodes are")
    print(sheet.clamped_nodes)
'''

#--------------Xvalues = pos[i,0]
#--------------Yvalues = pos[i,2]
#--------------Zvalues = pos[i,1]
a1 = lx/2
b1 = lx-2*notch_dist_corner
c1 = notch_dist_corner

if clamp:
    for i in range(len(sheet.pos)):
        #if (np.abs(sheet.pos[i,2] - (ly/5)) < delta):                   #
        if np.abs(sheet.pos[i,2] )> ly/1-20*a :                   #
            x = sheet.pos[i,0]
            if (np.abs(x) >=c1/2-2*a): #or x<=c1-a  ):   #----X-value pos[i,0]
                sheet.clamped_nodes.append(i)
    print("clamp nodes are")
    print(sheet.clamped_nodes)

if prescribe:
    #specify boundary velocity
    for i in range(len(sheet.pos)):
        if (np.abs(sheet.pos[i,2] - (s/2)) < 1e-12):
            if (np.abs(sheet.pos[i,0] - (0)) < delta):
                sheet.vel[i] += [0,0, v_val]
else:
    part = particles[1][0]
    part.shift([0, 0,dist+contact_radius*1.1]) 
    part.vel += [0,0, v_val]
    part.breakable = 0
# contact properties????????????????????
if stiffness == 'constant-expression':
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
elif stiffness == 'delta-dependent':
    normal_stiffness = material.cnot / contact_radius
elif stiffness == 'constant-value':
    normal_stiffness = 3.56234e+20*stif_scale
else:
    print('Wrong stiffness type')
damping_ratio = 0.8

friction_coefficient = 0.8

normal_stiffness = material_dict.peridem_3d(contact_radius).cnot/contact_radius

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)
    
#plot3d_setup(particles, dotsize=10,contact_radius=contact_radius, delta = delta, wall=wall3d)

# return Experiment(particles, wall, contact)
exp =  Experiment(particles, wall3d, contact)

#######################################################################

# save the data
print('saving experiment setup to', args.setup_file)
exp.save(args.setup_file)
