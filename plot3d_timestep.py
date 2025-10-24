import numpy as np

import h5py

import re
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from sys import argv
import pdb
#pdb.set_trace()
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')


# Optional argument
parser.add_argument('--data_dir', type=str, help='input data directory', default='output/hdf5/')
parser.add_argument('--img_dir', type=str, help='output image directory', default='output/img/')
parser.add_argument('--setup_file', type=str, help='output setup file', default='data/hdf5/all.h5')
parser.add_argument('--all_dir', type=str, help='directory where all files are located')
parser.add_argument('--img_prefix', type=str, help='string prefix to go in the filename', default='img')

parser.add_argument('--fc', type=int, help='first counter')
parser.add_argument('--lc', type=int, help='last counter')
parser.add_argument('--dotsize', type=float, help='dotsize', default=1)
parser.add_argument('--alpha', type=float, help='alpha', default=0.5)
parser.add_argument('--colormap', type=str, help='quantity to plot in color', default='viridis')
parser.add_argument('--xlim', type=str, help='space-separated xlim, within quote. e.g. --xlim \'-1 1\'')
parser.add_argument('--ylim', type=str, help='same as ylim')
parser.add_argument('--zlim', type=str, help='only for 3d')
parser.add_argument('--nogrid', action='store_true', help='do not show grid')
parser.add_argument('--nowall', action='store_true', help='do not show wall')
parser.add_argument('--nocolorbar', action='store_true', help='do not show colorbar')
parser.add_argument('--seaborn', action='store_true', help='use seaborn package')

parser.add_argument('--view_az', type=float, help='camera azimuthal', default=30)
parser.add_argument('--view_angle', type=float, help='camera horizontal angle', default=45)
parser.add_argument('--motion', action='store_true', help='camera motion',default=1)
parser.add_argument('--motion_az_stepsize', type=float, help='camera azimuthal step size', default=0)
parser.add_argument('--motion_angle_stepsize', type=float, help='camera horizontal angle step size', default=0)


parser.add_argument('--quantity', type=str, help='quantity to plot in color', default='damage')
parser.add_argument('--serial', action='store_true', help='read timesteps in serial')
parser.add_argument('--plot_contact_rad', action='store_true', help='plot contact radius')
parser.add_argument('--contact_rad_ind', type=float, help='particle on which to draw contact radius', default=0)
parser.add_argument('--contact_rad_node', type=float, help='node on which to draw contact radius', default=0)
parser.add_argument('--adaptive_dotsize', action='store_true', help='scale dotsizes by volume element')
parser.add_argument('--colorize_only', nargs='+', help='only colorize particle indices', required=False)

# for plotting contact forces
parser.add_argument('--plot_contact_force', action='store_true', help='plot nodes experiencing contact force')
parser.add_argument('--wheel_ind', type=int, help='index of the particle experiencing contact force', default=0)
parser.add_argument('--nbr_ind', type=int, help='index of neighboring particle with which particle[wheel_ind] is in contact', default=4)

# finish parsing
args = parser.parse_args()

data_dir = args.data_dir
img_dir = args.img_dir
setup_filename = args.setup_file
dotsize = args.dotsize

if args.all_dir:
    data_dir       = args.all_dir
    img_dir        = args.all_dir
    setup_filename = args.all_dir+'/setup.h5'

# plot_damage = 1

# plot info
plotinfo_file = data_dir+'/plotinfo.h5'
p = h5py.File(plotinfo_file, "r")
fc = int(p['f_l_counter'][0,0])
lc = int(p['f_l_counter'][1,0])
dt = float(p['dt'][0,0])
modulo = int(p['modulo'][0,0])

if args.fc:
    fc = args.fc
if args.lc:
    lc = args.lc


if args.seaborn:
    import seaborn as sns
    sns.set()

def total_neighbors(conn, N):
    """Compute the total number of neighbors from connectivity data
    :conn: connectivity matrix, mx2, m=total intact bonds
    :N: total number of nodes
    :returns: TODO
    """
    deg = np.zeros(N)
    for i in range(len(conn)):
        deg[conn[i][0]] += 1
        deg[conn[i][1]] += 1
    return deg


# ref_connectivity 
f = h5py.File(setup_filename, "r")
ref_n = {}
for name in f:
    if re.match(r'P_[0-9]+', name):
        # pid = int(name[2:])
        ref_connectivity = np.array(f[name+'/Connectivity']).astype(int)
        N = len(f[name+'/Pos'])
        orig_tot_nbrs = total_neighbors(ref_connectivity, N)
        ref_n[name] = orig_tot_nbrs
contact_radius = np.array(f['pairwise/contact_radius']).astype(float)[0][0]

if args.adaptive_dotsize:
    N = (f['total_particles_univ']).astype(int)[0][0]
    slot = np.zeros(N)
    for name in f:
        if re.match(r'P_[0-9]+', name):
            pid = int(name[2:])
            vol = np.array(f[name+'/Vol'])
            slot[pid] = np.min(vol)
    min_vol = np.min(slot)
    print(min_vol)

    scaled_ds = [[]] * N
    for name in f:
        if re.match(r'P_[0-9]+', name):
            pid = int(name[2:])
            vol = np.array(f[name+'/Vol'])
            scaled_ds[pid] = np.sqrt(vol/min_vol) * float(dotsize)



def genplot(t):
    print(t)
    tc_ind = ('%05d' % t)
    filename = data_dir+'/tc_'+tc_ind+'.h5'
    wall_filename = data_dir+'/wall_'+tc_ind+'.h5'
    out_png = img_dir+'/'+args.img_prefix+'_'+tc_ind+'.png'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #highlight_coords = [(1,1,1),(1,1,3)]
    f = h5py.File(filename, "r")
    
    
    for name in f:
        if re.match(r'P_[0-9]+', name):
            pid = int(name[2:])
            ####### set a condition for particel id!! 
            Pos = np.array(f[name+'/CurrPos'])
            c = 'blue'
            if (args.quantity == 'damage'):
                # print('plotting damage')
                # damage
                P_conn = np.array(f[name+'/Connectivity']).astype(int)
                N = len(Pos)
                now_nbrs = total_neighbors(P_conn, N)
                orig_nbrs = np.array(ref_n[name])
                damage = (orig_nbrs - now_nbrs)/ orig_nbrs
                c = damage
                # print('damage', damage)
                vmin=0
                vmax=1
            if (args.quantity == 'force'):
                # print('plotting damage')
                # damage
                force = np.array(f[name+'/force'])
                f_norm = np.sqrt(np.sum(force**2, axis=1))
                c = f_norm
                # print('damage', damage)
                vmin=None
                vmax=None

            if args.colorize_only:
                indx = [int(ii) for ii in args.colorize_only]
                if pid in indx:
                    pass
                else:
                    c = 'blue'
                    vmin = 0
                    vmax = 1


            if args.adaptive_dotsize:
                dval = scaled_ds[pid]
            else:
                dval = dotsize

            # forcolorbar = ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], c=c, s=dval, marker = '.', linewidth=0, cmap=args.colormap, vmin=vmin, vmax=vmax)
            forcolorbar = ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], c=c, s=dval, marker = '.', linewidth=0, cmap=args.colormap, vmin=vmin, vmax=vmax, alpha=args.alpha)

            if args.plot_contact_force:
                if pid == args.wheel_ind:
                    cnode_list = []
                    nbr_name = ('P_%05d' % args.nbr_ind)
                    nbr_currpos = np.array(f[nbr_name+'/CurrPos'])
                    for node in range(len(Pos)):
                        for i in range(len(nbr_currpos)):
                            dist = np.sqrt(np.sum((nbr_currpos[i] - Pos[node])**2))
                            if dist <= contact_radius:
                                cnode_list.append(node)
                                break
                    plt.scatter(Pos[cnode_list,0], Pos[cnode_list,1], Pos[cnode_list,2], color='red', s=dval, marker='.', linewidth=0, cmap=args.colormap)


            if args.plot_contact_rad:
                if pid == args.contact_rad_ind:
                    node = args.contact_rad_node
                    x0 = Pos[node][0]
                    y0 = Pos[node][1]
                    steps=20
                    tt = np.linspace(0, 2*np.pi, steps, endpoint=True)
                    xx = contact_radius * np.cos(tt)
                    yy = contact_radius * np.sin(tt)
                    plt.plot( xx + x0, yy + y0)

    # wall
    if args.nowall:
        pass
    else:
        wall_alpha = 0.8
        wall_linewidth = 1

        w = h5py.File(wall_filename, "r")
        ss = np.array(w['wall_info'])[:,0]
        x_min = ss[0] 
        y_min = ss[1]
        z_min = ss[2]
        x_max = ss[3]
        y_max = ss[4]
        z_max = ss[5]

        Z = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
            ])

        # list of faces
        faces = [
             [Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]], 
             [Z[0],Z[1],Z[5],Z[4]], 
             [Z[2],Z[3],Z[7],Z[6]], 
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]]
             ]

        ls = np.array([
            [Z[0], Z[1]],
            [Z[1], Z[2]],
            [Z[2], Z[3]],
            [Z[3], Z[0]],

            [Z[4], Z[5]],
            [Z[5], Z[6]],
            [Z[6], Z[7]],
            [Z[7], Z[4]],

            [Z[0], Z[4]],
            [Z[1], Z[5]],
            [Z[2], Z[6]],
            [Z[3], Z[7]]
            ])

        lc = Line3DCollection(ls, linewidths=wall_linewidth, colors='k', alpha=wall_alpha)
        ax.add_collection(lc)

    if not args.nocolorbar:
        fig.colorbar(forcolorbar)
    if not args.nogrid:
        plt.grid()
    if args.xlim:
        xx = args.xlim.split(' ')
        ax.set_xlim3d([float(xx[0]), float(xx[1])])
    if args.ylim:
        yy = args.ylim.split(' ')
        ax.set_ylim3d([float(yy[0]), float(yy[1])])
    if args.zlim:
        zz = args.zlim.split(' ')
        ax.set_zlim3d([float(zz[0]), float(zz[1])])


    mx = np.amax(np.abs(ss))
    XYZlim = [-mx, mx]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_box_aspect((1, 1, 1))

    if args.motion:
        view_az = args.view_az + args.motion_az_stepsize * (t - fc)
        view_angle = args.view_angle + args.motion_angle_stepsize * (t - fc)
        ax.view_init(view_az, view_angle)
    else:
        ax.view_init(args.view_az, args.view_angle)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

if args.serial:
    ## serial j
    for t in range(fc, lc+1):
        genplot(t)
else:
    ## parallel
    a_pool = Pool()
    a_pool.map(genplot, range(fc, lc+1))
    a_pool.close()

