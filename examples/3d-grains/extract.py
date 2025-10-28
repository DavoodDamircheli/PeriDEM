import numpy as np
import h5py
import re
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import time

import argparse
parser = argparse.ArgumentParser(description='Optional app description')
# Optional argument
parser.add_argument('--data_dir', type=str, help='input data directory', default='output/hdf5/')
parser.add_argument('--modulo', type=int, help='downsample data points', default=1)

parser.add_argument('--output_file', type=str, help='output file with quantities')

parser.add_argument('--fc', type=int, help='first counter')
parser.add_argument('--lc', type=int, help='last counter')

parser.add_argument('--wheel_ind', type=int, help='index of wheel particle', default=0)
parser.add_argument('--serial', action='store_true', help='read timesteps in serial')
# finish parsing
args = parser.parse_args()


if args.output_file:
    output_file = args.output_file
else:
    output_file = args.data_dir+'/quantities.h5'

setup_file = args.data_dir+'/setup.h5'
# plot info
plotinfo_file = args.data_dir+'/plotinfo.h5'
p = h5py.File(plotinfo_file, "r")
dt = float(p['dt'][0])
modulo = int(p['modulo'][0])
if args.fc:
    fc = args.fc
else:
    fc = int(p['f_l_counter'][0])
if args.lc:
    lc = args.lc
else:
    lc = int(p['f_l_counter'][1])

tt = range(fc, lc+1, args.modulo)

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

class RefSetup(object):
    """Reference setup"""
    def __init__(self, setup_filename, wheel_ind):
        self.setup_filename = setup_filename
        self.wheel_ind = wheel_ind
        self.orig_f = h5py.File(self.setup_filename, "r")

        # precomputed reference quantities
        self.wheel_mean_pos = self.get_mean_Pos(self.wheel_ind)
        self.wheel_vol = self.get_volume(self.wheel_ind)

    def get_volume(self, index):
        name = ('P_%05d' % index)
        return np.array(self.orig_f[name+'/Vol'])

    def get_mean_Pos(self, index):
        name = ('P_%05d' % index)
        ref_pos = np.array(self.orig_f[name+'/Pos'])
        return np.mean(ref_pos, axis=0)

    
    def get_moment_of_inertia(self, index, include_density=False):
        """ Return the moment of inertia, divided the rho
        i.e. int r^2 dA
        """
        name = ('P_%05d' % index)
        vol = np.array(self.orig_f[name+'/Vol'])
        pos = np.array(self.orig_f[name+'/Pos'])
        centroid = np.mean(pos, axis=0)
        r_vec = pos - centroid
        r = np.sqrt( np.sum(r_vec**2, axis=1) )
        # out = np.sum((r**2) * vol)
        out = np.sum((r**2) * np.squeeze(np.asarray(vol)))
        if include_density:
            rho = np.array(self.orig_f[name+'/rho'])
            out *= rho
        return out

class QuantitySingleTime(object):
    """quantities associated with per timestep"""
    def __init__(self):
        self.time         = 0
        self.index        = 0
        self.vel_x        = 0
        self.vel_y        = 0
        self.vel_z        = 0
        self.vel_angular  = 0
        # self.slip         = 0 # moved to a derived quantity in `multiplot.py`
        self.acc_x        = 0
        self.acc_y        = 0
        self.disp_x       = 0
        self.disp_y       = 0
        self.currpos_x    = 0
        self.currpos_y    = 0

        self.applied_fd_x = 0
        self.applied_fd_y = 0
        self.applied_torque_density   = 0

        # requires accessing reference setup
        self.disp_x        = 0
        self.disp_y        = 0
        self.applied_f_x = 0
        self.applied_f_y = 0
        self.applied_torque = 0

class QuantityTimeSeries(object):
    """Convert an array of single-time data to timeseries data"""
    # def __init__(self, arrQT, refSetup):
    def __init__(self, arrQT, filename):

        self.filename = filename

        with h5py.File(filename, "w") as f:
            f.create_dataset('index',        data=np.array([QT.index        for QT in arrQT]) )
            f.create_dataset('time',         data=np.array([QT.time        for QT in arrQT]) )
            f.create_dataset('vel_x',        data=np.array([QT.vel_x        for QT in arrQT]) )
            f.create_dataset('vel_y',        data=np.array([QT.vel_y        for QT in arrQT]) )
            f.create_dataset('vel_z',        data=np.array([QT.vel_z        for QT in arrQT]) )
            f.create_dataset('vel_angular',  data=np.array([QT.vel_angular  for QT in arrQT]) )
            # f.create_dataset('slip',         data=np.array([QT.slip         for QT in arrQT]) )
            f.create_dataset('acc_x',        data=np.array([QT.acc_x        for QT in arrQT]) )
            f.create_dataset('acc_y',        data=np.array([QT.acc_y        for QT in arrQT]) )
            f.create_dataset('currpos_x',    data=np.array([QT.currpos_x    for QT in arrQT]) )
            f.create_dataset('currpos_y',    data=np.array([QT.currpos_y    for QT in arrQT]) )

            f.create_dataset('applied_fd_x', data=np.array([QT.applied_fd_x for QT in arrQT]) )
            f.create_dataset('applied_fd_y', data=np.array([QT.applied_fd_y for QT in arrQT]) )
            f.create_dataset('applied_torque_density',   data=np.array([QT.applied_torque_density   for QT in arrQT]) )

            # requires accessing reference setup
            f.create_dataset('disp_x',        data=np.array([QT.disp_x        for QT in arrQT]) )
            f.create_dataset('disp_y',        data=np.array([QT.disp_y        for QT in arrQT]) )
            f.create_dataset('applied_f_x', data=np.array([QT.applied_f_x for QT in arrQT]) )
            f.create_dataset('applied_f_y', data=np.array([QT.applied_f_y for QT in arrQT]) )
            f.create_dataset('applied_torque',   data=np.array([QT.applied_torque   for QT in arrQT]) )

# Reference setup
RS = RefSetup(setup_file, args.wheel_ind)

def computeSingleTime(t):
    # print(t)
    tc_ind = ('%05d' % t)
    rem = t % (int((lc-fc)/5))
    if rem == 1:
        print(t, end=' ', flush=True)

    filename = args.data_dir+'/tc_'+tc_ind+'.h5'
    setup_filename = args.data_dir+'/setup.h5'

    f = h5py.File(filename, "r")

    QT = QuantitySingleTime()

    name = ('P_%05d' % args.wheel_ind)

    # time, index
    QT.index = t
    QT.time = np.array(t) * dt * modulo

    # currpos
    currpos = np.array(f[name+'/CurrPos'])
    centroid = np.mean(currpos, axis=0)
    QT.currpos_x = centroid[0]
    QT.currpos_y = centroid[1]

    QT.disp_x = centroid[0] - RS.wheel_mean_pos[0]
    QT.disp_y = centroid[1] - RS.wheel_mean_pos[0]

    # vel
    vel = np.array(f[name+'/vel'])
    vel_mean = np.mean(vel, axis=0)
    QT.vel_x =  vel_mean[0]
    QT.vel_y = vel_mean[1]
    QT.vel_z = vel_mean[2]

    # acc
    acc = np.array(f[name+'/acc'])
    acc_mean = np.mean(acc, axis=0)

    QT.acc_x = acc_mean[0]
    QT.acc_y = acc_mean[1]

    if 0:
        # applied_fd
        applied_fd = np.array(f[name+'/applied_force_density'])
        app_fd_tot = np.sum(applied_fd, axis=0)

        QT.applied_fd_x = app_fd_tot[0]
        QT.applied_fd_y = app_fd_tot[1]

        applied_f = applied_fd * RS.wheel_vol
        app_f_tot = np.sum(applied_f, axis=0)
        QT.applied_f_x = app_f_tot[0]
        QT.applied_f_y = app_f_tot[1]

        ## Moved to slip to a derived quantity  in `multiplot.py`


    # angular stuff
    r_vec = (currpos - centroid)
    r = np.sqrt( np.sum(r_vec**2, axis=1) )
    r_dir = r_vec / r[:,None]
    # angular velocity
    w_tot = 0
    # R = 0 # radius of the body
    for j in range(len(currpos)):
        # r_vec = (currpos[j] - centroid)
        # r = np.sqrt( np.sum(r_vec**2) )
        # R = max(R, r)
        # r_dir = r_vec / r
        # velocity on the frame of reference at the centroid
        v = vel[j] - vel_mean
        v_r = np.dot(v, r_dir[j]) * r_dir[j]
        v_t = v - v_r
        v_t_norm = np.sqrt(np.sum(v_t**2))
        w_this = v_t_norm/r[j]

        w_tot += w_this
    w_mean = w_tot/len(currpos)
    QT.vel_angular =  w_mean

    

    if 0:
        # total applied torque to maintain rotation
        torq_tot_density = 0
        torq_tot = 0
        for j in range(len(currpos)):
            # r_vec = (currpos[j] - centroid)
            # r = np.sqrt( np.sum(r_vec**2) )
            # r_dir = r_vec / r
            # velocity on the frame of reference at the centroid
            # v = vel[j] - vel_mean
            af = applied_fd[j]
            f_r = np.dot(af, r_dir[j]) * r_dir[j]
            f_t = af - f_r # in the tangential direction

            # f_t_norm = np.sqrt(np.sum(f_t**2))
            # torq_this = f_t_norm * r[i]

            ## computing r \cross t_t
            torq_this = np.cross(r[j]*r_dir[j], f_t)


            torq_tot_density += torq_this
            torq_tot += torq_this * RS.wheel_vol[j]

        QT.applied_torque_density = torq_tot_density
        QT.applied_torque = torq_tot

    return QT


arrQT = []
if args.serial:
    # serial j
    for t in tt:
        arrQT.append(computeSingleTime(t))
else:
    ## parallel
    a_pool = Pool()
    arrQT = a_pool.map(computeSingleTime, tt)
    a_pool.close()


# Converting quntity-series data to timeseries data
print('saving to', output_file)
QS = QuantityTimeSeries(arrQT, output_file)
