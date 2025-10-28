import numpy as np
import h5py
import re
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import time

from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from matplotlib.collections import LineCollection

import seaborn as sns
sns.set()

import sys, os
sys.path.append(os.getcwd())
from figsize import set_size
from matplotlib.pyplot import figure
fig_size = set_size('amsart')

import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# input
parser.add_argument('-p','--parent', help='parent path')
parser.add_argument('--subdirs', nargs='+', help='list of subdirectories', required=True)
parser.add_argument('--labels', nargs='+', help='list of labels corresponding to each subdirectory')
parser.add_argument('--quantity', type=str, help='quantity to plot', default='vel_angular')
parser.add_argument('--title', type=str, help='title of the plot')
parser.add_argument('--paramstr', action='store_true', help='treat input parameter list as strings')
parser.add_argument('--xquantity', type=str, help='quantity representing the x axis: time, index, etc', default='index')
# output
parser.add_argument('--img_dir', type=str, help='output image directory', default='output/img/')
parser.add_argument('--prefix', type=str, help='a prefix for the output filename', default='')
# sampling
parser.add_argument('--modulo', type=int, help='downsample data points', default=1)
parser.add_argument('--fc', type=int, help='first counter')
parser.add_argument('--lc', type=int, help='last counter')
# statistics
parser.add_argument('--smoothing', action='store_true', help='smooth out the data')
parser.add_argument('--smoothing_window', type=int, help='window size to smooth over', default=2)
parser.add_argument('--cumulative', action='store_true', help='cumulative plot')
parser.add_argument('--gradient', action='store_true', help='gradient plot')
parser.add_argument('--average_repeated_xlabels', action='store_true', help='group repeated xlabels by averaging')
# parser.add_argument('--regression_only', action='store_true', help='only the quantity after regression')
parser.add_argument('--regression', action='store_true', help='first timestep for regression')
# time-average plot
parser.add_argument('--timeavg', action='store_true', help='compress timeseries data by averaging')
parser.add_argument('--timeavg_xlabel', type=str, help='label of x-axis when --timeavg used', default='Parameter')
parser.add_argument('--timeavg_drawline', action='store_true', help='Draw a line joining the --timeavg points')
parser.add_argument('--timeavg_drawline_style', type=str, help='line stype', default='--o')

parser.add_argument('--timeavg_std', action='store_true', help='Reduce data dimension by computing the standard derivation, not by averaging. Use with --timeavg.')

parser.add_argument('--nolegend', action='store_true', help='Do not draw legends')
parser.add_argument('--notitle', action='store_true', help='Do not draw titles')
parser.add_argument('--noset_figsize', action='store_true', help='Let python decide the size of the fig')

# other
parser.add_argument('--wheel_ind', type=int, help='index of wheel particle', default=0)
parser.add_argument('--plot', action='store_true', help='whether show plot or not')
parser.add_argument('--serial', action='store_true', help='read timesteps in serial')

# finish parsing
args = parser.parse_args()

if args.parent:
    data_dirs = [args.parent+'/'+d_dir for d_dir in args.subdirs]
if args.labels:
    labels = args.labels 
else:
    labels = args.subdirs

latexlabels={
        'vel_x':  r'$v_x(ms^{-1})$',
        'vel_y':  r'$v_y(ms^{-1})$',
        'vel_z':  r'$v_z(ms^{-1})$',
        'acc_x':  r'$a_x(ms^{-2})$',
        'acc_y':  r'$a_y(ms^{-2})$',
        'vel_angular':  r'$\omega (s^{-1})$',
        'disp_x': r'$u_x(m)$',
        'disp_y': r'$u_y(m)$',
        'currpos_x': r'$p_x(m)$',
        'currpos_y': r'$p_y(m)$',
        'slip':  'Slip',
        'applied_fd_x':  r'$f_x (Nm^{-2})$',
        'applied_fd_y':  r'$f_y(Nm^{-2})$',
        'applied_torque_density':  r'$\tau (Nm^{-1})$',
        'applied_f_x':  r'$F_x (Nm^{-2})$',
        'applied_f_y':  r'$F_y(N)$',
        'applied_torque':  r'$\tau (Nm)$',
        'scaled_angular_power':  r'$P^{ang} (Nm^{-1}t^(-1})$',
        'cumulative_torque_density':  r'$\tau (Nm^{-1})$',
        'time':  'Time (s)',
        'index':  'Time step',
        'bulk_damage':  'Bulk damage',
        }

titles={
        'vel_x':                     'x velocity of wheel',
        'vel_y':                     'y velocity of wheel',
        'vel_z':                     'z velocity of wheel',
        'acc_x':                     'Horizontal acceleration of wheel',
        'acc_y':                     'Vertical acceleration of wheel',
        'vel_angular':               'Angular velocity of wheel',
        'disp_x':                    'Horizontal displacement of wheel',
        'disp_y':                    'Vertical displacement of wheel',
        'currpos_x':                 'Horizontal position of wheel',
        'currpos_y':                 'Vertical position of wheel',
        'slip':                      'Wheel slip',
        'applied_fd_x':              'Applied horizontal force density on wheel',
        'applied_fd_y':              'Applied vertical force density on wheel',
        'applied_torque_density':    'Applied torque density on wheel',
        'applied_f_x':              'Applied horizontal force on wheel',
        'applied_f_y':              'Applied vertical force on wheel',
        'applied_torque':    'Applied torque on wheel',
        'scaled_angular_power':      'Scaled angular power of wheel',
        'cumulative_torque_density': 'Cumulative torque density',
        }

extra_titlestr = ""
if args.cumulative:
    extra_titlestr = " (cumulative)"
if args.regression:
    extra_titlestr = " (with fit)"
if args.regression:
    extra_titlestr = " (smoothened)"

# image annotation
def get_flag(name):
    path = "img_annotations/{}.png".format(name.title())
    im = plt.imread(path)
    return im


print('Plotting quantity:', args.quantity)

def movingaverage(array, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(array, window, 'same')

data_timeavg_x = []
data_timeavg_y = []



if args.noset_figsize:
    pass
else:
    figure(figsize=fig_size)

# plot
for i,data_dir in enumerate(data_dirs):

    if args.quantity in ['bulk_damage']:
        filename = data_dir+'/graveldata.h5'
    else:
        filename = data_dir+'/quantities.h5'

    f=h5py.File(filename, "r")

    xd = np.array(f[args.xquantity])

    ## derived quantities
    if args.quantity == 'slip':
        v_x = np.array(f['vel_x'])
        w = np.array(f['vel_angular'])
        f_wh = open(data_dir+'/wheel_rad', "r")
        R = float(f_wh.read())
        f_wh.close()


        # in base.conf, the prescribed_angular_vel is in the counterclockwise direction
        # whereas, extract.py computes the angular velocity in the clockwise direction
        wR = w*R

        #print(wR)
        #print(v_x)

        yd = np.zeros(len(xd))
        for idx in range(len(v_x)):
            if wR[idx] == 0:
                # technically, in this case, the slip should be infinity
                yd[idx] = 0
            else:
                yd[idx] = (wR[idx] - v_x[idx])/wR[idx]
    else:
        yd = np.array(f[args.quantity])


    # fix for applied force etc: negative to be consistent with [smiths et al]
    if args.quantity in ['applied_fd_x', 'applied_fd_y', 'applied_torque_density', 'applied_f_x', 'applied_f_y', 'applied_torque']:
        yd = -yd

    if args.fc:
        fc = args.fc
    else:
        fc = 0
    if args.lc:
        lc = args.lc
    else:
        lc = len(xd)

    xd = xd[fc:lc+1]
    yd = yd[fc:lc+1]

    # statistics on data
    if args.smoothing:
        yd = movingaverage(yd, args.smoothing_window)
    if args.gradient:
        yd = np.gradient(yd, xd)
    if args.cumulative:
        yd = np.cumsum(yd)
    # if args.regression_only:
    #     slope, intercept, r_value, p_value, std_err = linregress(xd, yd)
    #     yd = slope * xd + intercept


    if args.timeavg:
        if args.paramstr:
            xp = i
        else:
            xp = float(labels[i])

        if args.timeavg_std:
            yp = np.std(yd)
        else:
            yp = np.mean(yd)

        data_timeavg_x.append(xp)
        data_timeavg_y.append(yp)

        plt.scatter(xp, yp, label=labels[i])
        plt.xlabel(args.timeavg_xlabel)
    else:
        plt.plot(xd, yd, label=labels[i])
        plt.xlabel(latexlabels[args.xquantity])

    if args.regression:
        slope, intercept, r_value, p_value, std_err = linregress(xd, yd)
        regr = slope * xd + intercept
        plt.plot(xd, regr, '--', label=labels[i]+' fit')

if args.timeavg:
    if args.timeavg_drawline:
        plt.plot(data_timeavg_x, data_timeavg_y, args.timeavg_drawline_style)

if args.notitle:
    pass
else:
    if args.title:
        plt.title(r'%s' % title)
    else:
        plt.title(titles[args.quantity] + extra_titlestr)
 
plt.ylabel(latexlabels[args.quantity])

if args.paramstr:
    xtick_loc = range(len(labels))
    xtick_str = [str(label) for label in labels]
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(xtick_loc, xtick_str)

if args.nolegend:
    pass
else:
    plt.legend()

out_png = args.img_dir+'/'+args.prefix+'_'+args.quantity+'.png'

print('Saving img to', out_png)
plt.savefig(out_png, dpi=300, bbox_inches='tight')

if args.plot:
    plt.show()
