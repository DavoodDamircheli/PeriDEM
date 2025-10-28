import numpy as np
import h5py
import argparse
from pathos.multiprocessing import ProcessingPool as Pool
import sys
import matplotlib.pyplot as plt

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--path', type=str, help='path to store output files', default='out/')
parser.add_argument('--noplot', action='store_true', help='do not plot')
parser.add_argument('--fc', type=int, help='first counter', default=1)
parser.add_argument('--lc', type=int, help='last counter', default=10)
parser.add_argument('--serial', action='store_true', help='read timesteps in serial')
args = parser.parse_args()

def get_reaction(t):
    print(t)
    tc_ind = ('%05d' % t)
    wall_filename = args.path+'/wall_'+tc_ind+'.h5'

    w = h5py.File(wall_filename, "r")
    ss = np.array(w['wall_info'])[:,0]
    reaction = np.array(w['reaction'])
    # print('reaction', reaction)

    # which value?
    # For 3d, the order is # x_min # y_min # z_min # x_max # y_max # z_max

    # norm of the wall forces
    r_walls = [np.linalg.norm(reaction[face]) for face in range(6)]

    # r_walls = reaction[5][2]
    # r_walls = np.linalg.norm(reaction[3])

    # return reaction
    return r_walls


def get_reaction_all():
    this_range = range(args.fc, args.lc+1)
    reaction_list = []
    if args.serial:
        for t in this_range:
            reaction_list.append(get_reaction(t))
    else:
        a_pool = Pool()
        reaction_list = a_pool.map(get_reaction, this_range)
        a_pool.close()

    print('reaction', reaction_list)

    # rl = [ val[2] for val in reaction_list]
    # plt.plot(rl)
    plt.plot(reaction_list, label=range(len(reaction_list[0])))
    # plt.plot(reaction_list[0:30])
    # plt.axis('scaled')
    plt.legend()
    plt.grid()
    filename = args.path + '/wall_reaction.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    



    
if __name__ == "__main__":
    get_reaction_all()
