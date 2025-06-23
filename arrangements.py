import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
# to plot a collection of lines
from matplotlib.collections import LineCollection
from gekko import GEKKO

# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import combinations
# from functools import partial
import time
from random import seed
from random import random

import shape_dict
import material_dict
from genmesh import genmesh

class IncenterNodes(object):
    """Incenter and radius of incircles"""
    def __init__(self, incenter, incircle_rad, mesh):
        self.incenter = incenter
        self.incircle_rad = incircle_rad
        self.mesh = mesh

    def dim(self):
        # return len(self.mesh.pos[0])
        return len(self.incenter[0])

    def count(self):
        return len(self.incircle_rad)

    def range(self):
        """min and max of the radius
        """
        return [np.amin(self.incircle_rad), np.amax(self.incircle_rad)]

    def mean(self):
        """ mean 
        """
        return np.mean(self.incircle_rad)

    def std(self):
        """standard deviation
        """
        return np.std(self.incircle_rad)

    def total_volume(self):
        """Total volume of all the bounding polygon
        """
        return np.sum(self.mesh.vol)

    def occupied_volume(self):
        """Total volume of all the circles combined
        :returns: TODO

        """
        dimension = self.dim()
        if (dimension ==2):
            return np.sum(np.pi * self.incircle_rad**2)
        else:
            return np.sum(4 * np.pi/3 * self.incircle_rad**3)

    def packing_ratio(self):
        """Ratio of occupied_volume to total polygon volume
        """
        return self.occupied_volume() / self.total_volume()

    def info(self):
        """Prints info about the arrangement
        """
        print('Count: ', self.count())
        print('Range: ', self.range())
        print('Mean: ', self.mean())
        print('Standard deviation: ', self.std())
        print('Packing ratio: ', self.packing_ratio())

    def plot_distribution(self, filename='size_distribution.png', bins=None):
        """plot the distribution of particle sizes
        """
        plt.hist(self.incircle_rad, bins=bins)
	# plt.title('Particle size distribution')
	# plt.xlabel('Radius')
	# plt.ylabel('Number')
        # save the image
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        


    ## Delete beyond min and max or create two methods for min and max
    def trim(self, min_rad = None, max_rad = None):
        """Drop elements that are beyond min and max radius
        :min_rad: TODO
        :max_rad: TODO
        """

        # Todo: merge two operations into a single for loop
        if min_rad is not None:
            del_list = []
            for i in range(len(self.incircle_rad)):
                if ( self.incircle_rad[i] < min_rad ):
                    del_list.append(i)
            self.incenter = np.delete(self.incenter, del_list, 0)
            self.incircle_rad = np.delete(self.incircle_rad, del_list, 0)

        if max_rad is not None:
            del_list = []
            for i in range(len(self.incircle_rad)):
                if ( self.incircle_rad[i] > max_rad ):
                    del_list.append(i)
            self.incenter = np.delete(self.incenter, del_list, 0)
            self.incircle_rad = np.delete(self.incircle_rad, del_list, 0)

    def plot(self, do_plot = True, plot_edge = True, plot_mesh = True, remove_axes = True, save_file = None, plot_circles=True):
        """TODO: Docstring for plot.
        :line: plot lines
        """

        # compute the dimension
        dimension = self.dim()

        if plot_circles:
            if (dimension==2):
                for i in range(len(self.incircle_rad)):
                    t = np.linspace(0, 2*np.pi, num = 50)
                    plt.plot( self.incenter[i,0] + self.incircle_rad[i] * np.cos(t), self.incenter[i,1] + self.incircle_rad[i] * np.sin(t), 'k-', linewidth = 0.5)
            else:
                pass


        if plot_mesh:
            if (dimension ==2):
                edges = self.mesh.get_edges()
                V1 = edges[:,0]
                V2 = edges[:,1]

                P1 = self.mesh.pos[V1]
                P2 = self.mesh.pos[V2]
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 

                lc = LineCollection(ls, linewidths=0.5, colors='b')
                plt.gca().add_collection(lc)
            else:
                pass

        if plot_edge:
            if (dimension == 2):
                P1 = self.mesh.pos[self.mesh.bdry_edges[:,0]]
                P2 = self.mesh.pos[self.mesh.bdry_edges[:,1]]
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 

                lc = LineCollection(ls, linewidths=0.5, colors='r')
                plt.gca().add_collection(lc)
            else:
                pass

        if remove_axes:
            if (dimension == 2):
                plt.gca().set_axis_off()
            else:
                pass

        # adjust plot properties
        if (dimension ==2):
            # scale the axes
            plt.axis('scaled')
        else:
            pass
        

        if save_file:
            # save the image
            plt.savefig(save_file, dpi=300, bbox_inches='tight')

        if do_plot:
            if (dimension ==2):
                plt.show()
            else:
                print('Plotting 3D arrangement not implemented yet')

def get_nbd_vertex_elem(T, total_vertices):
    """ From the element (facet) array, compute the facet-neighborhood array of the vertices

    :T: vertex indices of all elements
    :returns: neighboring facet indices of all vertices

    """

    full_list = []

    for i in range(total_vertices):
        # vertex i won't appear more than once in a single row
        row_list = np.where(T == i)[0]
        full_list.append(row_list.tolist())

    return full_list

def get_incenter_mesh_loc(P, meshsize, dimension = 2, msh_file = None, modify_nodal_circles = True, gen_bdry = True, min_rad = None, max_rad = None):
    """Given a polygon, generate particle locations and radii that fit within the polygon wall via the computation of the incenter

    :P: The vertices of a polygon, list
    :meshsize: 
    :min_rad: drop any radius below this
    :max_rad: drop any radius above this
    :returns: a class that contains the location of the centers, and the radii
    """
    # generate mesh
    # mesh = genmesh(P, meshsize, msh_file = msh_file, do_plot = False, dimension = dimension)
    mesh = genmesh(P, meshsize, msh_file = msh_file, dimension = dimension)


    if (dimension ==2):
        # vertices
        v_a = mesh.pos[mesh.T[:,0]]
        v_b = mesh.pos[mesh.T[:,1]]
        v_c = mesh.pos[mesh.T[:,2]]

        ab = v_b - v_a
        ac = v_c - v_a
        bc = v_c - v_b

        # lengths of the sides
        l_a = np.sqrt(np.sum( (bc)**2, axis = 1))[:, None]
        l_b = np.sqrt(np.sum( (ac)**2, axis = 1))[:, None]
        l_c = np.sqrt(np.sum( (ab)**2, axis = 1))[:, None]

        sum_l = l_a + l_b + l_c

        # print('l_a :', l_a.shape)
        # print('v_a :', v_a.shape)
        
        # incenter
        incenter = (l_a * v_a + l_b * v_b + l_c * v_c) / sum_l

        # print('incenter :', incenter.shape)
        # time.sleep(5.5)


        # area of the triangles
        T_area = 0.25 * np.sqrt(sum_l * (l_a-l_b+l_c) * (l_b-l_c+l_a) * (l_c-l_a+l_b) )

        ## another approach for area of the triangles
        # u_x = mesh.pos[mesh.T[:,1], 0] - mesh.pos[mesh.T[:,0], 0]
        # u_y = mesh.pos[mesh.T[:,1], 1] - mesh.pos[mesh.T[:,0], 1]
        # v_x = mesh.pos[mesh.T[:,2], 0] - mesh.pos[mesh.T[:,0], 0]
        # v_y = mesh.pos[mesh.T[:,2], 1] - mesh.pos[mesh.T[:,0], 1]
        # cp = 0.5 * abs(u_x * v_y - u_y * v_x)
        # print(T_area - np.array([cp]).transpose())

        # incircle radius
        incircle_rad = 2 * T_area / sum_l
    else:
        # dim = 3

        print('Total tetrahedrons: ', len(mesh.T))

        x1 = mesh.pos[mesh.T[:, 0], 0]
        y1 = mesh.pos[mesh.T[:, 0], 1]
        z1 = mesh.pos[mesh.T[:, 0], 2]

        x2 = mesh.pos[mesh.T[:, 1], 0]
        y2 = mesh.pos[mesh.T[:, 1], 1]
        z2 = mesh.pos[mesh.T[:, 1], 2]

        x3 = mesh.pos[mesh.T[:, 2], 0]
        y3 = mesh.pos[mesh.T[:, 2], 1]
        z3 = mesh.pos[mesh.T[:, 2], 2]

        x4 = mesh.pos[mesh.T[:, 3], 0]
        y4 = mesh.pos[mesh.T[:, 3], 1]
        z4 = mesh.pos[mesh.T[:, 3], 2]

        T_vol = np.abs( 1/6*( (x4-x1) * ((y2-y1)*(z3-z1)-(z2-z1)*(y3-y1)) + (y4-y1) * ((z2-z1)*(x3-x1)-(x2-x1)*(z3-z1)) + (z4-z1) * ((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)) ) )

        alpha = (T_vol * 6)[:, None]

        # print('alpha shape: ', alpha.shape)

        # copied from sage
        denom = np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2) + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2) + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2) + np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)
        
        # convert to (n,1) array from (n,)
        denom  = denom[:,None]

        # print('denom shape: ', denom.shape)

        numer = np.zeros((len(denom), 3))

        # print('numer shape: ', numer.shape)

        numer[:,0] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*x1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*x2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*x3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*x4

        numer[:,1] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*y1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*y2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*y3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*y4

        numer[:,2] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*z1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*z2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*z3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*z4

        incenter = numer / denom
        incircle_rad = alpha / denom


    ## Add entries for nodes, exclude the boundary nodes
    # nodal_rads = []
    # nodal_centers = []
    # neighboring triangles of each node
    nbd_node_tri = get_nbd_vertex_elem(mesh.T, len(mesh.pos))

#######################################################################
# Parallel attempt

#######################################################################
    
    if modify_nodal_circles:
        m_arr = []
        for i in range(len(mesh.pos)):
            nbd_elems = nbd_node_tri[i]
            max_fitting_rad = np.amin(np.sqrt(np.sum( (incenter[nbd_elems] - mesh.pos[i])**2, axis = 1)) - incircle_rad[nbd_elems].transpose())
            if i not in set(mesh.bdry_nodes):
                # gekko stuff
                m = GEKKO(remote=False)
                m.options.SOLVER = 1
                m.options.DIAGLEVEL = 0
                ## solving maximin problem
                x1,x2,Z = m.Array(m.Var,3)
                # x1, x2, Z = [m.Var() for i in range(3)]
                x1.value = mesh.pos[i][0]
                x2.value = mesh.pos[i][1]
                m.Maximize(Z)
                # m.Equation(x1+x2+x3==15)
                m.Equation( x1 >= np.min( incenter[nbd_elems][:,0] ) )
                m.Equation( x1 <= np.max( incenter[nbd_elems][:,0] ) )
                m.Equation( x2 >= np.min( incenter[nbd_elems][:,1] ) )
                m.Equation( x2 <= np.max( incenter[nbd_elems][:,1] ) )
                for k in range(len(nbd_elems)):
                    nbd_id = nbd_elems[k]
                    rk = incircle_rad[nbd_id][0]
                    pk1 = incenter[nbd_id][0]
                    pk2 = incenter[nbd_id][1]
                    m.Equation( Z <= ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 - rk )
                    m.Equation( ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 >= rk )
                # m.solve()
                m_arr.append(m)

                ## add to the list
                # nodal_centers.append(np.array([x1.value[0], x2.value[0]]))
                # nodal_rads.append(Z.value[0])

                ## add to the list: the Original node
                # nodal_rads.append(max_fitting_rad)
                # nodal_centers.append(mesh.pos[i])
            else:
                if gen_bdry:
                    # gekko stuff
                    m = GEKKO(remote=False)
                    m.options.SOLVER = 1
                    m.options.DIAGLEVEL = 0
                    ## solving maximin problem
                    x1,x2,Z = m.Array(m.Var,3)
                    m.Maximize(Z)
                    for k in range(len(nbd_elems)):
                        nbd_id = nbd_elems[k]
                        rk = incircle_rad[nbd_id][0]
                        pk1 = incenter[nbd_id][0]
                        pk2 = incenter[nbd_id][1]
                        m.Equation( Z <= ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 - rk )
                        m.Equation( ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 >= rk )

                    ## get boundary edges associated with the boundary nodes
                    for e in range(len(mesh.bdry_edges)):
                        edge = mesh.bdry_edges[e]
                        if i in set(edge):
                            e2 = mesh.pos[edge[1]]
                            e1 = mesh.pos[edge[0]]

                            u = e2 - e1
                            u_norm = (u[0]**2 + u[1]**2)**0.5

                            # cross_prod = np.abs( u[0]*(x2 - e2) - u[1]*(x1 - e1) )
                            # normal distance to the edge: h = |uxv|/|u|, u=e2-e1, v=x-e1
                            m.Equation( Z <= np.abs( u[0]*(x2 - e1[1]) - u[1]*(x1 - e1[0]) ) / u_norm)

                    # initial guess, the centroid
                    all_x1 = np.append(incenter[nbd_elems][:,0], mesh.pos[i][0])
                    all_x2 = np.append(incenter[nbd_elems][:,1], mesh.pos[i][1])

                    x1.value = np.mean(all_x1)
                    x2.value = np.mean(all_x2)

                    # generic bounds
                    m.Equation( x1 >= np.min(all_x1) )
                    m.Equation( x1 <= np.max(all_x1) )
                    m.Equation( x2 >= np.min(all_x2) )
                    m.Equation( x2 <= np.max(all_x2) )
                    # m.solve()
                    m_arr.append( m )

                    ## add to the list
                    # nodal_centers.append(np.array([x1.value[0], x2.value[0]]))
                    # nodal_rads.append(Z.value[0])


        # print(m_arr)
        start = time.time()

        ## Solve all at once in parallel
        a_pool = Pool()
        # cl_solved_arr = a_pool.map(solve_par, m_arr)
        print('Modifying nodal circles in parallel.')
        cl_solved_arr = np.array(a_pool.map(solve_par, m_arr))
        print('time taken ', time.time() - start)

        nodal_centers = cl_solved_arr[:,0:2]
        nodal_rads = cl_solved_arr[:,2]

        ## Add to the main array
        incircle_rad = np.append(incircle_rad, np.array([nodal_rads]).transpose(), axis = 0)
        incenter = np.append(incenter, nodal_centers, axis = 0)

    else:   # if there is no modification to the nodal circles
        nodal_rads = []
        nodal_centers = []
        for i in range(len(mesh.pos)):
            if i not in set(mesh.bdry_nodes):
                nbd_elems = nbd_node_tri[i]
                max_fitting_rad = np.amin(np.sqrt(np.sum( (incenter[nbd_elems] - mesh.pos[i])**2, axis = 1)) - incircle_rad[nbd_elems].transpose())

                # add to the list: the Original node
                nodal_rads.append(max_fitting_rad)
                nodal_centers.append(mesh.pos[i])

        print('Adding for each node: ', len(nodal_rads))

        # append to the main array
        incircle_rad = np.append(incircle_rad, np.array([nodal_rads]).transpose(), axis = 0)
        incenter = np.append(incenter, nodal_centers, axis = 0)


    return IncenterNodes(incenter, incircle_rad, mesh)


def solve_par(m):
    """ Solving gekko optimization problem (to be called in parallel)
    """
    m.solve(disp=False)
    # print(m._variables)
    ## outputs a row vector, rank 1
    solved_var = np.array(m._variables).transpose()[0]
    # print(solved_var)
    return solved_var
    
    # nodal_centers.append(np.array([solved_var[0][0], cl_solved_arr[i][1][0]]))
    # nodal_rads.append(cl_solved_arr[i][2][0])
    # return m._variables
    # return np.array([m_arr[i].x1.value[0], m_arr[i].x2.value[0], m_arr[i].Z.value[0]])

