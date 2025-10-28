import pygmsh
import meshio
import numpy as np
import math 

import gmsh
import sys
from scipy.spatial import ConvexHull

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import pdb
#pdb.set_trace()
class Shape:
    """Returns the boundary nodes and the nonconvex_interceptor"""
    def __init__(self, P, nonconvex_interceptor=None, msh_file = None, pygmsh_geom=None, scale_mesh_to=None, centroid_origin=False):
        self.P = P
        self.pygmsh_geom = pygmsh_geom
        self.nonconvex_interceptor = nonconvex_interceptor
        self.msh_file = msh_file
        self.scale_mesh_to = scale_mesh_to
        self.centroid_origin = centroid_origin

    def plot(self, bdry_arrow=True, extended_bdry=True, angle_bisector=True, plot_bounding_ball=False, bounding_ball_rad=1e-3, bounding_ball_steps=30, plot_included_ball=False, included_ball_rad=0.5e-3, included_ball_steps=30):
        """Plot the shape
        :returns: TODO

        """
        if (self.P is not None):
            dim = len(self.P[0])
            print('dim = ', dim)
            if(dim ==2):
                print('Plotting shape')
                P1 = self.P
                P2 = np.roll(self.P,-1, axis = 0)
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 
                lc = LineCollection(ls, linewidths=0.5, colors='b')
                plt.gca().add_collection(lc)

                if self.nonconvex_interceptor is not None:
                    # draw arrows
                    line_dir = self.nonconvex_interceptor.l_dir
                    if bdry_arrow:
                        plt.gca().quiver(P1[:,0], P1[:,1], line_dir[:,0], line_dir[:,1], color = 'blue') 
                    else:
                        pass

                    # print('Plotting nonconvex_interceptor', shape.nonconvex_interceptor)
                    nci = self.nonconvex_interceptor

                    if angle_bisector:
                        # if nci.obt_bisec is not None:
                        if nci.obt_bisec.any():
                            X12 = np.transpose(nci.obt_bisec[:,[0,2]])
                            Y12 = np.transpose(nci.obt_bisec[:,[1,3]])
                            plt.plot(X12, Y12, 'r-', linewidth = 0.5 )

                    if extended_bdry:
                        # if nci.ext_bdry is not None:
                        if nci.obt_bisec.any():
                            X12 = np.transpose(nci.ext_bdry[:,[0,2]])
                            Y12 = np.transpose(nci.ext_bdry[:,[1,3]])
                            plt.plot(X12, Y12, 'c-', linewidth = 0.5 )
                if plot_bounding_ball:
                    if dim==3:
                        angles = np.linspace(0, 2* np.pi, num = bounding_ball_steps, endpoint = True)
                        P = bounding_ball_rad * np.array([np.cos(angles), np.sin(angles)]).transpose()
                        # return column matrices
                        plt.plot(P[:,0], P[:,1], '--')
                if plot_included_ball:
                    if dim==2:
                        angles = np.linspace(0, 2* np.pi, num = included_ball_steps, endpoint = True)
                        P = included_ball_rad * np.array([np.cos(angles), np.sin(angles)]).transpose()
                        # return column matrices
                        plt.plot(P[:,0], P[:,1], '--')

                plt.axis('scaled')
                # plt.gca().set_axis_off()
                plt.show()
            else:
                pass

class NonConvexInterceptor(object):
    """ A set of lines that intersect with bonds that extend beyond the domain"""
    def __init__(self, obt_bisec, ext_bdry, unit_normal, l_dir):
        self.obt_bisec = obt_bisec
        self.ext_bdry = ext_bdry
        self.unit_normal = unit_normal
        self.l_dir = l_dir
        
        self.use = 'all'
        self.proj_3d = 'xy'
        
    def all(self):
        """all lines combined
        :returns: TODO

        """
        if (self.use == 'all'):
            return np.append(self.obt_bisec, self.ext_bdry, axis = 0)
        elif (self.use == 'bisec'):
            return self.obt_bisec
        elif (self.use == 'bdry'):
            return self.ext_bdry
        elif (self.use == 'edges'):
            return self.obt_bisec
        else:
            print('Wrong nonconvex_intercetor.use string given.')

def gen_nonconvex_interceptors(P, extension = 5e-5 ):
    """
    :P: A counterclockwise oriented polygon
    :returns: nx4 array, each row contains the endpoint of each interceptor

    """
    n = len(P)
    l = np.zeros((n,2))
    for i in range(n):
        l[i] = P[(i+1)%n] - P[i%n] 

    # max of absolute value for shape boundary rectangle
    P_max = np.max(np.sqrt(P**2), axis = 0)
    # print('P_max: ', P_max)

    P_ext = np.zeros((n,2))
    unit_normal = np.zeros((n,2))
    l_dir = np.zeros((n,2))

    obt_bisec = []
    
    for i in range(n):
        cross_prod = np.cross(l[i], l[(i+1)%n])

        # angle_between = np.arcsin(-cross_prod/np.linalg.norm(l[i])/np.linalg.norm(l[(i+1)%n]))
        # print('nonconvex_angle in degree: ', angle_between * 180/np.pi)

        # if(cross_prod < 0):
        # one vertex of the interceptor
        A0 = P[(i+1)%n]
        # flip the inward tangent
        l0_u = -l[i]
        l0_u /= np.linalg.norm(l0_u)
        # store
        l_dir[i] = -l0_u
        l1_u = l[(i+1)%n]
        l1_u /= np.linalg.norm(l1_u)
        # print('l0, l1:', l0_u, l1_u)
        v = (l0_u + l1_u)/2 
        unit_normal[(i+1)%n] = v

        if (np.linalg.norm(v) == 0):
            # print('yes, the norm is zero')
            # clockwise 90 degree rotation of l0_u or l1_u, (-y, x)
            v[0] =  -l0_u[1]
            v[1] = l0_u[0]
        # print('i = ', i, ' v = ', v)
            
        v /= np.linalg.norm(v)

        if (cross_prod>0):
            P_ext[(i+1)%n] = A0 - extension * v
        else:
            P_ext[(i+1)%n] = A0 + extension * v
        

        if(cross_prod < 0):
            t_list = []
            # find the intersections with other lines (excluding l(i), l(i+1))
            for j in range(n):
                if j not in [i, j+1]:
                    mat = np.c_[ v, P[j]-P[(j+1)%n] ]
                    b = P[j] - A0

                    if (np.linalg.det(mat)):
                        ts = np.linalg.solve(mat, b)
                        # print('solution for i=', i, ' j=', j, ': ', ts)
                        t = ts[0]
                        s = ts[1]

                        if ( ((s >=0)and(s<=1)) and (t > 0) ):
                            t_list.append(t)
                            # print('Self-intersection detected: ', j)

            if t_list:
                t_min = np.min(np.array(t_list))
                A1 = A0 + t_min * v
                # print('Found self-intersection: i=', i, ';from ', A0, ' to ', A1)
            else:
                # print('No self-intersection. Using maximum for i=', i)
                t_bd = np.linalg.norm(A0) + P_max
                # print('v = ', v)
                # print('max = ', t_bd)
                A1 = A0 + t_bd * v
                # print('No intersection with self-boundary. A1 = ', A1)

            obt_bisec.append( [A0[0], A0[1], A1[0], A1[1]] )

    B_ext = np.append(P_ext, np.roll(P_ext,-1, axis = 0), axis = 1)

    # out = np.append(np.array(nonconvex_interceptor), B_ext, axis = 0)
    # return out
    return NonConvexInterceptor(obt_bisec=np.array(obt_bisec), ext_bdry=B_ext, unit_normal=unit_normal, l_dir=l_dir)



def line_1d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/1d/line.msh')


def plank(l=5e-3, s=1e-3):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        [-l,s]
        ])
    return Shape(P)

def tie(l=5e-3, s=1e-3):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        # [-l,s]
        ])
    return Shape(P)

def plank_wedge(l=5e-3, s=1e-3, w_loc_ratio=0.25, w_thickness_ratio=0.05, w_depth_ratio=1):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        [l*(1-w_loc_ratio)+l*(w_thickness_ratio), s],
        [l*(1-w_loc_ratio), s*(1-w_depth_ratio)],
        [l*(1-w_loc_ratio)-l*(w_thickness_ratio), s],
        [-l,s]
        ])

    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def ice_wedge(l=5e-3, s=1e-3, extra_ratio=1.2, w_loc_ratio=0.25, w_thickness_ratio=0.05, w_depth_ratio=1):
    P = np.array([
        [-l,-s],
        [l,-s],
        # [l,s],
        [l*extra_ratio, s],
        [l*(1-w_loc_ratio)+l*(w_thickness_ratio), s],
        [l*(1-w_loc_ratio), s*(1-w_depth_ratio)],
        [l*(1-w_loc_ratio)-l*(w_thickness_ratio), s],
        [-l,s]
        ])

    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,1],
        [0,1]
        ])
    return Shape(P)

def rect(scaling=1e-3, ratio_length=0.7, ratio_vol=None):
    """
    Rectangle with specified ratio of side lengths inscribed in a circle of radius scaling
    ratio_length: ratio between the lengths of the sides (e.g. ratio=1 means square) 
    ratio_vol: area of the rectangle compared to the area of the square 2R^2; takes precedence over ratio_length
    """

    if ratio_vol:
        # if specified, modify ratio_length
        ratio_length = np.sqrt(1/(ratio_vol**2) - 1) + 1/ratio_vol

    a = 2  / np.sqrt(1 + ratio_length**2)
    b = ratio_length * a

    P = float(scaling) * np.array([
        [-b/2, -a/2],
        [b/2, -a/2],
        [b/2, a/2],
        [-b/2, a/2]
        ])
    return Shape(P)

def L_symm(scaling=1e-3, c_ratio=0.2, ratio_vol=None):
    """
    L-shape with symmetric arms at 45 degrees
    :c_ratio: between 0 and 1; c_ratio = 1 means the shape is a square, 0=two perpendicular lines
    """

    if ratio_vol:
        # if specified, modify ratio_length
        ratio_length = np.sqrt(1/(ratio_vol**2) - 1) + 1/ratio_vol

    c = c_ratio * np.sqrt(2)

    P = float(scaling) * np.array([
        [0, -1],
        [1, 0],
        [0, 1],
        [0-c/np.sqrt(2), 1-c/np.sqrt(2)],
        [1-np.sqrt(2)*c, 0],
        [0-c/np.sqrt(2), -1+c/np.sqrt(2)],
        ])
    return Shape(P)

def triangle(scaling = 1e-3, L = 10, angle=np.pi/3):
    P = scaling * np.array([
        [0,0],
        [L,0],
        [L,L*np.tan(angle)]
        ])
    return Shape(P)

def box_notch():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box_prenotch(l=5e-3, s=1e-3, a=0.1e-3):
    # a: slit length
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,-a/2],
        [0,-a/2],
        [-a/2,0],
        [0,a/2],
        [l,a/2],
        [l,s],
        [-l,s]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box_notch_2():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        [0.5,0.75],
        [1,0.75],
        [1,1],
        [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'all'
    # print(nci)
    return Shape(P, nci)

def box_notch_3():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        # [0.3,0.6],
        [0.3,0.3],
        [0.2,0.6],
        [1,0.7],
        [1,1],
        # intentional defect, parallel line
        # [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'all'

    # print(nci)
    return Shape(P, nci)

def kalthoff(l=200e-3, s=100e-3, dist=50e-3, a=1.5e-3):
    # a: slit thickness
    # dist: width of the midsection
    P = np.array([
        [-l/2,      -s/2],
        [l/2,       -s/2],
        [l/2,       s/2],
        [dist/2+a,  s/2],
        [dist/2+a,  s/2-dist],
        [dist/2+a/2,  s/2-dist-a/2],
        [dist/2,    s/2-dist],
        [dist/2,    s/2],
        [-dist/2,   s/2],
        [-dist/2,   s/2-dist],
        [-dist/2-a/2,   s/2-dist-a/2],
        [-dist/2-a, s/2-dist],
        [-dist/2-a, s/2],
        [-l/2,      s/2],
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def kalthoff_simpler(l=200e-3, s=100e-3, dist=50e-3, a=1.5e-3):
    # a: slit thickness
    # dist: width of the midsection
    P = np.array([
        [-l/2,      -s/2],
        [l/2,       -s/2],
        [l/2,       s/2],
        [dist/2+a,  s/2],
        [dist/2+a/2,  s/2-dist-a/2],
        [dist/2,    s/2],
        [-dist/2,   s/2],
        [-dist/2-a/2,   s/2-dist-a/2],
        [-dist/2-a, s/2],
        [-l/2,      s/2],
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)



def kalthoff3d(xyz=[0,0,0], L1=200e-3, L2=100e-3, L3=9e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):
# def kalthoff3d(xyz=[-10,10,10], L1=20, L2=20, L3=20, meshsize=1, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/k3d_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addRectangle(xyz[0], xyz[1], xyz[2], L1, L2, tag=1)
    nz = L3/meshsize

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 1)], 0, 0, L3, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    c1x = (200/2-50/2-1.5/2)*1e-3
    c2x = (200/2+50/2+1.5/2)*1e-3
    c1y = (50-1.5/2)*1e-3

    crack_1 = [c1x, c1y, c1x, 100e-3 + 1e-4]
    crack_2 = [c2x, c1y, c2x, 100e-3 + 1e-4]
    bisec = [crack_1, crack_2]
    nonconvex_interceptor = NonConvexInterceptor(obt_bisec=bisec, ext_bdry=None, unit_normal=None, l_dir=None)
    nonconvex_interceptor.use = 'bisec'
    nonconvex_interceptor.proj_3d = 'xy'

    return Shape(P=None, nonconvex_interceptor=nonconvex_interceptor, msh_file=msh_file)


def vase(l=4e-3, s=1e-3, amp=0.5e-3, steps = 20, n_pi=4, phase_1=np.pi/2, phase_2=np.pi/2):

    xx = np.linspace(-l, l, num=steps)
    P = np.c_[xx, -s + amp*np.sin(xx*n_pi*np.pi/l + phase_1)]
    xx_rev = np.linspace(l, -l, num=steps)
    P_rev = np.c_[xx_rev, s + amp*np.sin(xx*n_pi*np.pi/l + phase_2)]
    P = np.append(P, P_rev, axis=0)

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)


def small_disk(scaling = 1e-3, steps = 20):
    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = scaling * P.transpose()
    return Shape(P)

def belt_reverse_gear(scaling = 1e-3, teethn=5, half_thickness_ratio=0.8, inner_circle_ratio=0.7, filename_suffix='00', meshsize=1e-3, tooth_gap_ratio=0.6, meshdata_dir='meshdata'):

    msh_file = meshdata_dir+'/gear_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    # outer_rad = scaling*half_thickness_ratio
    outer_rad = scaling * half_thickness_ratio
    inner_rad = scaling * inner_circle_ratio

    gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)

    mid_angles = np.linspace(0, 2*np.pi, num=teethn, endpoint = False)

    n_point = 5   # total points to introduce
    n_line = 4

    arclist = []
    for i in range(teethn):
        angle_1 = mid_angles[i]
        angle_2 = mid_angles[i] + 2 * np.pi/ teethn
        angle_3 = mid_angles[i] + 2 * np.pi/ teethn * (tooth_gap_ratio)

        p1_x = outer_rad * np.cos(angle_1)
        p1_y = outer_rad * np.sin(angle_1)

        p2_x = outer_rad * np.cos(angle_2)
        p2_y = outer_rad * np.sin(angle_2)

        p3_x = outer_rad * np.cos(angle_3)
        p3_y = outer_rad * np.sin(angle_3)

        p4_x = inner_rad * np.cos(angle_3)
        p4_y = inner_rad * np.sin(angle_3)

        p5_x = inner_rad * np.cos(angle_2)
        p5_y = inner_rad * np.sin(angle_2)

        ptag_1 = 1+n_point*i +1
        ptag_2 = 1+n_point*i +2
        ptag_3 = 1+n_point*i +3
        ptag_4 = 1+n_point*i +4
        ptag_5 = 1+n_point*i +5

        gmsh.model.occ.addPoint(p1_x, p1_y, 0, tag=ptag_1)
        gmsh.model.occ.addPoint(p2_x, p2_y, 0, tag=ptag_2)
        gmsh.model.occ.addPoint(p3_x, p3_y, 0, tag=ptag_3)
        gmsh.model.occ.addPoint(p4_x, p4_y, 0, tag=ptag_4)
        gmsh.model.occ.addPoint(p5_x, p5_y, 0, tag=ptag_5)

        ltag_1 = n_line*i +1
        ltag_2 = n_line*i +2
        ltag_3 = n_line*i +3
        ltag_4 = n_line*i +4

        gmsh.model.occ.addCircleArc(ptag_1, 1, ptag_3, tag=ltag_1)
        gmsh.model.occ.addLine(ptag_3, ptag_4, tag=ltag_2)
        gmsh.model.occ.addCircleArc(ptag_4, 1, ptag_5, tag=ltag_3)
        gmsh.model.occ.addLine(ptag_5, ptag_2, tag=ltag_4)

        arclist.append(ltag_1)
        arclist.append(ltag_2)
        arclist.append(ltag_3)
        arclist.append(ltag_4)

    gmsh.model.occ.addCurveLoop(arclist, 1)
    # gmsh.model.occ.addPlaneSurface([1], 1)

    in_tag = n_point * teethn +2
    gmsh.model.occ.addCircle(0, 0, 0, scaling, in_tag)
    gmsh.model.occ.addCurveLoop([in_tag], tag=2)
    gmsh.model.occ.addPlaneSurface([2, 1], 1)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("General.Verbosity", 0);
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)
    gmsh.finalize()

    # nonconvex_interceptor
    nci_steps = 10
    angles = np.linspace( 2* np.pi, 0, num = nci_steps, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = 0.99 * inner_rad * P.transpose()
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'all'

    return Shape(P=None, nonconvex_interceptor=nonconvex_interceptor, msh_file=msh_file)
    # return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def gear_inscribed_par(scaling = 1e-3, teethn=5, inner_circle_ratio=0.7, filename_suffix='00', meshsize=1e-3, hollow=True, hollow_inner_rad=0.3, meshdata_dir='meshdata'):

    msh_file = meshdata_dir + '/gear_'+str(filename_suffix)+'.msh'
    gmsh.initialize()
    inner_rad = scaling*inner_circle_ratio
    
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)

    
    gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)

    angle_width = np.pi / teethn/4 # angular width of each tooth

    mid_angles = np.linspace(0, 2*np.pi, num=teethn, endpoint = False)

    n_point = 5   # total points to introduce
    n_line = 4
    #n_line = 1

    arclist = []
    for i in range(teethn):
        # angle_before = mid_angles[i] - angle_width/2
        # angle_after = mid_angles[i] + angle_width/2
        # next_angle_before = mid_angles[i] + np.pi/teethn 

        angle_1 = mid_angles[i]
        angle_2 = mid_angles[i] + 2 * np.pi/ teethn
        angle_3 = mid_angles[i] + 2 * np.pi/ teethn/2

        p1_x = scaling * np.cos(angle_1)
        p1_y = scaling * np.sin(angle_1)

        p2_x = scaling * np.cos(angle_2)
        p2_y = scaling * np.sin(angle_2)

        p3_x = scaling * np.cos(angle_3)
        p3_y = scaling * np.sin(angle_3)

        p4_x = inner_rad * np.cos(angle_3)
        p4_y = inner_rad * np.sin(angle_3)

        p5_x = inner_rad * np.cos(angle_2)
        p5_y = inner_rad * np.sin(angle_2)

        ptag_1 = 1+n_point*i +1
        ptag_2 = 1+n_point*i +2
        ptag_3 = 1+n_point*i +3
        ptag_4 = 1+n_point*i +4
        ptag_5 = 1+n_point*i +5

        gmsh.model.occ.addPoint(p1_x, p1_y, 0, tag=ptag_1)
        gmsh.model.occ.addPoint(p2_x, p2_y, 0, tag=ptag_2)
        gmsh.model.occ.addPoint(p3_x, p3_y, 0, tag=ptag_3)
        gmsh.model.occ.addPoint(p4_x, p4_y, 0, tag=ptag_4)
        gmsh.model.occ.addPoint(p5_x, p5_y, 0, tag=ptag_5)

        ltag_1 = n_line*i +1
        ltag_2 = n_line*i +2
        ltag_3 = n_line*i +3
        ltag_4 = n_line*i +4

        gmsh.model.occ.addCircleArc(ptag_1, 1, ptag_3, tag=ltag_1)
        gmsh.model.occ.addLine(ptag_3, ptag_4, tag=ltag_2)
        gmsh.model.occ.addCircleArc(ptag_4, 1, ptag_5, tag=ltag_3)
        gmsh.model.occ.addLine(ptag_5, ptag_2, tag=ltag_4)

        arclist.append(ltag_1)
        arclist.append(ltag_2)
        arclist.append(ltag_3)
        arclist.append(ltag_4)

    gmsh.model.occ.addCurveLoop(arclist, 1)

    if hollow:
        print(arclist)
        in_tag = n_point * teethn +2
        print(in_tag)
        gmsh.model.occ.addCircle(0, 0, 0, hollow_inner_rad*scaling, in_tag)
        gmsh.model.occ.addCurveLoop([in_tag], tag=2)
        gmsh.model.occ.addPlaneSurface([1, 2], 1)
    else:
        gmsh.model.occ.addPlaneSurface([1], tag=1)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("General.Verbosity", 0);
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)
    gmsh.finalize()

    ## Todo: nonconvex_interceptor

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def gear_inscribed(scaling = 1e-3, steps=40, teethn=5, teethl=0.7):
    inner_scaling = scaling * teethl
    scaling_vec = []
    for n in range(teethn):
        thisnum = int(steps/teethn)
        scaling_vec.extend( [scaling] * int(thisnum/2) )
        scaling_vec.extend( [inner_scaling] * int(thisnum/2) )

    angles = np.linspace(0, 2* np.pi, num = len(scaling_vec), endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = (scaling_vec * P).transpose()

    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    return Shape(P, nci)

def reverse_disk(scaling = 1e-3, steps = 20):
    angles = np.linspace( 2* np.pi, 0, num = steps, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = scaling * P.transpose()
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)

def star(steps = 20, scaling=1e-3, incirc_ratio=0.2):
    """ star shaped
    """

    # rads = np.random.uniform(low=incirc_ratio, high=1, size=steps)
    rads = np.ones(steps)
    rads[::2] = incirc_ratio


    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)
    P = scaling * rads * np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P =  P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def pertdisk_by_mean(steps = 20, seed=5, scaling=1e-3, mean_r_ratio=0.8, noise_unif_low=0.1, noise_unif_high=0.1, angle_drift_amp = 0, angle_drift_supp_ratio = 0.25, drift_type='exp'):

    """ r-theta curve with mean r given
    :steps: how many radial steps
    :mean_r_ratio: desired mean radius
    :noise_unif_low: a where noise of uniform[-a,b] added to the mean radius
    :noise_unif_high: b where noise of uniform[-a,b] added to the mean radius
    :angle_drift_amp: amplitude in the drift of the radius (amount of bulging)
    :angle_drift_supp_ratio: angular support of the drift function (area of bulging)
    :drift_type: exp, sin
    """

    np.random.seed(seed)

    rads = np.ones(steps) * mean_r_ratio
    rads += np.random.uniform(low=-noise_unif_low, high=noise_unif_high, size=steps)

    if drift_type=='exp':
        xx = np.linspace(-1, 1, num=steps, endpoint=False)
        yy = angle_drift_amp * np.exp( -(xx - 0)**2 / (angle_drift_supp_ratio)**2)
    elif drift_type=='sin_full':
        xx = np.linspace(0, 2*np.pi, num=steps, endpoint=False)
        yy = angle_drift_amp * np.sin(xx / angle_drift_supp_ratio)
    elif drift_type=='sin':
        xx = np.linspace(0, 2*np.pi, num=steps, endpoint=False)
        xx[xx > angle_drift_supp_ratio * np.pi] = 0
        yy = angle_drift_amp * np.sin(xx / angle_drift_supp_ratio)
    else:
        yy = np.zeros(steps)

    rads += yy

    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)
    P = scaling * rads * np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P =  P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def pertdisk_incirc(steps = 20, seed=5, scaling=1e-3, incirc_ratio=0.2, angle_drift_amp = 0, angle_drift_std_ratio = 0.25):

    """ 
    """

    np.random.seed(seed)

    rads = np.random.uniform(low=incirc_ratio, high=1, size=steps)

    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)
    P = scaling * rads * np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P =  P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def perturbed_disk(steps = 20, seed=5, scaling=1e-3, std=0.2, angle_drift_amp = 0, angle_drift_std_ratio = 0.25):

    """ 
    : radius is taken from iid Unif([scaling*(1-std), scaling]), i.e., std=1-phi in paper
    :angle_drift_amp: If you want the particle to protrude outward, make sure (scaling + angle_drift_amp) <= radius of bounding circle
    If you want a dent in the particle, use negative value of angle_drift_amp


    """

    np.random.seed(seed)

    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)

    # rads = 0 + (np.random.rnd(steps, 1) * (2*np.pi - 0))

    xx = np.linspace(0, steps, num=steps)

    # to ensure the particle is within a bounding circle of radius `scaling`
    # rad_scales =  (np.ones((1,steps)) - angle_drift_amp) + yy
    rad_scales =  (np.ones((1,steps)))

    if angle_drift_amp:
        var = steps * angle_drift_std_ratio
        yy = angle_drift_amp * np.exp( - (xx - np.floor(steps/2))**2/var )

        rad_scales += yy


    # uniform sample from [0, scaling*std]
    rands = np.random.rand(1,steps)
    # print(rands * std)
    # rads = scaling * (1-std) + (np.random.rand(1,steps) * scaling * std)
    rads = scaling * (1-std) * rad_scales + (rands * scaling * std)

    # normal sample
    # random_l = np.random.normal(loc = std * scaling, scale=scaling*std/2, size = (1,steps))
    # print(random_l)
    # rads = scaling * (1-std) + random_l

    P = rads * np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P =  P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def polygon_inscribed(sides = 5, scaling = 1e-3):
    angles = np.linspace(0, 2* np.pi, num = sides, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = scaling * P.transpose()
    return Shape(P)

def pacman(angle = np.pi/2):
    scaling = 1e-3;
    steps = 20;
    angles = np.linspace(angle/2, 2* np.pi - angle/2, num = steps, endpoint = True)
    P = np.array([np.cos(angles), np.sin(angles)])
    # print(P)
    P = np.append(P, np.array([[0], [0]]), axis = 1)
    # return column matrices
    P = scaling * P.transpose()

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([
        # [0,0, 1,0]
    # ])
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    # print('pacman nonconvex_interceptor: ', nonconvex_interceptor)
    return Shape(P, nonconvex_interceptor)

def ring_segment(scaling = 1e-3, steps=5, angle = np.pi/2, inner_rad=0.5):
    angles = np.linspace(angle/2, 2* np.pi - angle/2, num = steps, endpoint = True)
    P = np.array([np.cos(angles), np.sin(angles)])

    angles_rev = np.linspace(2* np.pi - angle/2, angle/2, num = steps, endpoint = True)
    Q = inner_rad*np.array([np.cos(angles_rev), np.sin(angles_rev)])

    P = np.append(P, Q, axis = 1)

    # return column matrices
    P = scaling * P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)

def plus(ratio = 0.25):
    scaling = 1e-3;

    # ratio = 0.25
    long = 1
    short = ratio * long

    P = scaling * np.array([ [short, -short],
            [long, -short],
            [long, short],
            [short, short],
            [short, long],
            [-short, long],
            [-short, short],
            [-long, short],
            [-long, -short],
            [-short, -short],
            [-short, -long],
            [short, -long]
            ])

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([ [short, -short, long, -long],
            # [short, short, long, long],
            # [-short, short, -long, long],
            # [-short, -short, -long, -long],
            # ])
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def plus_inscribed(notch_dist = 0.25, scaling = 1e-3):
    """ a plus with maximum possible lenth while being inscribed in a disk of radius 1e-3
    The maximum possible thickness is 
    : notch_dist: the distance from the center of disk to the inner notch, betwen 0 and 1 
    """

    short = notch_dist/np.sqrt(2)
    long = np.sqrt(1 - short**2)

    P = scaling * np.array([ [short, -short],
            [long, -short],
            [long, short],
            [short, short],
            [short, long],
            [-short, long],
            [-short, short],
            [-long, short],
            [-long, -short],
            [-short, -short],
            [-short, -long],
            [short, -long]
            ])

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([ [short, -short, long, -long],
            # [short, short, long, long],
            # [-short, short, -long, long],
            # [-short, -short, -long, -long],
            # ])

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)

def small_disk_fromfile():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/peridem_mat.msh')

def box_wo_circ():
    msh_file='meshdata/2d/box_wo_circ.msh'
    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def test():
    filename = "meshdata/2d/test.msh"
    with pygmsh.occ.Geometry() as geom:

        P_bdry=[
            [0.0, 0.0],
            [1.0, -0.2],
            [1.1, 1.2],
            [0.1, 0.7],
        ]
        meshsize = 0.1
        # geom.add_polygon(
            # [
                # [0.0, 0.0],
                # [1.0, -0.2],
                # [1.1, 1.2],
                # [0.1, 0.7],
            # ],
            # mesh_size=0.1,
        # )
        # mesh = geom.generate_mesh()

        polygon1 = geom.add_polygon(
            P_bdry,
            mesh_size= meshsize,
        )
        geom.add_physical(polygon1.surface, "surface1")
        mesh = geom.generate_mesh()
        meshio.write(filename, mesh, file_format="gmsh")

    return Shape(P=None, nonconvex_interceptor=None, msh_file=filename)

def pygmsh_geom_test_works(scaling=1e-3, meshsize = 0.5e-3):
    P_bdry= scaling * np.array([
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
    ])
    msh_file = 'meshdata/geom_test.msh'
    with pygmsh.occ.Geometry() as geom:
        polygon1 = geom.add_polygon(
            P_bdry,
            mesh_size= meshsize,
        )
        geom.add_physical(polygon1.surface, 'surface1')
        mesh = geom.generate_mesh()
        print(mesh)
        # mesh.write(msh_file)
        pygmsh.write(msh_file)
        print('saved')

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def pygmsh_geom_test_also_works(scaling=1e-3, meshsize = 0.5e-3):
    P_bdry= scaling * np.array([
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
    ])
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()
    # add polygon
    polygon1 = model.add_polygon(
        P_bdry,
        mesh_size= meshsize,
    )
    return Shape(P=None, nonconvex_interceptor=None, pygmsh_geom=geometry)

def pygmsh_geom_test_spline(scaling=5e-3, meshsize = 0.5e-3):
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()

    # circle1 = model.add_circle([0,0], radius=scaling, mesh_size=meshsize)
    # circle2 = model.add_circle([0,0], radius=scaling/2, mesh_size=meshsize)

    # loop1 = model.add_curve_loop(circle1.curve_loop)
    # loop2 = model.add_curve_loop([circle2])
    # loop1 = model.add_curve_loop(circle1)
    # loop2 = model.add_curve_loop(circle2)

    # plane_surface = model.add_plane_surface(circle1.curve_loop, holes=[circle2.curve_loop])

    print('geom deon')
    # model.add_physical([circle1, circle2], "surface1")
    # model.add_plane_surface([circle1.curve_loop, circle2.curve_loop], meshsize)
    # model.add_plane_surface(circle1.curve_loop)


    lcar = meshsize
    p1 = model.add_point([0.0*scaling, 0.0*scaling], lcar)
    p2 = model.add_point([1.0*scaling, 0.0*scaling], lcar)
    p3 = model.add_point([1.0*scaling, 0.5*scaling], lcar)
    p4 = model.add_point([1.0*scaling, 1.0*scaling], lcar)
    s1 = model.add_bspline([p1, p2, p3, p4])

    p2 = model.add_point([0.0*scaling, 1.0*scaling], lcar)
    p3 = model.add_point([0.5*scaling, 1.0*scaling], lcar)
    s2 = model.add_spline([p4, p3, p2, p1])

    ll = model.add_curve_loop([s1, s2])
    pl = model.add_plane_surface(ll)

    return Shape(P=None, nonconvex_interceptor=None, pygmsh_geom=geometry)

def pygmsh_geom_test(scaling=5e-3, meshsize = 0.5e-3):
    msh_file = 'meshdata/geom_test.msh'
    with pygmsh.geo.Geometry() as geom:
        circle1 = geom.add_circle([0,0], radius=scaling, mesh_size=meshsize)
        circle2 = geom.add_circle([0,0], radius=scaling/2, mesh_size=meshsize)

        geom.boolean_difference(circle1, [circle2])
        # geom.boolean_difference(circle1, circle2)
        # geom.add_physical(polygon1.surface, 'surface1')
        mesh = geom.generate_mesh()
        print(mesh)
        # mesh.write(msh_file)
        pygmsh.write(msh_file)
        print('saved')
    
    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def annulus():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/2d/annulus.msh')

def unif_rect(x_min=-1, y_min=-1, length_x=2, length_y=2, meshsize=0.5, ny=4, filename_suffix='00'):
    """
    :returns: Rectangle with uniform mesh using extrusion
    :meshsize: meshsize in the x direction (=length_x/nx)
    :ny: number of times to extrude in the y-direction (meshsize in the y-dir: length_y/nx)
    :filename_suffix: some string tag that is appended to the generated msh file

    """
    msh_file = 'meshdata/unif_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)

    # gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    # gmsh.model.occ.addCircle(0, 0, 0, inner_rad, 2)
    # gmsh.model.occ.addCurveLoop([1], 1)
    # gmsh.model.occ.addCurveLoop([2], 2)
    # gmsh.model.occ.addPlaneSurface([1, 2], 1)

    gmsh.model.occ.addPoint(x_min, y_min, 0, meshSize=meshsize, tag=1)
    gmsh.model.occ.addPoint(x_min+length_x, y_min, 0, meshSize=meshsize, tag=2)
    gmsh.model.occ.addLine(1,2, tag=1)
    # extruding the line: dimension 1, tag 1 i.e., (1,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(1, 1)], 0, length_y, 0, numElements=[ny], heights=[1])


    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    gmsh.option.setNumber("General.Verbosity", 0);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # close gmsh, as opposed to initialize()
    gmsh.finalize()
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    # nonconvex_interceptor
    # nci_steps = 10
    # angles = np.linspace( 2* np.pi, 0, num = nci_steps, endpoint = False)
    # P = np.array([np.cos(angles), np.sin(angles)])
    # # return column matrices
    # P = inner_rad * P.transpose()
    # nonconvex_interceptor = gen_nonconvex_interceptors(P)
    # nonconvex_interceptor.use = 'all'

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)
    # return Shape(P=None, nonconvex_interceptor=nonconvex_interceptor, msh_file=msh_file)
    # return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def unif_rect_test(x_min=-1, y_min=-1, length_x=2, length_y=2, meshsize=0.5, ny=4, filename_suffix='00', meshdata_dir='meshdata'):
    """
    :returns: Rectangle with uniform mesh using extrusion
    :meshsize: meshsize in the x direction (=length_x/nx)
    :ny: number of times to extrude in the y-direction (meshsize in the y-dir: length_y/nx)
    :filename_suffix: some string tag that is appended to the generated msh file

    """
    msh_file = meshdata_dir + '/unif_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)

    # gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    # gmsh.model.occ.addCircle(0, 0, 0, inner_rad, 2)
    # gmsh.model.occ.addCurveLoop([1], 1)
    # gmsh.model.occ.addCurveLoop([2], 2)
    # gmsh.model.occ.addPlaneSurface([1, 2], 1)

    gmsh.model.occ.addPoint(x_min, y_min, 0, meshSize=meshsize, tag=1)
    gmsh.model.occ.addPoint(x_min+length_x, y_min, 0, meshSize=meshsize, tag=2)
    gmsh.model.occ.addLine(1,2, tag=1)
    # extruding the line: dimension 1, tag 1 i.e., (1,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(1, 1)], -1, length_y, 0, numElements=[ny], heights=[1])


    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    gmsh.option.setNumber("General.Verbosity", 0);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # close gmsh, as opposed to initialize()
    gmsh.finalize()
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    # nonconvex_interceptor
    # nci_steps = 10
    # angles = np.linspace( 2* np.pi, 0, num = nci_steps, endpoint = False)
    # P = np.array([np.cos(angles), np.sin(angles)])
    # # return column matrices
    # P = inner_rad * P.transpose()
    # nonconvex_interceptor = gen_nonconvex_interceptors(P)
    # nonconvex_interceptor.use = 'all'

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def wheel_annulus(scaling=1e-3, meshsize=1e-3, inner_circle_ratio=0.7, meshdata_dir='meshdata', filename_suffix='00', nci_steps=10):
    """
    :returns: TODO

    """
    # msh_file = 'meshdata/ring_'+str(filename_suffix)+'.msh'
    msh_file = meshdata_dir+'/ring_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    inner_rad = scaling*inner_circle_ratio
    
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    gmsh.model.occ.addCircle(0, 0, 0, inner_rad, 2)
    gmsh.model.occ.addCurveLoop([1], 1)
    gmsh.model.occ.addCurveLoop([2], 2)
    gmsh.model.occ.addPlaneSurface([1, 2], 1)

    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    gmsh.option.setNumber("General.Verbosity", 0);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # close gmsh, as opposed to initialize()
    gmsh.finalize()
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    # nonconvex_interceptor
    nci_steps = 10
    angles = np.linspace( 2* np.pi, 0, num = nci_steps, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = inner_rad * P.transpose()
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'all'

    return Shape(P=None, nonconvex_interceptor=nonconvex_interceptor, msh_file=msh_file)
    # return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def gmsh_test(scaling=1e-3, meshsize=1e-3):
    """
    :returns: TODO

    """
    msh_file = 'meshdata/msh_test.msh'
    gmsh.initialize()
    print('done')
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling/2, 2)
    gmsh.model.occ.addCurveLoop([1], 1)
    gmsh.model.occ.addCurveLoop([2], 2)
    gmsh.model.occ.addPlaneSurface([1, 2], 1)

    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)
#######################################################################
#                                 3D                                  #
#######################################################################
#######################################################################

#-------------- double sphere-------------------------
def hollow_sphere(R=200e-3, r=100e-3, meshsize=10e-3, meshdata_dir='meshdata', filename_suffix='00'):
    msh_file = meshdata_dir+'/hollow_sphere_'+str(filename_suffix)+'.msh'
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new model
    #gmsh.model.add("Sphere with Hollow Sphere")
    myOrigx, myOrigy, myOrigz = 0,0,0
    cx = -myOrigx
    cy = -myOrigy
    cz = -myOrigz
    # Parameters for the radii
    # R   : Outer sphere radius
    # r   : Inner sphere radius

    # Global characteristic length for finer mesh
    #lc = meshsize  # Adjust this value to make the mesh finer or coarser

    # Add the outer sphere
    outer_sphere = gmsh.model.occ.addsphere(cx, cy, cz, r)

    # Add the inner sphere
    inner_sphere = gmsh.model.occ.addSphere(cx, cy, cz, r)

    # Perform a boolean difference to subtract the inner sphere from the outer sphere
    gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])

    # Synchronize the model
    gmsh.model.occ.synchronize()

    # Set the mesh size for the entire model
    #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    # Generate the mesh
    gmsh.model.mesh.generate(3)
    # Optionally, save the mesh to a file
    
    gmsh.write(msh_file) # save to file
   
    # Finalize Gmsh
    gmsh.finalize()
    nonconvex_vertex =[r,cx,cy,cz]
    #print(nonconvex_vertex)
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)

    print("this is nonconvex_vertex")
    print(nonconvex_vertex)         
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)
def sphere(rad=200e-3, meshsize=10e-3, meshdata_dir='meshdata', filename_suffix='00'):
    msh_file = meshdata_dir+'/sphere_'+str(filename_suffix)+'.msh'
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new model
    #gmsh.model.add("Sphere with Hollow Sphere")
    myOrigx, myOrigy, myOrigz = 0,0,0
    cx = -myOrigx
    cy = -myOrigy
    cz = -myOrigz
    # Parameters for the radii
    # R   : Outer sphere radius
    # r   : Inner sphere radius

    # Global characteristic length for finer mesh
    #lc = meshsize  # Adjust this value to make the mesh finer or coarser

    # Add the outer sphere
    gmsh.model.occ.addSphere(cx, cy, cz, rad)

    # Add the inner sphere

    # Perform a boolean difference to subtract the inner sphere from the outer sphere

    # Synchronize the model
    gmsh.model.occ.synchronize()

    # Set the mesh size for the entire model
    #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    # Generate the mesh
    gmsh.model.mesh.generate(3)
    # Optionally, save the mesh to a file
    
    gmsh.write(msh_file) # save to file
   
    # Finalize Gmsh
    gmsh.finalize()
    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)



def _sphere_3d(rad=1e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/sphere_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addSphere(0, 0, 0, rad)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);


    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


#-----------------------------------------------------muri 2025--------------------------
#----------------------------------------------------------------------------------------


def plate_2d_notch_2(L=150, meshsize=0.2e-3, meshdata_dir='meshdata', filename_suffix='00'):
    


    msh_file = 'meshdata/plate_2d_notch_2.msh'
    gmsh.initialize()
    gmsh.model.add("plate_2d_notch_2")

    plate_width = L
    plate_height = L
    notch_width = 8.0
    notch_depth = 65.0

    # Top notch: 53 mm from left top corner
    top_offset_left = 53.0
    top_x = top_offset_left + notch_width / 2
    top_y = plate_height - notch_depth

    # Bottom notch: 53 mm from right bottom corner
    bottom_offset_right = 53.0
    bottom_x = plate_width - bottom_offset_right - notch_width / 2
    bottom_y = notch_depth

    p = {}
    p[1] = gmsh.model.geo.addPoint(0, 0, 0, meshsize)
    p[2] = gmsh.model.geo.addPoint(bottom_x - notch_width / 2, 0, 0, meshsize)
    p[3] = gmsh.model.geo.addPoint(bottom_x, bottom_y, 0, meshsize)
    p[4] = gmsh.model.geo.addPoint(bottom_x + notch_width / 2, 0, 0, meshsize)
    p[5] = gmsh.model.geo.addPoint(plate_width, 0, 0, meshsize)
    p[6] = gmsh.model.geo.addPoint(plate_width, plate_height, 0, meshsize)
    p[7] = gmsh.model.geo.addPoint(top_x + notch_width / 2, plate_height, 0, meshsize)
    p[8] = gmsh.model.geo.addPoint(top_x, top_y, 0, meshsize)
    p[9] = gmsh.model.geo.addPoint(top_x - notch_width / 2, plate_height, 0, meshsize)
    p[10] = gmsh.model.geo.addPoint(0, plate_height, 0, meshsize)

    
    #----------------------------------------------
    # After you’ve created p[2], p[3], p[4], p[7], p[8], p[9]
    # top_edges = np.array([
    #     [ [top_x + notch_width/2, plate_height], [top_x, top_y] ],   # p7–p8
    #     [ [top_x, top_y], [top_x - notch_width/2, plate_height] ]    # p8–p9
    # ], dtype=float)
    #
    # bot_edges = np.array([
    #     [ [bottom_x - notch_width/2, 0.0], [bottom_x, bottom_y] ],   # p2–p3
    #     [ [bottom_x, bottom_y], [bottom_x + notch_width/2, 0.0] ]    # p3–p4
    # ], dtype=float)
    #
    # nci = NonConvexInterceptor(
    #     obt_bisec=np.vstack([top_edges, bot_edges]),  # shape (4, 2, 2)
    #      ext_bdry=None, unit_normal=None, l_dir=None
    # )
    # nci.use = 'edges'   # up to you; just a hint flag
    #self.interceptor = nci  # make sure your object saves it here
     

    # nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)



    #----------------------------------------------



    lines = [
        gmsh.model.geo.addLine(p[1], p[2]),
        gmsh.model.geo.addLine(p[2], p[3]),
        gmsh.model.geo.addLine(p[3], p[4]),
        gmsh.model.geo.addLine(p[4], p[5]),
        gmsh.model.geo.addLine(p[5], p[6]),
        gmsh.model.geo.addLine(p[6], p[7]),
        gmsh.model.geo.addLine(p[7], p[8]),
        gmsh.model.geo.addLine(p[8], p[9]),
        gmsh.model.geo.addLine(p[9], p[10]),
        gmsh.model.geo.addLine(p[10], p[1]),
    ]

    loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

   
    gmsh.write(msh_file)

    gmsh.finalize()
    
    #----- top v shape notche, A1: point of v notches, B1= midpoint of mouth of the notch
    A1 = [top_x, top_y]
    B1 = [top_x,plate_height]


    A2 = [bottom_x, bottom_y]
    B2 = [bottom_x,0]

    nonconvex_vertex =[A1,B1,A2,B2]

    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)



    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)
#----------------------------------------------------------------------------------------------------------

def plate_2d_notch_1(L=150, meshsize=0.2e-3, meshdata_dir='meshdata', filename_suffix='00'):
    
    msh_file = 'meshdata/plate_2d_notch_2.msh'
    gmsh.initialize()
    gmsh.model.add("plate_2d_notch_1")

    plate_width = L
    plate_height = L
    notch_width = 8.0
    notch_depth = 65.0
    notch_offset_left = 53.0

    tip_x = notch_offset_left + notch_width / 2
    tip_y = plate_height - notch_depth

    p = {}
    p[1] = gmsh.model.geo.addPoint(0, 0, 0, meshsize)
    p[2] = gmsh.model.geo.addPoint(plate_width, 0, 0, meshsize)
    p[3] = gmsh.model.geo.addPoint(plate_width, plate_height, 0, meshsize)
    p[4] = gmsh.model.geo.addPoint(tip_x + notch_width / 2, plate_height, 0, meshsize)
    p[5] = gmsh.model.geo.addPoint(tip_x, tip_y, 0, meshsize)
    p[6] = gmsh.model.geo.addPoint(tip_x - notch_width / 2, plate_height, 0, meshsize)
    p[7] = gmsh.model.geo.addPoint(0, plate_height, 0, meshsize)

    lines = [
        gmsh.model.geo.addLine(p[1], p[2]),
        gmsh.model.geo.addLine(p[2], p[3]),
        gmsh.model.geo.addLine(p[3], p[4]),
        gmsh.model.geo.addLine(p[4], p[5]),
        gmsh.model.geo.addLine(p[5], p[6]),
        gmsh.model.geo.addLine(p[6], p[7]),
        gmsh.model.geo.addLine(p[7], p[1]),
    ]

    loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    
    
    gmsh.write(msh_file)

    gmsh.finalize()
     
    #----- top v shape notche, A1: point of v notches, B1= midpoint of mouth of the notch
    A1 = [tip_x, tip_y]
    B1 = [tip_x,plate_height]

    
   
    
    nonconvex_vertex =[A1,B1]
    
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)


    
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)






#----------------------------------------------------------------------------------------------------------

def plate_2d_circle_void(L=150, r=10 , cx=None, cy=None,  meshsize=0.2e-3, meshdata_dir='meshdata', filename_suffix='00'):
    """
    Create a 2D square plate with a circular hole using OpenCASCADE kernel in Gmsh.

    Parameters:
    - L: Length of the square plate [mm]
    - r: Radius of the circular hole [mm]
    - distance_from_corner: reserved for future offset options (currently unused)
    - conveTol: reserved for mesh convergence (currently unused)
    - meshsize: target mesh size [mm]
    - meshdata_dir: directory to save mesh/geometry files
    - filename_suffix: suffix for output filenames
    - cx, cy: coordinates of the hole center (defaults to center of the plate)
    """
    msh_file = 'meshdata/plate2s_circle_void.msh'
    gmsh.initialize()
  

    if cx is None:
        cx = L / 2
    if cy is None:
        cy = L / 2

    # Create square plate as a rectangle
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, L)

    # Create circle (hole)
    circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)

    # Subtract the hole from the square
    plate_with_hole = gmsh.model.occ.cut([(2, rect)], [(2, circle)])

    # Synchronize CAD with model
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshsize)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Create output directory if not exists
    # Write output files
    
    gmsh.write(msh_file)

    gmsh.finalize()
    nonconvex_vertex =[r,cx,cy]
    
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)

     
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)


#----------------------------------------------------------


# Assuming these exist in your project (same as your example)
# from your_project import Shape, NonConvexInterceptor

def plate_2d_vnotch_circle_void(
    L=150.0,
    r=10.0,
    cx=None,
    cy=None,
    notch_width=3.0,
    notch_depth=65.0,
    notch_offset_left=53.0,
    meshsize=0.2e-3,
    meshdata_dir='meshdata',
    filename_suffix='00'
):
    """
    Create a 2D square plate (0..L x 0..L) with:
      - a circular hole centered at (cx, cy) [defaults to (L/2, L/2)], radius r
      - a V-shaped notch on the top edge, located near the left side

    Notch geometry (on y = L):
      M1 = (notch_offset_left, L)
      M2 = (notch_offset_left + notch_width, L)
      A1 = ( (M1.x + M2.x)/2, L - notch_depth )  # apex
      B1 = ( (M1.x + M2.x)/2, L )                # mouth midpoint

    Parameters
    ----------
    L : float
        Plate side length.
    r : float
        Radius of the circular hole.
    cx, cy : float or None
        Center of circular hole; defaults to plate center if None.
    notch_width : float
        Mouth width of the V-notch (horizontal distance between M1 and M2).
    notch_depth : float
        Vertical depth from the top edge (y=L) down to the apex A1.
    notch_offset_left : float
        Horizontal offset from the left edge to the left mouth point M1.
    meshsize : float
        Target mesh size.
    meshdata_dir : str
        Directory for output mesh files.
    filename_suffix : str
        Suffix to append to the output filenames.

    Returns
    -------
    shape : Shape
        Your project Shape object with NonConvexInterceptor set at the apex.
    key_points : dict
        Dictionary with coordinates for A1 (apex), B1 (mouth midpoint),
        and also M1, M2 for convenience.
    """
    if cx is None:
        cx = L / 2.0
    if cy is None:
        cy = L / 2.0

    # --- Notch key points on the top edge (y = L)
    M1 = (float(notch_offset_left), float(L))
    M2 = (float(notch_offset_left + notch_width), float(L))
    A1 = ((M1[0] + M2[0]) * 0.5, float(L - notch_depth))  # apex
    B1 = ((M1[0] + M2[0]) * 0.5, float(L))                # mouth midpoint

    # Safety checks (no negative depth, within plate, etc.)
    if not (0.0 <= M1[0] <= L and 0.0 <= M2[0] <= L):
        raise ValueError("Notch mouth lies outside the plate horizontally. "
                         "Adjust 'notch_offset_left' and/or 'notch_width'.")
    if not (0.0 <= A1[1] <= L):
        raise ValueError("A1 apex lies outside the plate vertically. "
                         "Reduce 'notch_depth' or increase 'L'.")

    msh_file = 'meshdata/plate2s_vnotch_circle_void.msh'
  

    gmsh.initialize()
    gmsh.model.add("plate2d_vnotch_circle")

    # Plate
    rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, float(L), float(L))

    # Circle hole
    circle = gmsh.model.occ.addDisk(float(cx), float(cy), 0.0, float(r), float(r))

    # V-notch triangle surface: M1--A1--M2
    # Create points
    pM1 = gmsh.model.occ.addPoint(M1[0], M1[1], 0.0)
    pM2 = gmsh.model.occ.addPoint(M2[0], M2[1], 0.0)
    pA1 = gmsh.model.occ.addPoint(A1[0], A1[1], 0.0)

    # Create lines for the triangle boundary
    l_M1A1 = gmsh.model.occ.addLine(pM1, pA1)
    l_A1M2 = gmsh.model.occ.addLine(pA1, pM2)
    l_M2M1 = gmsh.model.occ.addLine(pM2, pM1)

    tri_loop = gmsh.model.occ.addCurveLoop([l_M1A1, l_A1M2, l_M2M1])
    tri_surface = gmsh.model.occ.addPlaneSurface([tri_loop])

    # Cut circle and V-notch from the plate
    # (2, ...) denotes 2D surfaces in OpenCASCADE
    cut_res, _ = gmsh.model.occ.cut(
        objectDimTags=[(2, rect)],
        toolDimTags=[(2, circle), (2, tri_surface)],
        removeObject=True,
        removeTool=True
    )

    gmsh.model.occ.synchronize()

    # Mesh sizes (global + local refinement near the notch apex/mouth if desired)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(meshsize))
    # Optionally refine a bit around the notch for better resolution:
    gmsh.model.mesh.setSize([(0, pA1), (0, pM1), (0, pM2)], float(meshsize) * 0.5)

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)
    gmsh.finalize()

    # Non-convex interceptor set at the notch apex (your project convention)
    nonconvex_vertex = [A1[0], A1[1], 0.0]  # apex coordinates
    nci = NonConvexInterceptor(
        obt_bisec=np.array(nonconvex_vertex),
        ext_bdry=None,
        unit_normal=None,
        l_dir=None
    )
    nci.use = 'bisec'

    shape = Shape(P=None, nonconvex_interceptor=nci, msh_file=msh_file)

    key_points = {
        "A1": A1,   # apex of V-notch
        "B1": B1,   # midpoint of mouth
        "M1": M1,   # left mouth point
        "M2": M2    # right mouth point
    }

    # If your Shape class supports attaching metadata, you could also do:
    # shape.key_points = key_points

    #key_points_2=[r,cx,cy,M1,M2,A1] 
    key_points_2=[r,cx,cy,A1,B1] 
    nci = NonConvexInterceptor(obt_bisec=np.array(key_points_2,dtype=object), ext_bdry=None, unit_normal=None, l_dir=None)

     
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)




#----------------------------------------------------------

def rectangle_2d(lx=1.0, ly=1.0, meshsize=0.1, meshdata_dir='meshdata', filename_suffix='00'):
    """
    Create a 3D rectangle of size lx × ly using Gmsh (OpenCASCADE kernel).

    Parameters:
    - lx, ly: dimensions of the rectangle
    - meshsize: target mesh size
    - meshdata_dir: folder to save files
    - filename_suffix: suffix for output filenames
    """
    gmsh.initialize()
    gmsh.model.add("rectangle")

    msh_file = 'meshdata/rectangel_2d.msh'
    # Create rectangle surface using OpenCASCADE
    rect = gmsh.model.occ.addRectangle(0, 0, 0, lx, ly)

    # Synchronize the CAD kernel
    gmsh.model.occ.synchronize()

    # Set mesh size on all points (0D entities)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshsize)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)




    gmsh.write(msh_file)
    gmsh.finalize()
    

    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)



#-------------------------muri 2025--------------------------
#------------------------------------------------------------


def kalthoff_3d_plate(ix=6e-3, iy=2e-3,iz=.2e-3, TH = 1e-3, theta = 20, distance_from_corner=1e-3,conveTol=.2e-3,   meshsize=.3e-3, meshdata_dir='meshdata', filename_suffix='00'):
    '''
    In this function we find the blocks of nodes and remove all the bonds from each block 
    each V shape crack has two blocks (left and right)
    '''
#def kalthoff_3d_plate(ix=1e-3, iy=1e-3,iz=1e-3, TH = 1e-4, theta = 10, distance_from_corner=1e-4,conveTol=1e-5,   meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):


    msh_file = 'meshdata/kalthof3d.msh'
    #msh_file = meshdata_dir+'/kalthof3d1.msh'
    # ix,iy,iz are size of the plate
    # TH is the depth of the triangles
    # theta is the angle of the triangle
    gmsh.initialize()


    #set my origin in the right corner 
    #myOrigx, myOrigy, myOrigz = ix/2, iy/2, iz/2
    myOrigx, myOrigy, myOrigz = 0,0,0
    cx = -myOrigx
    cy = -myOrigy
    cz = -myOrigz
    #cx = 0 
    #cy = 0
    #cz = 0


    # Create the rectangle
    rectangle = gmsh.model.occ.addRectangle(cx, cy, cz, ix, iy)

    # Compute the base of the isosceles triangle
    triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)
    # Points and lines for the first triangle (left corner)
    p1 = gmsh.model.occ.addPoint(cx + distance_from_corner , cy + iy, cz)
    p2 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2,cy +  iy - TH  , cz)
    p3 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base  ,cy + iy, cz)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    triangle1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1, l2, l3])])

    # Points and lines for the second triangle (right corner)
    p4 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner,cy +  iy,cz )
    p5 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 ,cy +  iy - TH  , cz)
    p6 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base ,cy +  iy  , cz)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p4)
    triangle2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4, l5, l6])])

    # Cut the triangles from the rectangle
    remaining, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, triangle1)])
    gmsh.model.occ.cut(remaining, [(2, triangle2)])
    gmsh.model.occ.extrude([(2,remaining[0][1])],0,0,iz)
    
 
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);

    
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.fltk.run()
    gmsh.write(msh_file)
    gmsh.finalize()
    
    nonconvex_vertex =  creat_nonvexpart_blocks_version_beta(ix, iy,iz, TH, theta,conveTol,msh_file, distance_from_corner)
    #print(nonconvex_vertex) 
    #print(nonconvex_vertex.shape())
    

    #---------------------------Here call the nonvonce and find the inceptors

    nci = nonconvex_part_kalthoff_3d(ix, iy,iz, TH, theta, conveTol, distance_from_corner, meshsize, meshdata_dir, filename_suffix)
    
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)
#---------------------------Kalthoff Wall shape nci------------------------
#   this one is update for the wall  V shape notckes -----------------------------------
def kalthoff_3d_plate_beta0(ix=6e-3, iy=2e-3,iz=.2e-3, TH = 1e-3, theta = 20, distance_from_corner=1e-3,conveTol=.2e-3,   meshsize=.3e-3, meshdata_dir='meshdata', filename_suffix='00'):
    '''
    In this setting cracks are V sape
    and bonds that cross a surface that interset the V shape cracks, will be removed
    '''

    msh_file = 'meshdata/kalthof3d_beta0.msh'
    #msh_file = meshdata_dir+'/kalthof3d1.msh'
    # ix,iy,iz are size of the plate
    # TH is the depth of the triangles
    # theta is the angle of the triangle
    gmsh.initialize()


    #set my origin in the right corner 
    #myOrigx, myOrigy, myOrigz = ix/2, iy/2, iz/2
    myOrigx, myOrigy, myOrigz = 0,0,0
    cx = -myOrigx
    cy = -myOrigy
    cz = -myOrigz
    #cx = 0 
    #cy = 0
    #cz = 0


    # Create the rectangle
    rectangle = gmsh.model.occ.addRectangle(cx, cy, cz, ix, iy)

    # Compute the base of the isosceles triangle
    triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)
    # Points and lines for the first triangle (left corner)
    p1 = gmsh.model.occ.addPoint(cx + distance_from_corner , cy + iy, cz)
    p2 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2,cy +  iy - TH  , cz)
    p3 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base  ,cy + iy, cz)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    triangle1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1, l2, l3])])

    # Points and lines for the second triangle (right corner)
    p4 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner,cy +  iy,cz )
    p5 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 ,cy +  iy - TH  , cz)
    p6 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base ,cy +  iy  , cz)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p4)
    triangle2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4, l5, l6])])

    # Cut the triangles from the rectangle
    remaining, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, triangle1)])
    gmsh.model.occ.cut(remaining, [(2, triangle2)])
    gmsh.model.occ.extrude([(2,remaining[0][1])],0,0,iz)
    

#--------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);

    
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.fltk.run()
    gmsh.write(msh_file)
    gmsh.finalize()
    
    #nonconvex_vertex =  creat_nonvexpart_blocks_version_beta(ix, iy,iz, TH, theta,conveTol,msh_file, distance_from_corner)
    #print(nonconvex_vertex) 
    #print(nonconvex_vertex.shape())
   #-------------------------WALLLL--------------------------------- 
    A1 = [cx + distance_from_corner + triangle_base/2,cy +  iy - TH  , cz]
    B1 = [cx + distance_from_corner + triangle_base/2,cy +  iy   , cz]

    #firstWall = [A1,B1,iz]
    A2 = [cx + ix - distance_from_corner - triangle_base/2 ,cy +  iy - TH  , cz]
    B2 = [cx + ix - distance_from_corner - triangle_base/2 ,cy +  iy   , cz]
    #secondWall = [A2,B2,iz]
    z = [0,0,iz] 
    nonconvex_vertex =[A1,B1,A2,B2,z]
    #print(nonconvex_vertex)
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)

    
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)




#---------------------------Kalthof with rectangel notch ########## |_|--------------------------

def kalthoff_3d_plate_beta(ix=6e-3, iy=2e-3, iz=.2e-3, Ln=1e-3, Wn=0.5e-3,distance_from_edge=1e-3, meshsize=.3e-3, meshdata_dir='meshdata', filename_suffix='00'):
    '''
    In this setting cracks are Rectangle |_|  sape
    and bonds that cross a surface that interset the V shape cracks, will be removed
    '''


    msh_file = 'meshdata/kalthof3d_beta.msh'
    
    gmsh.initialize()
    myOrigx, myOrigy, myOrigz = 0, 0, 0
    cx, cy, cz = myOrigx, myOrigy, myOrigz

    # Create the rectangle (main plate)
    rectangle = gmsh.model.occ.addRectangle(cx, cy, cz, ix, iy)

    # Function to create a notch
    def create_notch(x, y, w, l):
        p1 = gmsh.model.occ.addPoint(x, y, cz)
        p2 = gmsh.model.occ.addPoint(x + w, y, cz)
        p3 = gmsh.model.occ.addPoint(x + w, y - l, cz)
        p4 = gmsh.model.occ.addPoint(x, y - l, cz)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        return gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1, l2, l3, l4])])

    # Create and position the notches
    notch1 = create_notch(cx + distance_from_edge, cy + iy, Wn, Ln)
    notch2 = create_notch(cx + ix - distance_from_edge - Wn, cy + iy, Wn, Ln)
    
    # Cut the notches from the rectangle
    remaining, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, notch1)], removeTool=True)
    remaining, _ = gmsh.model.occ.cut(remaining, [(2, notch2)], removeTool=True)


    # Extrude to 3D
    gmsh.model.occ.extrude([(2, remaining[0][1])], 0, 0, iz)

    # Mesh settings
    gmsh.option.setNumber("Mesh.Algorithm", 6)#9
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5)
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)


    #save the mesh
    #gmsh.fltk.run()
    gmsh.write(msh_file)
    gmsh.finalize()
    
    
    A1 = [cx + distance_from_edge + Wn/2,cy +  iy - Ln   , cz]
    B1 = [cx + distance_from_edge + Wn/2,cy +  iy   , cz]

    #firstWall = [A1,B1,iz]
    A2 = [cx + ix - distance_from_edge - Wn/2 ,cy +  iy - Ln   , cz]
    B2 = [cx + ix - distance_from_edge - Wn/2 ,cy +  iy   , cz]
    #secondWall = [A2,B2,iz]
    z = [0,0,iz] 
    nonconvex_vertex =[A1,B1,A2,B2,z]
    #print(nonconvex_vertex)
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)



    #---------------------------Here call the nonvonce and find the inceptors
    
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)


#-----------------------END     Kalthof with rectangel notch---------------------------



def cylinder(r=1e-3, h=4e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/cylender'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    # Set the model name
    gmsh.model.add("3d_cylinder")

    # Define the cylinder parameters
    radius = r # Radius of the cylinder
    height = h  # Height of the cylinder

    # Create the cylinder
    cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)

    # Synchronize the model
    gmsh.model.occ.synchronize()

    # Generate the mesh


    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0);
    gmsh.option.setNumber("General.Verbosity", 0);
                   
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()
    

    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)


def kalthoff_impactor_3d(ix=1e-3, iy=1e-3,iz=1e-3,imx =1e-3,imy = 3e-3,imz = 2e-3,meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/impactor_3d_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    # Create impactor, dmp is the initial distance of impactor from plate  
     
    dmp = 5
    myOrigx, myOrigy, myOrigz = ix/2, iy/2, iz/2 
    cx = myOrigx
    cy = myOrigy 
    cz = myOrigz 
  

    box_start_x = cx + (ix-imx)/2 
    box_start_y = cy + iy + dmp
    box_start_z = cz + (iz-imz)/2


    impactor = gmsh.model.occ.addBox(box_start_x,box_start_y,box_start_z,imx,imy,imz)
   
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);
                   
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()
    

    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)


# Define a vectorized function to check if points are between parallel lines
def are_points_between_lines( line1, line2, points):
    # Unpack the line endpoints
    (x0, y0), (x1, y1) = line1
    (x2, y2) = line2

    # Calculate the direction vector of the lines
    A = np.array([x1 - x0, y1 - y0])

    # Calculate vectors from the line1 and line2 to the points
    B1 = points[:,:2] - np.array([x0, y0])
    B2 = points[:,:2] - np.array([x2, y2])

    # Compute the cross product z-components
    z1 = A[0] * B1[:, 1] - A[1] * B1[:, 0]
    z2 = A[0] * B2[:, 1] - A[1] * B2[:, 0]

    # Check if points are between the lines
    #return(z1>=0)&(z2<=0)|(z1<=0)&(z2>=0) 
    return (z1 * z2) <= 0

def creat_nonvexpart_blocks_version_beta(ix, iy,iz, TH, theta,conveTol,msh_file, distance_from_corner,plot_nodes=True):

        mesh = meshio.read(msh_file)
        pos = mesh.points

        #set my origin in the right corner
        #myOrigx, myOrigy, myOrigz = ix/2, iy/2,iz/2
        myOrigx, myOrigy, myOrigz = 0, 0,0
        cx = myOrigx
        cy = myOrigy

        cz = myOrigz

        # Create the rectangle

        # Compute the base of the isosceles triangle
        triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)
        a = triangle_base
        M = distance_from_corner
        n = conveTol
        h = TH
        # Define a threshold or reference x-coordinate to separate left and right sides.
        p1x = cx+M
        p1y = cy+iy
        p2x = cx+M+a/2
        p2y = cy+iy-h
        p3x = cx+M+a
        p3y = cy+iy
        p4x = cx+ix-M-a
        p4y = cy+iy+n
        p5x = cx+ix-M-a/2
        p5y = cy+iy-h
        p6x = cx+ix-M
        p6y = cy+iy

        q1x = p2x-n
        q1y = p2y
        q2x = p2x+n
        q2y = p2y
        q3x = p5x-n
        q3y = p5y
        q4x = p5x+n
        q4y = p5y

        # Assume your sets of parallel lines are defined as follows:
        # Each set is a tuple with a point on the first line, a point on the second line, and an offset for the second line
        line_sets = [
            ((p1x,p1y), (p2x,p2y),(q1x,q1y)),
            ((p3x,p3y), (p2x,p2y),(q2x,q2y)),
            ((p4x,p4y), (p5x,p5y),(q3x,q3y)),
            ((p6x,p6y), (p5x,p5y),(q4x,q4y))
            ]
        #print(line_sets)
        # Convert points from mesh.pos to a NumPy array
        mesh_points = np.array(pos)
        filtered_points = mesh_points[mesh_points[:,1]>=q2y]
        #print(filtered_points)
        # Initialize lists to hold points for each set
        points_in_sets = [[] for _ in range(len(line_sets))]

        # Categorize points for each set of lines
        for i, (p1,p2,p3) in enumerate(line_sets):
           line1 = (p1,p2)
           line2 = p3
           # Check if points are between the current set of lines
           mask = are_points_between_lines( line1, line2, filtered_points)
           # Add the points to the corresponding list
           points_in_sets[i].extend(filtered_points[mask])
        # 'left_vertices' and 'right_vertices' now contain the coordinates of the left and right vertices, respectively.
        bl1 =  points_in_sets[0]
        bl2 =  points_in_sets[1]
        br1 =  points_in_sets[2]
        br2 =  points_in_sets[3]
 
        # find the maximum length among the lists
        max_length = max(len(bl1), len(bl2), len(br1), len(br2))
        #print("the is      second block")
        #print(br2)
                  # Initialize empty lists to store the concatenated 3D coordinates
        result = []

        # Concatenate the four lists horizontally
        for i in range(max_length):
            if i < len(bl1):
                result.extend(bl1[i])
            else:
                result.extend([None, None, None])
            
            if i < len(bl2):
                result.extend(bl2[i])
            else:
                result.extend([None, None, None])
            
            if i < len(br1):
                result.extend(br1[i])
            else:
                result.extend([None, None, None])
            
            if i < len(br2):
                result.extend(br2[i])
            else:
                result.extend([None, None, None])

        #print(result) 


        if plot_nodes:
            print(points_in_sets)
            removed_nodes =  np.array([point for point in mesh_points if not any(np.array_equal(point, p) for p in points_in_sets[0])])
            removed_nodes =  np.array([point for point in removed_nodes if not any(np.array_equal(point, p) for p in points_in_sets[1])])
            removed_nodes =  np.array([point for point in removed_nodes if not any(np.array_equal(point, p) for p in points_in_sets[2])])
            removed_nodes =  np.array([point for point in removed_nodes if not any(np.array_equal(point, p) for p in points_in_sets[3])])



            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            x0,y0,z0 = zip(*points_in_sets[0])
            x1,y1,z1 = zip(*points_in_sets[1])
            x2,y2,z2 = zip(*points_in_sets[2])
            x3,y3,z3 = zip(*points_in_sets[3])
            x4,y4,z4 = zip(*removed_nodes)
            x5,y5,z5 = zip(*mesh_points)

            ax.scatter(x0,y0,z0,color='blue',marker = '*',s=2)
            ax.scatter(x1,y1,z1,color='blue',marker = '*',s=2)
            ax.scatter(x2,y2,z2,color='blue',marker = '*',s=2)
            ax.scatter(x3,y3,z3,color='blue',marker = '*',s=2)
            ax.scatter(x4,y4,z4,color='red',marker = '*',s=.5)

            #ax.scatter(x5,y5,z5,color='red',s = 1)
            plt.show()

        return result 

def creat_nonvexpart_blocks(ix, iy,iz, TH, theta,conveTol,msh_file, distance_from_corner):
        mesh = meshio.read(msh_file)
        pos = mesh.points

        #set my origin in the right corner 
        myOrigx, myOrigy, myOrigz = ix/2, iy/2,iz/2
        cx = myOrigx
        cy = myOrigy

        cz = myOrigz

        # Create the rectangle

        # Compute the base of the isosceles triangle
        triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)

        nConv = conveTol
        # Define a threshold or reference x-coordinate to separate left and right sides.
        threshold_x1 = cx + distance_from_corner + triangle_base/2
        threshold_x2 = cx + distance_from_corner + triangle_base + nConv
        threshold_x3 = cx + ix - distance_from_corner - triangle_base - nConv
        threshold_x4 = cx + ix - distance_from_corner - triangle_base/2
        #
        # Initialize empty lists to store left and right vertices.
        left_notch_blk1 = []
        left_notch_blk2 = []
        right_notch_blk1 = []
        right_notch_blk2 = []
        siz = len(pos)

        # Iterate through all vertices and categorize them based on their x-coordinate.
        for vertex in pos:
            x_coordinate = vertex[0]  # Assuming x-coordinate is in the first column (column 0).
            if x_coordinate <= threshold_x1:
             left_notch_blk1.append(vertex)
            elif  x_coordinate >= threshold_x1 and x_coordinate<= threshold_x2:
                 left_notch_blk2.append(vertex)
            elif  x_coordinate <= threshold_x4 and x_coordinate >= threshold_x3:
                 right_notch_blk1.append(vertex)
            else:
                 right_notch_blk2.append(vertex)
        # 'left_vertices' and 'right_vertices' now contain the coordinates of the left and right vertices, respectively.
        bl1 = left_notch_blk1
        bl2 = left_notch_blk2
        br1 = right_notch_blk1
        br2 = right_notch_blk2
 
        # find the maximum length among the lists
        max_length = max(len(bl1), len(bl2), len(br1), len(br2))
        #print("the is      second block")
        #print(br2)
                  # Initialize empty lists to store the concatenated 3D coordinates
        result = []

        # Concatenate the four lists horizontally
        for i in range(max_length):
            if i < len(bl1):
                result.extend(bl1[i])
            else:
                result.extend([None, None, None])
            
            if i < len(bl2):
                result.extend(bl2[i])
            else:
                result.extend([None, None, None])
            
            if i < len(br1):
                result.extend(br1[i])
            else:
                result.extend([None, None, None])
            
            if i < len(br2):
                result.extend(br2[i])
            else:
                result.extend([None, None, None])

        #print(result) 

        return result 



def nonconvex_part_kalthoff_3d(ix, iy,iz, TH, theta, non_convex_Tol, distance_from_corner,  meshsize, meshdata_dir, filename_suffix):
    
    msh_file = meshdata_dir+'/nonConvxKalthof'+'.msh'
    #msh_file = meshdata_dir+'/nonconvex_kalthof_3d'+'.msh'
    # ix,iy, iz are size of the plate
    # TH is the depth of the triangles
    # theta is the angle of the triangle
    gmsh.initialize()
    gmsh.model.add("nonconvex_kalthof_3d")
    nConv = non_convex_Tol 

    #set my origin in the right corner 
    #myOrigx, myOrigy, myOrigz = ix/2, iy/2,iz/2 
    myOrigx, myOrigy, myOrigz = 0,0,0 
    cx = myOrigx
    cy = myOrigy 
    cz = myOrigz 
  
    # Create the rectangle

    # Compute the base of the isosceles triangle
    triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)
    # Points and lines for the first triangle (left corner)
    p1 = gmsh.model.occ.addPoint(cx + distance_from_corner, cy + iy, cz)
    p2 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2, cy + iy - TH, cz)
    p3 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base, cy + iy, cz)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    triangle1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1, l2, l3])])

    # Nonconvex part (extend V in left corner)
    p1extend = gmsh.model.occ.addPoint(cx + distance_from_corner - nConv, cy + iy, cz)
    p2extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2 - nConv, cy + iy - TH, cz)
    p4extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base + nConv, cy + iy, cz)
    p3extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2 + nConv, cy + iy - TH, cz)
    l1extend = gmsh.model.occ.addLine(p1extend, p2extend)
    l2extend = gmsh.model.occ.addLine(p2extend, p3extend)
    l3extend = gmsh.model.occ.addLine(p3extend, p4extend)
    l4extend = gmsh.model.occ.addLine(p4extend, p1extend)
    trapazoil1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1extend, l2extend, l3extend, l4extend])])

    # Points and lines for the second triangle (right corner)
    p4 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner, cy + iy, cz)
    p5 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2, cy + iy - TH, cz)
    p6 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base, cy + iy, cz)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p4)
    triangle2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4, l5, l6])])

    # Nonconvex part (extend V in right corner)
    p4extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner + nConv, cy + iy, cz)
    p5extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 + nConv, cy + iy - TH, cz)
    p6extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 - nConv, cy + iy - TH, cz)
    p7extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base - nConv, cy + iy, cz)
    l4extend = gmsh.model.occ.addLine(p4extend, p5extend)
    l5extend = gmsh.model.occ.addLine(p5extend, p6extend)
    l6extend = gmsh.model.occ.addLine(p6extend, p7extend)
    l7extend = gmsh.model.occ.addLine(p7extend, p4extend)
    trapazoil2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4extend, l5extend, l6extend, l7extend])])

    # Cut the triangles from the rectangle
    remaining1, _ = gmsh.model.occ.cut([(2, trapazoil1)], [(2, triangle1)])
    remaining2, _ = gmsh.model.occ.cut([(2, trapazoil2)], [(2, triangle2)])

    gmsh.model.occ.extrude([(2, remaining1[0][1])], 0, 0, iz)
    gmsh.model.occ.extrude([(2, remaining1[1][1])], 0, 0, iz)
    gmsh.model.occ.extrude([(2, remaining2[0][1])], 0, 0, iz)
    gmsh.model.occ.extrude([(2, remaining2[1][1])], 0, 0, iz)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.fltk.run()
    gmsh.write(msh_file)
    gmsh.finalize()
    
    nonconvex_vertex =  creat_nonvexpart_blocks(ix, iy,iz, TH, theta,nConv,msh_file, distance_from_corner)
    

    return NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)
#------------------------------------------------------------------------------------------


def creat_nonvexpart_original(ix, iy, TH, theta, wz, imx, imy, imz, conveTol, distance_from_corner,meshsize):
    gmsh.initialize()
    gmsh.model.add("test")

    nConv = conveTol

    #set my origin in the right corner 
    myOrigx, myOrigy, myOrigz = ix/2, iy/2,wz/2 
    cx = myOrigx
    cy = myOrigy 

    cz = myOrigz 
  
    # Create the rectangle

    # Compute the base of the isosceles triangle
    triangle_base = 2 * TH * math.tan(math.radians(theta) / 2)
    # Points and lines for the first triangle (left corner)
    p1 = gmsh.model.occ.addPoint(cx + distance_from_corner, cy + iy, cz)
    p2 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2, cy + iy - TH, cz)
    p3 = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base, cy + iy, cz)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    triangle1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1, l2, l3])])

    # Nonconvex part (extend V in left corner)
    p1extend = gmsh.model.occ.addPoint(cx + distance_from_corner - nConv, cy + iy, cz)
    p2extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2 - nConv, cy + iy - TH, cz)
    p4extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base + nConv, cy + iy, cz)
    p3extend = gmsh.model.occ.addPoint(cx + distance_from_corner + triangle_base/2 + nConv, cy + iy - TH, cz)
    l1extend = gmsh.model.occ.addLine(p1extend, p2extend)
    l2extend = gmsh.model.occ.addLine(p2extend, p3extend)
    l3extend = gmsh.model.occ.addLine(p3extend, p4extend)
    l4extend = gmsh.model.occ.addLine(p4extend, p1extend)
    trapazoil1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l1extend, l2extend, l3extend, l4extend])])

    # Points and lines for the second triangle (right corner)
    p4 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner, cy + iy, cz)
    p5 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2, cy + iy - TH, cz)
    p6 = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base, cy + iy, cz)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p4)
    triangle2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4, l5, l6])])

    # Nonconvex part (extend V in right corner)
    p4extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner + nConv, cy + iy, cz)
    p5extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 + nConv, cy + iy - TH, cz)
    p6extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base/2 - nConv, cy + iy - TH, cz)
    p7extend = gmsh.model.occ.addPoint(cx + ix - distance_from_corner - triangle_base - nConv, cy + iy, cz)
    l4extend = gmsh.model.occ.addLine(p4extend, p5extend)
    l5extend = gmsh.model.occ.addLine(p5extend, p6extend)
    l6extend = gmsh.model.occ.addLine(p6extend, p7extend)
    l7extend = gmsh.model.occ.addLine(p7extend, p4extend)
    trapazoil2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addWire([l4extend, l5extend, l6extend, l7extend])])

    # Cut the triangles from the rectangle
    remaining1, _ = gmsh.model.occ.cut([(2, trapazoil1)], [(2, triangle1)])
    remaining2, _ = gmsh.model.occ.cut([(2, trapazoil2)], [(2, triangle2)])

    gmsh.model.occ.extrude([(2, remaining1[0][1])], 0, 0, wz)
    gmsh.model.occ.extrude([(2, remaining1[1][1])], 0, 0, wz)
    gmsh.model.occ.extrude([(2, remaining2[0][1])], 0, 0, wz)
    gmsh.model.occ.extrude([(2, remaining2[1][1])], 0, 0, wz)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);
    
    msh_file = "meshdata/kir0.msh"
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()
    points = 2
 
    '''
    d = creat_nonvexpart_blocks(ix, iy, TH, theta,wz,imx,imy,imz,conveTol,meshfile, distance_from_corner)
    print(len(d))
    print(d[1])     
    '''
    return gmsh.model


#----------------------------plus 3D-----------------------------------------------------

# def create_cube(x, y, z, size):
#     """
#     Function to create a cube at position (x, y, z) with the specified size.
#     """
#     return gmsh.model.occ.addBox(x, y, z, size, size, size)
#

def randomConvex3d(n_shape=20, r_sphere=10,meshsize=5, meshdata_dir='meshdata', filename_suffix='00'):
    msh_file = meshdata_dir+'/random_convex_3d_'+str(filename_suffix)+'.msh'

    gmsh.initialize()

    # Create a convex hull of random points which will have approximately n_shape faces
    points = np.random.rand(n_shape, 3) * 2 - 1  # points in [-1, 1]^3
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]  # project onto unit sphere
    points *= r_sphere  # scale to the desired sphere radius

    # Compute the convex hull using scipy
    hull = ConvexHull(points)

    # Add points to GMSH
    gmsh_point_tags = []   
    for p in hull.points:
        tag = gmsh.model.geo.addPoint(p[0], p[1], p[2], meshSize=1)
        gmsh_point_tags.append(tag)

    # Create the faces using the hull simplices
    for simplex in hull.simplices:
        curve_loop_tags = []
        for i in range(3):
            start_point_tag = gmsh_point_tags[simplex[i-1]]
            end_point_tag = gmsh_point_tags[simplex[i % 3]]
            line_tag = gmsh.model.geo.addLine(start_point_tag, end_point_tag)
            curve_loop_tags.append(line_tag)
        curve_loop_tag = gmsh.model.geo.addCurveLoop(curve_loop_tags)
        gmsh.model.geo.addPlaneSurface([curve_loop_tag])


    # Generate mesh


    # Finalize GMSH
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1);
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    #gmsh.option.setNumber("Mesh.ToleranceInitialDelauny", 1e-3);
    gmsh.option.setNumber("General.Verbosity", 0);
                   
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()



    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)



def create_cube(x, y, z, lx, ly=1e-3, lz=1e-3):
    return gmsh.model.occ.addBox(x, y, z, lx, ly, lz)

def plus3d_longHand(l=1e-3, l2=2e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):
    msh_file = meshdata_dir + '/plus3d_' + str(filename_suffix) + '.msh'
    gmsh.initialize()

    cube_size = l
    arm_length = l2

    # Central cube
    central_cube = create_cube(-cube_size/2, -cube_size/2, -cube_size/2, cube_size, cube_size, cube_size)

    # Arm cubes
    arm1 = create_cube(-cube_size/2, cube_size/2, -cube_size/2, cube_size, arm_length, cube_size)  # Positive y direction
    arm2 = create_cube(-cube_size/2, -cube_size/2 - arm_length, -cube_size/2, cube_size, arm_length, cube_size)  # Negative y direction
    arm3 = create_cube(cube_size/2, -cube_size/2, -cube_size/2, arm_length, cube_size, cube_size)  # Positive x direction
    arm4 = create_cube(-cube_size/2 - arm_length, -cube_size/2, -cube_size/2, arm_length, cube_size, cube_size)  # Negative x direction

    # Fuse the cubes to form the plus shape
    gmsh.model.occ.synchronize()
    entities = gmsh.model.occ.fuse([(3, central_cube)], [(3, arm1), (3, arm2), (3, arm3), (3, arm4)])

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5)
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    # Save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()
    # Define the intersection points
    pq11 = [l/2, l/2, l/2]
    pq12 = [l/2, l/2, -l/2]
    pq21 = [-l/2, l/2, l/2]
    pq22 = [-l/2, l/2, -l/2]
    pq31 = [-l/2, -l/2, l/2]
    pq32 = [-l/2, -l/2, -l/2]
    pq41 = [l/2, -l/2, l/2]
    pq42 = [l/2, -l/2, -l/2]
    
    # Define the vertexes
    pq1v1=[l/2+l2,l/2,l/2]
    pq1v2=[l/2,l/2+l2,-l/2]
    pq2v1=[-l/2-l2,l/2,l/2]
    pq2v2=[-l/2,l/2+l2,-l/2]
    pq3v1=[-l/2-l2,-l/2,l/2]
    pq3v2=[-l/2,-l/2-l2,-l/2]
    pq4v1=[l/2+l2,-l/2,l/2]
    pq4v2=[l/2,-l/2-l2,-l/2]
    # Intersection points
    t = 0.5  # Midpoint    
    pq10 = find_point_on_line(pq1v1, pq1v2, t)
    pq20 = find_point_on_line(pq2v1, pq2v2, t)
    pq30 = find_point_on_line(pq3v1, pq3v2, t)
    pq40 = find_point_on_line(pq4v1, pq4v2, t)

    # Combine all points into a list
    nonconvex_vertex = [pq11, pq12, pq21, pq22, pq31, pq32, pq41, pq42, pq10, pq20, pq30, pq40]
       
    
    
    print(nonconvex_vertex)
    nci = NonConvexInterceptor(obt_bisec=np.array(nonconvex_vertex), ext_bdry=None, unit_normal=None, l_dir=None)



   
    
    nci.use = 'bisec'
    return Shape(P=None,nonconvex_interceptor=nci, msh_file=msh_file)
   
def find_point_on_line(p1, p2, t):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    z = (1 - t) * z1 + t * z2
    
    return [x, y, z]



def plus3d(l = 1e-3,meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):
   
    msh_file = meshdata_dir+'/plus3d_'+str(filename_suffix)+'.msh'
    gmsh.initialize()
 

    cube_size = l

    # Central cube
    central_cube = create_cube(-cube_size/2, -cube_size/2, -cube_size/2, cube_size)

    # Arm cubes
    arm1 = create_cube(-cube_size/2, cube_size/2, -cube_size/2, cube_size)
    arm2 = create_cube(-cube_size/2, -1.5*cube_size, -cube_size/2, cube_size)
    arm3 = create_cube(cube_size/2, -cube_size/2, -cube_size/2, cube_size)
    arm4 = create_cube(-1.5*cube_size, -cube_size/2, -cube_size/2, cube_size)

    # Fuse the cubes to form the plus shape
    entities = gmsh.model.occ.fuse([(3, central_cube)], [(3, arm1), (3, arm2), (3, arm3), (3, arm4)])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);
                   
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()


    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)

#----------------------------plus 3D end--------------------------------------------------


def sphere_small_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/3d_sphere_small.msh')


def cube_3d(lx=1e-3, ly=1e-3,lz=1e-3,meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):


    msh_file = meshdata_dir+'/cube3d_'+str(filename_suffix)+'.msh'
    gmsh.initialize()
    # Create corner points for the cube
    lc = meshsize
    ce = lx/2
    p1 = gmsh.model.geo.addPoint(-ce, -ce, -ce, lc)
    p2 = gmsh.model.geo.addPoint(lx-ce, -ce, -ce, lc)
    p3 = gmsh.model.geo.addPoint(lx-ce,ly-ce, -ce, lc)
    p4 = gmsh.model.geo.addPoint(-ce, ly-ce, -ce, lc)
    # Create lines connecting the points
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    curve_loop1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface1 = gmsh.model.geo.addPlaneSurface([curve_loop1])
    gmsh.model.geo.extrude([(2,surface1)],0,0,lz)
    
    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);
    gmsh.option.setNumber("General.Verbosity", 0);
                   
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    #save the mesh
    gmsh.write(msh_file)
    gmsh.finalize()


    return Shape(P=None,nonconvex_interceptor=None, msh_file=msh_file)


def _sphere_3d(rad=1e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):


    msh_file = meshdata_dir+'/sphere_'+str(filename_suffix)+'.msh'
    gmsh.initialize()
    try:
        # --- Geometry: just the sphere (no point at center) ---
        # returns (volume_tag), we’ll keep tag=1
        vol = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, rad, tag=1)

        # OCC fixes to avoid tiny slivers/faces/edges created by CAD kernel
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # --- Mesh algorithms & size control ---
        # 2D surface meshing Frontal-Delaunay; 3D HXT (good quality & speed)
        gmsh.option.setNumber("Mesh.Algorithm", 6)     # 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # 3D

        # Make mesh size uniform: ignore points/curvature sizing
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)

        # Optional: keep the number of elements around great circles bounded
        # (helps avoid “too many” segments near poles)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

        # Provide a constant background field = meshsize (most robust way)
        f = gmsh.model.mesh.field
        fid = f.add("Constant")
        f.setNumber(fid, "LC", meshsize)
        f.setAsBackgroundMesh(fid)

        # Some cleanup/quality options
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("General.Verbosity", 0)

        # It can help to set surface sizes too (belt-and-suspenders):
        # get boundary surfaces of the volume and apply mesh size
        surfs = gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)
        gmsh.model.mesh.setSize(surfs, meshsize)

        # --- Generate surface then volume mesh ---
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.generate(3)

        gmsh.write(msh_file)

    finally:
        gmsh.finalize()

    # Return a Shape pointing to this .msh (no scaling needed; we built exact radius)
    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def box_extrude_3d(xyz=[0,0,0], L1=1e-3, L2=1e-3, L3=1e-3, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/box_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addRectangle(xyz[0], xyz[1], xyz[2], L1, L2, tag=1)
    nz = L3/meshsize

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 1)], 0, 0, L3, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def cylinder_3d_extrude(xyz=[0,0,0], rad=1e-3, L3=1e-3, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/cyl_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    gmsh.model.occ.addCircle(xyz[0], xyz[1], xyz[2], rad, tag=1)

    # gmsh.model.occ.addCircle(0, 0, -h/2, r_sc*R, tag=2)
    # nz = L3/meshsize

    # # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # # numElements is the number of steps by which we extrude, heights is the scaling factor
    # gmsh.model.occ.extrude([(2, 1)], 0, 0, L3, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def cylinder_3d(xyz=[0,0,0], rad=1e-3, L3=1e-3, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00', direction='z'):

    msh_file = meshdata_dir+'/box_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    # gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r, tag = -1, angle = 2*pi)
    #
    # Add a cylinder in the OpenCASCADE CAD representation, defined by the center (x, y, z) of its first circular face, the 3 components (dx, dy, dz) of the vector defining its axis and its radius r. The optional angle argument defines the angular opening (from 0 to 2*Pi). If tag is positive, set the tag explicitly; otherwise a new tag is selected automatically. Return the tag of the cylinder.

    if direction == 'z':
        gmsh.model.occ.addCylinder(xyz[0], xyz[1], xyz[2], 0, 0, L3, rad, tag=1)
    elif direction == 'y':
        gmsh.model.occ.addCylinder(xyz[0], xyz[1], xyz[2], 0, L3, 0, rad, tag=1)
    elif direction == 'x':
        gmsh.model.occ.addCylinder(xyz[0], xyz[1], xyz[2], L3, 0, 0, rad, tag=1)
    # nz = L3/meshsize

    # # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # # numElements is the number of steps by which we extrude, heights is the scaling factor
    # gmsh.model.occ.extrude([(2, 1)], 0, 0, L3, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def box_3d(xyz=[0,0,0], L1=1e-3, L2=1e-3, L3=1e-3, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/box_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addBox(xyz[0], xyz[1], xyz[2], L1, L2, L3)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def torus_inscribed_3d(xyz=[0,0,0], r1=1e-3, r2=0.2e-3, meshsize=0.1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/torus_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addTorus(xyz[0], xyz[1], xyz[2], r1-r2, r2, angle=2*np.pi)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def torus_3d(xyz=[0,0,0], r1=1e-3, r2=0.2e-3, meshsize=0.1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/torus_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addTorus(xyz[0], xyz[1], xyz[2], r1, r2, angle=2*np.pi)

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def cube_extrude_3d(L=20e-3, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/cube_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addRectangle(-L/2, -L/2, -L/2, L, L, tag=1)

    # gmsh.model.occ.addCurveLoop([1], tag=2)
    # gmsh.model.occ.addPlaneSurface([1], tag=1)

    nz = L/meshsize

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 1)], 0, 0, L, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def rect_w_cut_3d(R=1e-3, h_sc=0.2, crack_l=0.7, crack_w=0.05, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/rect_w_hole_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    h = R * h_sc
    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    # gmsh.model.occ.addCircle(0, 0, -h/2, R, tag=1)
    # gmsh.model.occ.addCircle(0, 0, -h/2, r_sc*R, tag=2)
    gmsh.model.occ.addRectangle(-R/2, -R/2, -h/2, R, R, tag=1)
    gmsh.model.occ.addRectangle(-R/2*crack_l, -R/2*crack_w, -h/2, R*crack_l, R*crack_w, tag=2)

    gmsh.model.occ.cut([(2, 1)], [(2, 2)], tag=3)

    ## Caution: addRectangle creats CurveLoop whereas addCircle does not.
    # gmsh.model.occ.addCurveLoop([1], tag=1)
    # gmsh.model.occ.addCurveLoop([2], tag=2)
    # gmsh.model.occ.addPlaneSurface([2, 1], tag=3)

    nz = h/meshsize 

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 3)], 0, 0, h/2, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)

def disk_w_hole_prog_3d(R=1e-3, height_sc=0.2, gamma=0.2, meshsize=0.5e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/cube_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    h = R * height_sc
    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addCircle(0, 0, -h/2, R, tag=1)
    gmsh.model.occ.addCircle(0, 0, -h/2, gamma*R, tag=2)


    gmsh.model.occ.addCurveLoop([1], tag=1)
    gmsh.model.occ.addCurveLoop([2], tag=2)
    gmsh.model.occ.addPlaneSurface([1, 2], tag=3)

    nz = h/meshsize

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 3)], 0, 0, h/2, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1); 
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def sheet_rect_w_gap_3d(L=20e-3, h=2e-3, gap_l_ratio=0.5, meshsize=1e-3, meshdata_dir='meshdata', filename_suffix='00'):

    msh_file = meshdata_dir+'/cube_'+str(filename_suffix)+'.msh'
    gmsh.initialize()

    ## See: https://gmsh.info/doc/texinfo/gmsh.html#index-gmsh_002fmodel_002focc_002faddSphere
    ## for more gmsh commands  and examples with python
    gmsh.model.occ.addRectangle(-L/2, -L/2, -h/2, L, L, tag=1)

    gmsh.model.occ.addRectangle(-L/2*gap_l_ratio, -L/2*gap_l_ratio, -h/2, L*gap_l_ratio, L*gap_l_ratio, tag=2)

    # gmsh.model.occ.addCurveLoop([1], tag=1)
    # gmsh.model.occ.addCurveLoop([2], tag=2)
    gmsh.model.occ.addPlaneSurface([1, 2], tag=3)

    nz = L/meshsize 

    # extruding the line: dimension 2, tag 1 i.e., (2,1)
    # 2nd, 3rd, 4th arguments are x, y, z values to which to extrude
    # numElements is the number of steps by which we extrude, heights is the scaling factor
    gmsh.model.occ.extrude([(2, 3)], 0, 0, h/2, numElements=[nz], heights=[1])

    gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1);
    gmsh.option.setNumber("General.Verbosity", 0);

    gmsh.model.occ.synchronize() # obligatory before generating the mesh
    gmsh.model.mesh.generate(3) # generate a 3D mesh...
    gmsh.write(msh_file) # save to file
    gmsh.finalize() # close gmsh, as opposed to initialize()

    return Shape(P=None, nonconvex_interceptor=None, msh_file=msh_file)


def sphere_small_3d_bad():
    print("Caution: bad msh because the .geo has a point in the center")
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/3d_sphere_small_bad.msh')

def disk_w_hole_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/disk_w_hole_small.msh')

def ring_sharp_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/ring_sharp.msh')
def ring_sharp_finer_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/ring_sharp_finer.msh')

def sphere_unit_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/3d_sphere_unit.msh')

def plus_small_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/3d_plus_small.msh')
def sphere_small_3d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/3d/3d_sphere_small.msh')


#######################################################################
#                            from peri-ice                            #
#######################################################################

def load_shape_index(ind=0, loc='meshdata/shapes/', scaling=1, downsample=1):
    file_pre = 'PointsOut'
    filename = loc + file_pre + str(ind) + '.csv'

    P = scaling * np.genfromtxt(filename, delimiter=' ')

    # trim every other row n times to smoothen out the boundary
    if downsample:
        for i in range(downsample):
            P = np.delete(P, list(range(0, P.shape[0], 2)), axis=0)
    # print(P)

    mean = np.mean(P, axis=0)
    # print('mean', mean)
    rad = np.max(np.sqrt(np.sum( (P - mean)**2 , axis=1)))
    print('shape radius', rad)

    nci = gen_nonconvex_interceptors(P)
    # nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def load_boundary_index(ind=0, loc='meshdata/boundaries/', scaling=1, downsample=1):
    file_pre = 'PointsOutV'
    filename = loc + file_pre + str(ind) + '.csv'

    P = scaling * np.genfromtxt(filename, delimiter=' ')

    # trim every other row n times to smoothen out the boundary
    if downsample:
        for i in range(downsample):
            P = np.delete(P, list(range(0, P.shape[0], 2)), axis=0)
    # print(P)

    mean = np.mean(P, axis=0)
    # print('mean', mean)
    rad = np.max(np.sqrt(np.sum( (P - mean)**2 , axis=1)))
    print('shape radius', rad)

    # nci = gen_nonconvex_interceptors(P)
    # nci.use = 'bisec'
    # print(nci)
    return Shape(P, nonconvex_interceptor=None)
