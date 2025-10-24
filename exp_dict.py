import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
# to plot a collection of lines
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
# from gekko import GEKKO

#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import combinations
# from functools import partial
import time
from random import seed
from random import random

import shape_dict
import material_dict
from genmesh import genmesh
from arrangements import get_incenter_mesh_loc
from scipy.spatial import cKDTree
from multiprocessing import Pool, get_context


import pdb
#pdb.set_trace()
#######################################################################
class Material(object):
    """docstring for Material"""
    def __init__(self, delta, rho, snot, cnot, bulk_modulus, E = None, nu = None, Gnot = None, shear_modulus = None, name=None):
        super(Material, self).__init__()
        self.delta  = delta
        self.rho    = rho     
        self.snot   = snot
        self.cnot   = cnot
        self.bulk_modulus = bulk_modulus

        self.E = E
        self.nu = nu
        self.Gnot = Gnot
        self.shear_modulus = shear_modulus

        self.name = name

def peri_deformable(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9 * K_scale
    shear_modulus = 1.33e+09 * G_scale
    rho=1200.0	* rho_scale
    Gnot = 135.0 * Gnot_scale

    nu = 1/3
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

#######################################################################


def single_bond_ext(ij_pair, pos, delta, remove_ncvx_bonds, interceptor):
    i = ij_pair[0]
    j = ij_pair[1]

    p_i = pos[i]
    p_j = pos[j]
    d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
    if (d <= delta):
        ##### remove nonconvex bonds
        if remove_ncvx_bonds:
            intersects = False
            for k in range(len(interceptor)):
                # print(k)
                A1 = interceptor[k][0:2]
                A2 = interceptor[k][2:4]
                if (lines_intersect(A1, A2, p_i, p_j)):
                    intersects = True
                    break
            if not intersects:
                return [i,j]
            # else:
                # return None
        else:
            return [i,j]
    # else:
        # return None



#-----------------intersection of line segment with 3d wall-------------
# z is in the direction of z axis
def wall_line_intrsect(p1, p2, p6, p7, z_extrusion):
    """
    Check if the line segment from p6 to p7 intersects with a wall created by extruding the line segment p1-p2 in the Z direction.
    The wall is perpendicular to the X-axis and parallel to the Y and Z axes.
    """
    # X-coordinate of the wall
    wall_x = p1[0]

    # X-coordinates of the line segment
    x6, x7 = p6[0], p7[0]
    # Check if the line segment crosses the X-coordinate of the wall
    if min(x6, x7) <= wall_x <= max(x6, x7):
        # Calculate the parameter t for the intersection point
        t = (wall_x - x6) / (x7 - x6)

        # Intersection point in 3D
        intersection_3d = np.array(p6) + t * (np.array(p7) - np.array(p6))

        # Check if the intersection point's Y and Z coordinates are within the wall's bounds
        y_bounds = sorted([p1[1], p2[1]])
        z_bounds = sorted([p1[2], p1[2] + z_extrusion])
        if y_bounds[0] <= intersection_3d[1] <= y_bounds[1] and z_bounds[0] <= intersection_3d[2] <= z_bounds[1]:
           return True
        
    return False 

#z is not necesseraily in the directionof z axis


def wall_line_intersect_general(p1, p2, p6, p7, z):
    """
    Check if the line segment from p6 to p7 intersects with a wall created by extruding the line segment p1-p2 
    in the direction of point z. The wall is defined by the plane containing p1, p2, and p1+z.
    """
    # Calculate the direction vector for the extrusion
    extrusion_direction = np.array(z) - np.array(p1)

    # Calculate the normal vector of the plane (p1, p2, p1+extrusion_direction)
    p1p2 = np.array(p2) - np.array(p1)
    normal_vector = np.cross(p1p2, extrusion_direction)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Line segment from p6 to p7 in parametric form: p(t) = p6 + t * (p7 - p6)
    line_direction = np.array(p7) - np.array(p6)

    # Check if the line segment is parallel to the plane
    dot_product = np.dot(normal_vector, line_direction)
    if abs(dot_product) < 1e-10:  # Handle numerical stability
        return False  # The line is parallel to the plane and doesn't intersect

    # Calculate the parameter t for the intersection point
    t = np.dot(normal_vector, np.array(p1) - np.array(p6)) / dot_product

    # Check if the intersection occurs within the segment (0 <= t <= 1)
    if not (0 <= t <= 1):
        return False  # The intersection is outside the line segment

    # Intersection point in 3D
    intersection_3d = np.array(p6) + t * line_direction

    # Check if the intersection point's projection onto the p1-p2 line is within bounds
    p1_to_intersection = np.dot(intersection_3d - np.array(p1), p1p2) / np.dot(p1p2, p1p2)
    if 0 <= p1_to_intersection <= 1:
        # Check if the intersection point's projection onto the extrusion direction is within bounds
        p1_to_intersection_in_extrusion = np.dot(intersection_3d - np.array(p1), extrusion_direction) / np.dot(extrusion_direction, extrusion_direction)
        if 0 <= p1_to_intersection_in_extrusion <= 1:
            return True

    return False

def line_sphere_intersection(p1, p2, cx, cy, cz, r):
    # Vector from p1 to p2
    dp = np.array(p2) - np.array(p1)

    # Vector from center of the sphere to p1
    f = np.array(p1) - np.array([cx, cy, cz])

    # Coefficients of the quadratic equation
    a = np.dot(dp, dp)
    b = 2 * np.dot(f, dp)
    c = np.dot(f, f) - r**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return False
    else:
        # Calculate the two possible solutions for t
        discriminant_sqrt = np.sqrt(discriminant)
        t1 = (-b - discriminant_sqrt) / (2 * a)
        t2 = (-b + discriminant_sqrt) / (2 * a)

        # Check if either t1 or t2 is within the segment [0, 1]
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        else:
            return False





#--------------end intersection of line segment with 3d wall------------



def line_intersection_3d(A1, A2, A3, A4):
    # Convert points to NumPy arrays
    A1, A2, A3, A4 = np.array(A1), np.array(A2), np.array(A3), np.array(A4)

    # Direction vectors for the lines
    d1 = A2 - A1
    d2 = A4 - A3

    # The vector from A1 to A3
    r = A3 - A1

    # Cross products
    cross_d1_d2 = np.cross(d1, d2)
    cross_r_d2 = np.cross(r, d2)

    # Check if lines are parallel (cross product of direction vectors is zero)
    if np.linalg.norm(cross_d1_d2) == 0:
        # Check if lines are collinear
        if np.linalg.norm(cross_r_d2) == 0:
            # Lines are collinear - they may intersect or overlap
            return True
        else:
            # Lines are parallel but not collinear - they do not intersect
            return False

    # The scalar projection of r onto the cross product of d1 and d2
    t = np.dot(cross_r_d2, cross_d1_d2) / np.linalg.norm(cross_d1_d2)**2

    # Check if the intersection point is within the line segments
    intersection_point = A1 + t * d1
    within_segment = all(np.min([A1[i], A2[i]]) <= intersection_point[i] <= np.max([A1[i], A2[i]]) for i in range(3))

    return within_segment



def lines_intersect(A1, A2, A3, A4):
    """ Whether the lines A1-A2 and A3-A4 intersect
    :returns: TODO
    : Ai = [Ai_x, Ai_y] is an np.array
    """

    A = np.array([A1, A2, A3, A4])
    x= A[:,0]
    y= A[:,1]

    M = np.array([ [x[0] - x[1], x[3] - x[2]], [y[0] - y[1], y[3] - y[2]] ])
    b = np.array( [ x[3] - x[1], y[3] - y[1] ] )

    if (np.linalg.det(M) == 0):
        return False
    else:
        sols = np.linalg.solve(M, b)
        if ((sols[0] < 1) and (sols[0] > 0) and (sols[1] < 1) and (sols[1] > 0)):
            return True
        else:
            return False

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





def line_circle_intersection(p1, p2, cx, cy, r, *, inclusive=False, eps=1e-12):
    """
    Robust segment–circle test in 2D.
    - Counts as intersection if the minimum distance from the circle center
      to the *segment* is < r (or <= r if inclusive=True).
    - Also catches 'both endpoints inside' cases.
    """
    p1 = np.asarray(p1, dtype=float)[:2]
    p2 = np.asarray(p2, dtype=float)[:2]
    c  = np.array([cx, cy], dtype=float)

    r = float(r)
    if r <= 0.0:
        return False

    # fast endpoint-inside checks
    d1 = np.linalg.norm(p1 - c)
    d2 = np.linalg.norm(p2 - c)
    if inclusive:
        if (d1 <= r + eps) or (d2 <= r + eps):
            return True
    else:
        if (d1 <  r - eps) or (d2 <  r - eps):
            return True

    # degenerate segment?
    v = p2 - p1
    vv = float(np.dot(v, v))
    if vv <= eps*eps:
        # treat as a point (already checked endpoints)
        # if both endpoints outside, no intersection
        return False

    # project center onto the segment
    t = float(np.dot(c - p1, v) / vv)
    t = max(0.0, min(1.0, t))  # clamp to segment
    closest = p1 + t * v
    dmin = np.linalg.norm(closest - c)

    if inclusive:
        return dmin <= r + eps
    else:
        return dmin <  r - eps





















# def line_circle_intersection(p1, p2, cx, cy, r):
#     # Vector from p1 to p2
#     dp = np.array(p2) - np.array(p1)
#
#     # Vector from center of the sphere to p1
#     f = np.array(p1) - np.array([cx, cy])
#
#     # Coefficients of the quadratic equation
#     a = np.dot(dp, dp)
#     b = 2 * np.dot(f, dp)
#     c = np.dot(f, f) - r**2
#
#     # Discriminant
#     discriminant = b**2 - 4 * a * c
#
#     if discriminant < 0:
#         # No intersection
#         return False
#     else:
#         # Calculate the two possible solutions for t
#         discriminant_sqrt = np.sqrt(discriminant)
#         t1 = (-b - discriminant_sqrt) / (2 * a)
#         t2 = (-b + discriminant_sqrt) / (2 * a)
#
#         # Check if either t1 or t2 is within the segment [0, 1]
#         if (0 <= t1 <= 1) or (0 <= t2 <= 1):
#             return True
#         else:
#             return False
# def segments_intersect(p1, p2, p3, p4):
#     """
#     Return True if line segments (p1-p2) and (p3-p4) intersect.
#     Each point should be a 2D tuple/list/array like (x, y).
#     """
#     p1 = np.array(p1)
#     p2 = np.array(p2)
#     p3 = np.array(p3)
#     p4 = np.array(p4)
#
#     return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _orient(a, b, c, eps=1e-12):
    """Signed area / orientation of triangle abc.
    >0: counterclockwise, <0: clockwise, ~0: colinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(val) <= eps:
        return 0.0
    return 1.0 if val > 0.0 else -1.0

def _on_segment(a, b, c, eps=1e-12):
    """Assumes a, b, c are colinear. Check if c lies on segment ab (inclusive)."""
    ax, ay = a; bx, by = b; cx, cy = c
    return (min(ax, bx) - eps <= cx <= max(ax, bx) + eps and
            min(ay, by) - eps <= cy <= max(ay, by) + eps)

def segments_intersect(p1, p2, p3, p4, *, inclusive=True, eps=1e-12):
    """
    Robust segment intersection in 2D.
    - inclusive=True: touching or overlapping counts as intersection.
    - inclusive=False: only proper interior intersection counts.
    """
    p1 = np.asarray(p1, dtype=float)[:2]
    p2 = np.asarray(p2, dtype=float)[:2]
    p3 = np.asarray(p3, dtype=float)[:2]
    p4 = np.asarray(p4, dtype=float)[:2]

    # Degenerate segments?
    if np.allclose(p1, p2, atol=eps) or np.allclose(p3, p4, atol=eps):
        # Treat a point hitting the other segment as intersection if inclusive
        if inclusive:
            # point-on-segment checks
            if np.allclose(p1, p2, atol=eps) and (_on_segment(p3, p4, p1, eps)):
                return True
            if np.allclose(p3, p4, atol=eps) and (_on_segment(p1, p2, p3, eps)):
                return True
        return False

    # Fast AABB reject
    def _aabb(a, b):
        return (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]))
    a1, a2, a3, a4 = _aabb(p1, p2), _aabb(p3, p4), None, None
    if (a1[1] + eps < a2[0] or a2[1] + eps < a1[0] or
        a1[3] + eps < a2[2] or a2[3] + eps < a1[2]):
        return False

    o1 = _orient(p1, p2, p3, eps)
    o2 = _orient(p1, p2, p4, eps)
    o3 = _orient(p3, p4, p1, eps)
    o4 = _orient(p3, p4, p2, eps)

    # Proper intersection (strictly opposite orientations)
    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True

    if inclusive:
        # Colinear + on-segment cases (touching/overlap)
        if o1 == 0.0 and _on_segment(p1, p2, p3, eps): return True
        if o2 == 0.0 and _on_segment(p1, p2, p4, eps): return True
        if o3 == 0.0 and _on_segment(p3, p4, p1, eps): return True
        if o4 == 0.0 and _on_segment(p3, p4, p2, eps): return True

    return False





class Particle(object):
    """ A discretized particle with nodes 
    Convention: All member variables are made accessible in the first level to make it compatible with other functions, i.e. self.pos instead of self.mesh.pos
    """
    # def __init__(self, shape, meshsize, material):
    def __init__(self, mesh, shape, material, nbdarr_in_parallel=True, keep_mesh=False,shape_type='kalthoff'):
        self.shape = shape

        if keep_mesh:
            self.mesh = mesh

        self.pos = mesh.pos
        self.vol = mesh.vol


        # boundary nodes
        self.edge_nodes = mesh.bdry_nodes

        # default nonlocal boundary nodes: all
        self.nonlocal_bdry_nodes = range(len(self.pos))

        self.bdry_edges = mesh.bdry_edges

        # dimension
        total_nodes = len(self.pos)
        self.dim = len(self.pos[0])
        # print('dimension = ', self.dim)

        # default initial data
        self.disp = np.zeros((total_nodes, self.dim));
        self.vel = np.zeros((total_nodes, self.dim));
        self.acc = np.zeros((total_nodes, self.dim));
        self.extforce = np.zeros((total_nodes, self.dim));

        self.clamped_nodes = []

        self.material = material

        # torque info
        self.torque_axis = 2
        self.torque_val = 0

        # extra properties
        self.movable = 1
        self.breakable = 1
        self.stoppable = 1

        # neighborhood array generation
        # This is connectivity: a list of edges that are connected
        self.NArr = []
        nci = self.shape.nonconvex_interceptor
        self.remove_ncvx_bonds = False
        self.interceptor = []
        self.remove_ncvx_bonds_3d = False 
        
        #------------------------>
        if (nci and self.dim ==2):
            self.remove_ncvx_bonds = True
            self.interceptor = nci.all()
            # print('(Will remove nonconvex bonds)')
        if (nci and self.dim ==3):
            self.remove_ncvx_bonds_3d = True
            self.interceptor = nci.all()
        

        r = self.interceptor
        self.wallsCoordinate = r 
        #print("-----------------------this is nonconcex-intercetor----------")
        self.name = shape_type 
        nbdarr_in_parallel = 1
                
        total_nodes = len(self.pos)
        
        #--------------------------------------------

                # ... inside your function ...

        all_bonds = []  # safety

        if nbdarr_in_parallel:
            print('nbdarr in parallel')

            # Build candidates ONCE (KDTree) to avoid O(N^2)
            pos_d = self.pos[:, :self.dim]
            tree = cKDTree(pos_d)
            horizon = self.material.delta
            cand_pairs = list(tree.query_pairs(horizon))  # set -> list of (i,j)

            # Choose the per-pair filter
            if nci and self.dim == 3:
                dispatcher = {
                    'kalthoff':       self.single_bond_3d_kalthoff,
                    'plus3d':         self.single_bond_3d_plus3d,
                    'hollow_sphere':  self.single_bond_3d_sphere,
                }
                worker = dispatcher.get(self.name, self.single_bond_3d)

            elif nci and self.dim == 2:
                dispatcher = {
                    'plate_2d_circle_void': self.single_bond_2d_plate_circle_void,
                    'plate_2d_vnotch_circle_void': self.single_bond_2d_plate_vnotch_circle_void,
                    'plate_2d_notch_1':     self.single_bond_2d_plate_2d_notch_1,
                    'plate_2d_notch_2':     self.single_bond_2d_plate_2d_notch_2,
                }
                worker = dispatcher.get(self.name, self.test_single_bond)
            else:
                # no NCI → geometric neighbors already computed
                self.NArr = np.array(sorted(cand_pairs))
                return

            # Map in parallel; ensure no plotting inside `worker`
            # On Linux you can use get_context("fork") for stability
            with get_context("fork").Pool() as a_pool:
                results = a_pool.map(worker, cand_pairs)

            # Filter out Nones and store
            all_bonds = [p for p in results if p is not None]

        else:
            print("Hi (non-parallel path disabled)")
            # If you truly never want non-parallel: keep as is or mirror the same KDTree path here if needed.

        # Finalize
        self.NArr = np.array(sorted(all_bonds))  # sorted keeps (i<j) order stable


        #--------------------------------------------

        # if nbdarr_in_parallel:
        #     print('nbdarr in parallel')
        #     with Pool() as a_pool:
        #         # ---------- 3D, with nonconvex interceptor ----------
        #         if nci and self.dim == 3:
        #             if self.name == 'kalthoff':
        #                 all_bonds = a_pool.map(self.single_bond_3d_kalthoff,
        #                                        combinations(range(total_nodes), 2))
        #             elif self.name == 'plus3d':
        #                 all_bonds = a_pool.map(self.single_bond_3d_plus3d,
        #                                        combinations(range(total_nodes), 2))
        #             elif self.name == 'hollow_sphere':
        #                 all_bonds = a_pool.map(self.single_bond_3d_sphere,
        #                                        combinations(range(total_nodes), 2))
        #             else:
        #                 # default 3D handler if specific case not matched
        #                 all_bonds = a_pool.map(self.single_bond_3d,
        #                                        combinations(range(total_nodes), 2))
        #
        #         # ---------- 2D, with nonconvex interceptor ----------
        #         elif nci and self.dim == 2:
        #             if self.name == 'plate_2d_circle_void':
        #                 all_bonds = list(a_pool.map(self.single_bond_2d_plate_circle_void,
        #                                             combinations(range(total_nodes), 2)))
        #             elif self.name == 'plate_2d_notch_1':
        #                 # NOTE: method name suggests 3d; keep as provided. Consider renaming to _2d_.
        #                 all_bonds = list(a_pool.map(self.single_bond_2d_plate_2d_notch_1,
        #                                             combinations(range(total_nodes), 2)))
        #             elif self.name == 'plate_2d_notch_2':
        #                 # Build pairs via KDTree first
        #                 tree = cKDTree(self.pos[:, :2])
        #                 horizon = self.material.delta
        #                 cand_pairs = list(tree.query_pairs(horizon))
        #                 results = a_pool.map(self.single_bond_2d_plate_2d_notch_2, cand_pairs)
        #                 all_bonds = [e for e in results if e is not None]
        #             else:
        #                 all_bonds = list(a_pool.map(self.test_single_bond,
        #                                             combinations(range(total_nodes), 2)))
        #
        #         # ---------- No NCI (fast near-neighbor via KDTree), any dim ----------
        #         else:
        #             print("i am finding bonds (KDTree)")
        #             tree = cKDTree(self.pos)   # self.pos: array of node coords (N,dim)
        #             horizon = self.material.delta
        #             pairs = tree.query_pairs(horizon)   # set of (i,j) with i<j
        #             all_bonds = list(pairs)
        #
        # else:
        #     print("Hi") 
            # Non-parallel path (optional): mirror the same logic without Pool
            # if nci and self.dim == 3:
            #     if self.name == 'kalthoff':
            #         all_bonds = list(map(self.single_bond_3d_kalthoff,
            #                              combinations(range(total_nodes), 2)))
            #     elif self.name == 'plus3d':
            #         all_bonds = list(map(self.single_bond_3d_plus3d,
            #                              combinations(range(total_nodes), 2)))
            #     elif self.name == 'hollow_sphere':
            #         all_bonds = list(map(self.single_bond_3d_sphere,
            #                              combinations(range(total_nodes), 2)))
            #     else:
            #         all_bonds = list(map(self.single_bond_3d,
            #                              combinations(range(total_nodes), 2)))
            # elif nci and self.dim == 2:
            #     if self.name == 'plate_2d_circle_void':
            #         all_bonds = [self.single_bond_2d_plate_circle_void(pair)
            #                      for pair in combinations(range(total_nodes), 2)]
            #     elif self.name == 'plate_2d_notch_1':
            #         all_bonds = [self.single_bond_2d_plate_2d_notch_1(pair)
            #                      for pair in combinations(range(total_nodes), 2)]
            #     elif self.name == 'plate_2d_notch_2':
            #         all_bonds = [self.single_bond_2d_plate_2d_notch_2(pair)
            #                      for pair in combinations(range(total_nodes), 2)]
            #     else:
            #         all_bonds = [self.test_single_bond(pair)
            #                      for pair in combinations(range(total_nodes), 2)]
            # else:
            #     tree = cKDTree(self.pos)
            #     horizon = self.material.delta
            #     all_bonds = list(tree.query_pairs(horizon))
            #



        #---------------------------------------------------------------------
        #self.NArr = np.array([i for i in all_bonds if i is not None])

        # print the total number of bonds in a particle
        
        #print('_'+str(len(self.NArr)), end=' ', flush=True)
        # Print the total number of bonds in this particle
        print(f"Total bonds: {len(self.NArr)}")
    

    def test_single_bond(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        if (d <= self.material.delta):
                return [i,j]
        # else:
            # return None


    def single_bond(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds:
                intersects = False
                for k in range(len(self.interceptor)):
                    # print(k)
                    A1 = self.interceptor[k][0:2]
                    A2 = self.interceptor[k][2:4]
                    if (lines_intersect(A1, A2, p_i, p_j)):
                        intersects = True
                        break
                if not intersects:
                    return [i,j]
                # else:
                    # return None
            else:
                return [i,j]


    #--------------eliminiating bonds in 2d_plate_circle_void--------------
    def single_bond_2d_plate_circle_void(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        
        if self.remove_ncvx_bonds: 

            r = self.interceptor
                    #print("-----------------------this is nonconcex-intercetor----------")
            #print(r)
            inside_rad = r[0]
            cx = r[1] 
            cy = r[2] 
            

        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds: 

                #print("remove is on") 
                intersects = False
                for k in range(1):
                    if (line_circle_intersection(p_i, p_j, cx, cy, inside_rad,inclusive=False)):                   
                        intersects = True
                        break

                if not intersects:
                    return [i,j]
            else:
                return [i,j]

    #--------------eliminiating bonds in 2d_plate_circle_void--------------
    def single_bond_2d_plate_vnotch_circle_void(self, ij_pair):
        i, j = ij_pair
        pi = self.pos[i][:2]
        pj = self.pos[j][:2]

        # horizon check (quick reject)
        if np.linalg.norm(pj - pi) > self.material.delta:
            return None

        if not getattr(self, "remove_ncvx_bonds", False):
            return (i, j)

        #mouths = r.obt_bisec   # expected shape (4,2): [A1,B1,A2,B2]
        # where M1, M2, A1 are 2D tuples/lists
        rcvma = self.interceptor
        inside_rad = float(rcvma[0])
        cx        = float(rcvma[1])
        cy        = float(rcvma[2])
        A1        = np.asarray(rcvma[3], dtype=float)
        B1        = np.asarray(rcvma[4], dtype=float)

        # If the bond crosses either mouth segment, drop it
        if segments_intersect(pi, pj, A1, B1):
            return None
        if (line_circle_intersection(pi, pj, cx, cy, inside_rad,inclusive=False)):                   
            return None

 

        return (i, j)







    #----------------------------------------------------
    def _single_bond_2d_plate_2d_notch_2(self, ij_pair):
        i, j = ij_pair
        pi = self.pos[i][:2]
        pj = self.pos[j][:2]
        # (Optional) The KDTree already enforces horizon; you can drop this:
        # if np.linalg.norm(pj - pi) > self.material.delta:
        #     return None

        # If we’re not removing non-convex bonds, keep it
        #if not getattr(self, "remove_ncvx_bonds", False):
        #    return (i, j)

        # Expect self.interceptor.notch_edges as an array of shape (4, 2, 2):
        # [ [E1_start, E1_end], [E2_start, E2_end], [E3_start, E3_end], [E4_start, E4_end] ]
        edges = self.interceptor
        if edges is None:
            
            print("here") 
            # Fallback: keep the bond if no edges are defined
            return (i, j)
        # Remove if the bond intersects ANY notch edge
        # Use inclusive=False so tangential touches aren’t over-removed
        for e in edges:
            a, b = e[0], e[1]
            if segments_intersect(pi, pj, a, b, inclusive=False):
                print("working") 
                return None

        return (i, j)



    #----------------------------------------------------
    def single_bond_2d_plate_2d_notch_2(self, ij_pair):
        i, j = ij_pair
        pi = self.pos[i][:2]
        pj = self.pos[j][:2]

        # horizon check (quick reject)
        if np.linalg.norm(pj - pi) > self.material.delta:
            return None

        if not getattr(self, "remove_ncvx_bonds", False):
            return (i, j)

        # Fetch notch mouth segments from interceptor (A1,B1,A2,B2)
        r = self.interceptor
        #mouths = r.obt_bisec   # expected shape (4,2): [A1,B1,A2,B2]
         
        mouths = self.interceptor
        A1, B1, A2, B2 = mouths[0], mouths[1], mouths[2], mouths[3]

        # If the bond crosses either mouth segment, drop it
        if segments_intersect(pi, pj, A1, B1) or segments_intersect(pi, pj, A2, B2):
            return None

        return (i, j)

    #--------------eliminiating bonds 2 D plot 2 notches--------------
    # def _single_bond_2d_plate_2d_notch_2(self, ij_pair):
    #     # print('ij', ij_pair, end='\n', flush=True)
    #     i = ij_pair[0]
    #     j = ij_pair[1]
    #
    #     p_i = self.pos[i]
    #     p_j = self.pos[j]
    #     d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
    #     
    #     if self.remove_ncvx_bonds: 
    #
    #         r = self.interceptor
    #             
    #         w1A = r[0]
    #         w1B = r[1]  
    #
    #
    #         w2A = r[2]
    #         w2B = r[3]
    #
    #     if (d <= self.material.delta):
    #        
    #         if self.remove_ncvx_bonds: 
    #
    #             
    #             intersects = False
    #             
    #                 
    #                 if (segments_intersect(w1A,w1B,p_i,p_j)):
    #                    
    #                     intersects = True
    #                     break
    #
    #             if not intersects:
    #               
    #                 return [i,j]
    #            
    #         else:
    #             return [i,j]
    #--------------------------------------------------------------

 
    def single_bond_2d_plate_2d_notch_1(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        
        if self.remove_ncvx_bonds: 

            r = self.interceptor
                
            w1A = r[0]
            w1B = r[1]  


        if (d <= self.material.delta):
           
            if self.remove_ncvx_bonds: 

                
                intersects = False
                for k in range(1):
                
                    
                    if (segments_intersect(w1A,w1B,p_i,p_j)):
                       
                        intersects = True
                        break


                if not intersects:
                  
                    return [i,j]
               
            else:
                return [i,j]
    #--------------------------------------------------------------





      #--------------eliminiating bonds in 3d--------------
    def kalthoff_single_bond_3d(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        r = self.wallsCoordinate

                #print("-----------------------this is nonconcex-intercetor----------")
        w1A = r[0]
        w1B = r[1]  


        w2A = r[2]
        w2B = r[3]
        
        z = r[4][2]
        # material.delta is peridynamic horizon
        #print(d<=self.material.delta)
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds_3d: 

                #print("remove is on") 
                intersects = True 
                    
                    #-- check the first notches
                #print(wall_line_intrsect(w1A,w1B,p_i,p_j, z))
                print(wall_line_intrsect(w2A,w2B,p_i,p_j, z))
                if wall_line_intrsect(w1A,w1B,p_i,p_j, z):
                   intersects = False 
                   #print('detected')
                    #-- check the second notches
                    #print(line_intersection_3d(B1, B2, p_i, p_j))
                if  wall_line_intrsect(w2A,w2B,p_i,p_j, z):
                    intersects = False
                     
                    #print('detected')

                if  intersects:
                    return [i,j]
                else: 
                    return None
            else:
                return [i,j]




  #--------------eliminiating bonds in 3d plus--------------
    def single_bond_3d_plus3d(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        
        if self.remove_ncvx_bonds_3d: 

            r = self.interceptor
                    #print("-----------------------this is nonconcex-intercetor----------")
            #print(r)
            w1A = r[0]
            w1B = r[1]  
        
                  # Define the points
            pq11 = r[0] 
            pq12 = r[1] 
            pq21 = r[2] 
            pq22 = r[3] 
            pq31 = r[4] 
            pq32 = r[5] 
            pq41 = r[6] 
            pq42 = r[7] 
            
            pq10 = r[8] 
            pq20 = r[9] 
            pq30 = r[10] 
            pq40 = r[11] 

            

        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds_3d: 

                #print("remove is on") 
                intersects = False
                for k in range(1):
                    
                    if (wall_line_intersect_general(pq11,pq12,p_i,p_j, pq10)):
                        intersects = True
                        break
                    if (wall_line_intersect_general(pq21,pq22,p_i,p_j, pq20)):
                        intersects = True
                        break
                    if (wall_line_intersect_general(pq31,pq32,p_i,p_j, pq30)):
                        intersects = True
                        break
                    if (wall_line_intersect_general(pq41,pq42,p_i,p_j, pq40)):
                        intersects = True
                        break


                if not intersects:
                    return [i,j]
            else:
                return [i,j]


#--------------eliminiating bonds in 3d plus--------------
    def single_bond_3d_sphere(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        
        if self.remove_ncvx_bonds_3d: 

            r = self.interceptor
                    #print("-----------------------this is nonconcex-intercetor----------")
            #print(r)
            inside_rad = r[0]
            cx = r[1] 
            cy = r[2] 
            cz = r[3] 

        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds_3d: 

                #print("remove is on") 
                intersects = False
                for k in range(1):
                    if (line_sphere_intersection(p_i, p_j, cx, cy, cz, inside_rad)):                   
                        intersects = True
                        break

                if not intersects:
                    return [i,j]
            else:
                return [i,j]

     #--------------eliminiating bonds in 3d alpha--------------
    def single_bond_3d_kalthoff(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        
        if self.remove_ncvx_bonds_3d: 

            r = self.interceptor
                    #print("-----------------------this is nonconcex-intercetor----------")
            #print(r)
            w1A = r[0]
            w1B = r[1]  


            w2A = r[2]
            w2B = r[3]
            
            z = r[4][2]

        #print(d<=self.material.delta)
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds_3d: 

                #print("remove is on") 
                intersects = False
                for k in range(1):
                    #print(k)
                    #print("this is A1")
                    #print(A1)
                    #plot_single_bonds(A1,A2,p_i,p_j) 
                    #print('line_intersection_3d(A1, A2, p_i, p_j)-------->``')
                    #print(line_intersection_3d(A1, A2, p_i, p_j))
                    
                    if (wall_line_intrsect(w1A,w1B,p_i,p_j, z)):
                        #print('detected w1')
                    #if (line_intersection_3d(A1, A2, p_i, p_j)):
                        #print(wall_line_intrsect(w1A,w1B,p_i,p_j, z))
                        intersects = True
                        break
                    #print('line_intersection_3d(B1, B2, p_i, p_j)-------->``')
                    #print(line_intersection_3d(B1, B2, p_i, p_j))
                    

                    if  (wall_line_intrsect(w2A,w2B,p_i,p_j, z)):
                    #if (line_intersection_3d(B1, B2, p_i, p_j)):
                        #print('detected w2')
                        #print("shit")
                        #print(wall_line_intrsect(w2A,w2B,p_i,p_j, z))
                        intersects = True
                        break

                if not intersects:
                    #print("flaseeeee")
                    return [i,j]
                # else:
                    # return None
            else:
                return [i,j]
        # else:
            # return None




     #--------------eliminiating bonds in 3d--------------
    def single_bond_3d(self, ij_pair):
        # print('ij', ij_pair, end='\n', flush=True)
        i = ij_pair[0]
        j = ij_pair[1]

        p_i = self.pos[i]
        p_j = self.pos[j]
        d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
        r = self.interceptor
                #print("-----------------------this is nonconcex-intercetor----------")
        Al1=self.Al1
        Al2=self.Al2
        Ar1=self.Ar1
        Ar2=self.Ar2

        # material.delta is peridynamic horizon
        tnodes = min(len(Al1),len(Al2),len(Ar1),len(Ar2))
        #print(d)
        #print("this is delta")
        #print(self.material.delta)
        #print(d<=self.material.delta)
        if (d <= self.material.delta):
            ##### remove nonconvex bonds
            if self.remove_ncvx_bonds_3d: 

                #print("remove is on") 
                intersects = False
                for k in range(tnodes):
                    #print(k)
                    A1 = Al1[k][0:3]
                    A2 = Al2[k][0:3]
                    #print("this is A1")
                    #print(A1)
                    #plot_single_bonds(A1,A2,p_i,p_j) 
                    #print('line_intersection_3d(A1, A2, p_i, p_j)-------->``')
                    #print(line_intersection_3d(A1, A2, p_i, p_j))
                    if (line_intersection_3d(A1, A2, p_i, p_j)):
                        intersects = True
                        break
                    #print("this is problematic K")
                    #print(k)
                    B1 = Ar1[k][0:3]
                    B2 = Ar2[k][0:3]
                    #plot_single_bonds(B1,B2,p_i,p_j) 
                    #print('line_intersection_3d(B1, B2, p_i, p_j)-------->``')
                    #print(line_intersection_3d(B1, B2, p_i, p_j))
                    if (line_intersection_3d(B1, B2, p_i, p_j)):
                        intersects = True
                        break

                if not intersects:
                    return [i,j]
                # else:
                    # return None
            else:
                return [i,j]
        # else:
            # return None

        # one parameter function to be used by pool
        # def oneparam_f_bond(ij):
        # return self.single_bond(self,ij)
        ## Using an external function to call for parallelization, otherwise, I get error 
        # return single_bond_ext(ij, self.pos, self.material.delta, remove_ncvx_bonds, interceptor)
        # print('Will generate nbdarr')
                
    def gen_nonlocal_boundry_nodes(self, contact_radius, scaled = 1):
        """ Generate nonlocal boundary nodes
        :contact_radius: 
        """
        # print('Computing nonlocal boundary nodes: ', end = ' ', flush = True)
        if contact_radius is not None:
            temp = []
            for p in range(len(self.edge_nodes)):
                e_node = self.edge_nodes[p]
                pos_node = self.pos[e_node]
                for q in range(len(self.pos)):
                    pos_other = self.pos[q]
                    d = np.sqrt(np.sum(np.square(pos_node - pos_other))) #norm
                    # if (d <= self.contact.contact_radius):
                    if (d <= contact_radius*scaled):
                        temp.append(q)
            self.nonlocal_bdry_nodes = list(set(temp))
        # print(len(self.nonlocal_bdry_nodes))


    def rotate(self, angle):
        """ Rotate by angle in the clockwise direction
        Only for 2D particles
        """
        if self.dim!=2:
            print('Wrong dimension for rotation. Need 2. Use rotate3d for 3D particle.')
        Rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
        self.pos = self.pos @ Rot   # @ is matrix multiplication, so is a.dot(b) 

    def rotate3d(self, axis, angle):
        """ Rotate by angle in the counterclockwise direction about an axis
        Only for 3d particles
        """
        if axis=='x':
            Rot = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)] ])
        if axis=='y':
            Rot = np.array([[np.cos(angle),0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)] ])
        if axis=='z':
            Rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1] ])

        self.pos = self.pos @ Rot   # @ is matrix multiplication, so is a.dot(b) 

    def trasnform_bymat(self, matrix, quiet=False):
        """ Transform using a matrix
        """
        # Rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
        self.pos = self.pos @ matrix   # @ is matrix multiplication, so is a.dot(b) 
        if not quiet:
            print('Caution: did not transform the volume to maintain the density.')

    def shift(self, shift):
        a = self.pos
        b = shift
        #print("this is pos",a)
        #print('this is shift',shift)
        self.pos += shift

    def scale(self, scale):
        self.pos *= scale
        # update the volume
        self.vol *= (scale**self.dim)

        # delta and cnot are modified since the neighbors are not being recomputed

class GridDist(object):
    """ Particle location distribution on a grid
    """
    def __init__(self, left, right, top, bottom, nx, ny):
        self.left    = left   
        self.right   = right  
        self.top     = top    
        self.bottom  = bottom 

        self.nx = nx
        self.ny = ny

    def maximalrad(self):
        lx = (self.right - self.left)/self.nx/2
        ly = (self.top - self.bottom)/self.ny/2
        print(lx,ly)
        return min(lx, ly)

    def gridloc(self):
        """TODO: Docstring for gridloc.
        :arg1: TODO
        :returns: TODO
        """
        lx = (self.right - self.left)/self.nx
        ly = (self.top - self.bottom)/self.ny
        xx = np.linspace(self.left + lx/2, self.right-lx/2, num=self.nx, endpoint=True)
        yy = np.linspace(self.bottom + ly/2, self.top-ly/2, num=self.ny, endpoint=True)
        xv, yv = np.meshgrid(xx, yy)
        loc_list = np.array([xv.flatten(), yv.flatten()]).transpose()
        return np.array(loc_list)


class ShapeList(object):
    """Contains a list of instances of Shape class, and other info """
    def __init__(self):
        self.shape_list = []
        self.count_list = []
        self.meshsize_list = []
        self.material_list = []
        self.shape_type_list = []
        # self.msh_file_list = []

    def append(self, shape, count, meshsize, material, plot_shape = True,shape_type=None):
    # def append(self, shape, count, meshsize, material, msh_file= None):
        """TODO: Docstring for append.

        :shape: TODO
        :count: TODO
        :meshsize: TODO
        :material: TODO
        :returns: TODO

        """
        self.shape_list.append(shape)
        self.count_list.append(count)
        self.meshsize_list.append(meshsize)
        self.material_list.append(material)
        self.shape_type_list.append(shape_type) 


    def generate_mesh(self, dimension = 2, contact_radius = None, plot_mesh = True, plot_node_text=False, plot_shape = True, shapes_in_parallel=False, print_nnodes=True, keep_mesh=False):
        """Returns a rank-2 array of particle meshes
        : shapes_in_parallel: if set to true, it computes the particle properties in parallel for each shape. Otherwise, the NbdArr and boundary node computation happens in parallel for each node. 
        Outermost parallelization is most preferable. So, for multiple shapes/particles with small number of nodes requires this option to be true. On the other hands, for single particle with many nodes, setting this to false works best.
        :returns: particles[sh][i] for shape sh count i
        """

        # if the shapes are computed in parallel, do not compute NbdArr etc in parallel
        # outermost parallelism
        nbdarr_in_parallel  = not shapes_in_parallel

        print('Shape:', end=' ', flush=True)

        # Helper function
        def gen_particle(sh):
            # print(sh, end=' ', flush=True)
            shape = self.shape_list[sh]
            if shape.msh_file is None:
                mesh = genmesh(P_bdry=shape.P, pygmsh_geom=shape.pygmsh_geom, meshsize=self.meshsize_list[sh], dimension = dimension)

                # print(shape.P)
                if (plot_shape and (shape.P is not None)):
                    shape.plot(bdry_arrow=True, extended_bdry=True, angle_bisector=True)
            else:
                # when the .msh is specified
                # mesh = genmesh(P_bdry=None, meshsize=None, msh_file=shape.msh_file, dimension = dimension, do_plot = plot_mesh)
                mesh = genmesh(P_bdry=None, meshsize=None, msh_file=shape.msh_file, dimension = dimension)
                if plot_mesh:
                    mesh.plot(dotsize=10, plot_node_text=plot_node_text, highlight_bdry_nodes=True)

            # scale mesh here, if specified. This is useful for msh files
            if shape.scale_mesh_to is not None:
                # scale about the centroid
                P = mesh.pos
                cent = np.mean(P, axis=0)
                print('centroid', cent)
                rad = np.sqrt(np.max(np.sum((P - cent)**2, axis=1)))
                print('rad', rad)
                P_new = P - cent
                P_new *= shape.scale_mesh_to/rad

                # put centroid back
                if not shape.centroid_origin:
                    P_new += cent

                mesh.pos = P_new
                # scale volume
                mesh.vol *=((shape.scale_mesh_to/rad) ** dimension)




            # print('total mesh volume: ', np.sum(mesh.vol))

            PP = Particle(mesh=mesh, shape=self.shape_list[sh], material=self.material_list[sh], nbdarr_in_parallel=nbdarr_in_parallel, keep_mesh=keep_mesh,shape_type=self.shape_type_list[sh])
            nnodes = len(PP.pos)
            #print("shape  :",str(sh)+"number of nodes:",'('+str(nnodes)+')', end=' ', flush=True)
            
            print("------------particle profile------------")
            print(f"shape: {sh}  number of nodes: ({nnodes})", end=" ", flush=True)

            # print('Done generating particles')

            # generate nonlocal boundary nodes
            PP.gen_nonlocal_boundry_nodes(contact_radius)


            particles_sh = []
            for i in range(self.count_list[sh]):
                # shallow copy (=) refers to the same object, instead of creating a new one
                this_PP = copy.deepcopy(PP)
                particles_sh.append(this_PP)

            # particles.append(particles_sh)
            return particles_sh


        start_sh = time.time()
        if shapes_in_parallel:
            print('Shapes in parallel')
            # parallel attempt
            shape_pool = Pool()
            particles = shape_pool.map(gen_particle, range(len(self.shape_list))) 
        else:
            print('Shapes in serial')
            # serial version
            particles = []
            for sh in range(len(self.shape_list)):
                particles.append(gen_particle(sh))

        print('\n')
        print('time taken to generate all shapes', time.time() - start_sh)
        
        return particles

        
# def generate_particles(shapes, shape_count, meshsize, material_list):
    # """ Generate Particle related to each shape
    # :shapes: a list of instances of the shape class generated from shape_dict.py
    # :shape_count: how many members of each shape to create
    # :returns: 2d array of particles, particles[shape][count]
    # """
    # particles = []
    # for sh in range(len(shapes)):
        # print('Shape', sh)
        # PP = Particle(shapes[sh], meshsize, material_list[sh])

        # particles_sh = []
        # for i in range(shape_count[sh]):
            # # shallow copy (=) refers to the same object, instead of creating a new one
            # this_PP = copy.deepcopy(PP)
            # particles_sh.append(this_PP)

        # particles.append(particles_sh)
    
    # return particles


class Wall(object):
    """ Geometric wall """
    def __init__(self, allow = 0, left = None, right = None, top = None, bottom = None ):
        self.allow  = allow
        self.left        = left
        self.right       = right
        self.top         = top 
        self.bottom      = bottom

        self.reaction = None

    def get_lrtp(self):
        """ Returns the wall dimensions in an array
        :returns: TODO

        """
        return np.array([self.left, self.right, self.top, self.bottom])

    def set_size(self, ss):
        """ Reads the wall size from an array
        """
        if (len(ss) !=4):
            print('Error: input array dimension is not 4.')
        self.left        = ss[0]
        self.right       = ss[1]
        self.top         = ss[2]
        self.bottom      = ss[3]

    def get_lines(self):
        """ List of lines that are wall boundaries
        To feed to the collection
        """
        a = [self.left, self.bottom]
        b = [self.right, self.bottom]
        c = [self.right, self.top]
        d = [self.left, self.top]

        return [ [a, b], [b,c], [c,d], [d,a] ]

    def print(self):
        print('allow =  ', self.allow)
        print('left: ', self.left)
        print('right: ', self.right)
        print('top: ', self.top)
        print('bottom: ', self.bottom)

    def get_h(self):
        """get the horizontal length of the wall
        """
        return self.right - self.left
    def get_v(self):
        """get the vertical height of the wall
        """
        return self.top - self.bottom
    def wall_vol(self):
        """returns the wall volume
        """
        return self.get_v() * self.get_h()

class Wall3d(object):
    """ Geometric wall """
    def __init__(self, allow = 0, x_min = None, y_min = None, z_min = None, x_max = None, y_max = None, z_max = None):
        self.allow  = allow
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def get_lrtp(self):
        """ Returns the wall dimensions in an array
        :returns: TODO

        """
        return np.array([self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max])

    def set_size(self, ss):
        """ Reads the wall size from an array
        """
        if (len(ss) !=6):
            print('Error: input array dimension is not 4.')

        self.x_min = ss[0] 
        self.y_min = ss[1]
        self.z_min = ss[2]
        self.x_max = ss[3]
        self.y_max = ss[4]
        self.z_max = ss[5]





    def get_faces(self):

        # list of vertices
        Z = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
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

        return faces
    
    def get_lines(self):
        # list of vertices
        Z = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
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

        lines = np.array([
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

        return lines

    def print(self):
        print('allow =  ', self.allow)
        print('x_min: ', self.x_min)
        print('y_min: ', self.y_min)
        print('z_min: ', self.z_min)
        print('x_max: ', self.x_max)
        print('y_max: ', self.y_max)
        print('z_max: ', self.z_max)

class Contact(object):
    """Pairwise contact properties"""
    def __init__(self, contact_radius, normal_stiffness, damping_ratio, friction_coefficient):
        self.contact_radius         = contact_radius
        self.normal_stiffness        = normal_stiffness
        self.damping_ratio           = damping_ratio
        self.friction_coefficient    = friction_coefficient 

    def print(self):
        """print info

        :f: TODO
        :returns: TODO

        """
        print('contact_radius: ', self.contact_radius)
        print('normal_stiffness: ', self.normal_stiffness)
        print('damping_ratio: ', self.damping_ratio)
        print('friction_coefficient: ', self.friction_coefficient)

#---------------------Plot Bonds---------------------
def plot_single_bonds(A1,A2,B1,B2):
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting lines
    #?????????????????????????????????/in qalat nist???????????????? start and end
    ax.plot([A1[0], A2[0]], [A1[1], A2[1]], [A1[2], A2[2]])
    ax.plot([B1[0], B2[0]], [B1[1], B2[1]], [B1[2], B2[2]])
    # Setting labels
    plt.title('bonds after removing nonconvex part')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show plot
    plt.show()


def plot_bonds(particles):
    """
    Plot bonds for 2D or 3D particles.
    Assumes each particle has:
      - pos: array of shape (N, dim) with coordinates
      - NArr: list/array of (i,j) bond indices
    """
    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            part = particles[sh][count]
            total_convex_bond = [(part.pos[a], part.pos[b]) for a, b in part.NArr]
            print("this the total bonds")
            print(len(total_convex_bond)) 
            dim = part.pos.shape[1]  # detect dimension (2 or 3)
            
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
                for start, end in total_convex_bond:
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]], 'b-')
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
            elif dim == 2:
                ax = fig.add_subplot(111)
                for start, end in total_convex_bond:
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]], 'b-')
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_aspect('equal', adjustable='box')  # keep proportions

            plt.title('Bonds after removing nonconvex part')
            plt.savefig('bonds.png', dpi=300, bbox_inches='tight')
            plt.show()

def _plot_bonds(particles):
    

    totalbonds = []
    total_convex_bond = []
    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            
            part = particles[sh][count]
            #nnodes_vec.append(len(part.pos))
            total_nodes = len(part.pos)
            #total_convex_bond =[(part.pos[a],part.pos[b]) for a,b in list(combinations(range(total_nodes), 2))] 
            total_convex_bond =[(part.pos[a],part.pos[b]) for a,b in part.NArr] 
            # Creating a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plotting lines
            #?????????????????????????????????/in qalat nist???????????????? start and end
            for start, end in total_convex_bond:  
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
            # Setting labels
            plt.title('bonds after removing nonconvex part')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            # save the image
            plt.savefig('bonds.png', dpi=300, bbox_inches='tight')
 
            # Show plot
            plt.show()
def plot_list_3d(A):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("no bond outside!!!")
    x2,y2,z2= zip(*A) 

    #Plotting the points from list1 in red and list2 in blue

    ax.scatter(x2, y2, z2, color='blue', label='concexNodes')
   
    plt.title('plot nodes')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # save the image

    # Show plot
    plt.show() 

def plot_bonds_nonConvex(particles):
    

    totalbonds = []
    total_convex_bond = []
    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            
            part = particles[sh][count]
 
            #------------------------
            total_convex_node_Idx  = list({element for sublist in part.NArr for element in sublist})
             
            total_nodes = part.pos
            convx_nodes =[part.pos[a] for a in total_convex_node_Idx] 
            removed_nodes =  np.array([point for point in total_nodes if not any(np.array_equal(point, p) for p in convx_nodes)])
 
            total_Nonconvex_bond =[(part.pos[a],part.pos[b]) for a,b in removed_nodes] 
            #----------------------
            # Creating a 3D plot

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plotting lines
            #?????????????????????????????????/in qalat nist???????????????? start and end
            for start, end in total_Nonconvex_bond:  
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
            # Setting labels
            plt.title('bonds after removing nonconvex part')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            # save the image
            plt.savefig('bondsNon.png', dpi=300, bbox_inches='tight')
 
            # Show plot
            plt.show()


#--------------------End Plot Bonds-------------------
def plot_nodes(particles):
    

    totalbonds = []
    total_convex_bond = []
    #for sh in range(len(particles)):
    
    for sh in [0]: 
        shape = particles[sh]
        for count in range(len(shape)):
            
            part = particles[sh][count]
            #nnodes_vec.append(len(part.pos))
            #x,y,z = zip(*part.pos) 
            total_convex_node_Idx  = list({element for sublist in part.NArr for element in sublist})
             
            total_nodes = part.pos
            convx_nodes =[part.pos[a] for a in total_convex_node_Idx] 
            #print(part.pos)
            #print(par.NArr)
            #removed_nodes = [point for point in total_nodes if point not in convx_nodes]   
            removed_nodes =  np.array([point for point in total_nodes if not any(np.array_equal(point, p) for p in convx_nodes)])
            #print(removed_nodes)
            #print(total_convex_node_Idx)
            #print(total_nodes)
                        # Creating a 3D plot
            print("\n number of removed\n"+ str(len(removed_nodes)))
            print("\n total nodes are \n"+ str(len(total_nodes)))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            print("\n size removed nodes are \n ")
            print(len(removed_nodes))
            if len(removed_nodes)<1:
                print("\n no bond outside!!!\n")
                x2,y2,z2= zip(*convx_nodes) 
            else:
                print("\n ha we removed some bonds\n")
                x2,y2,z2= zip(*convx_nodes) 
                x1,y1,z1= zip(*removed_nodes) 
                ax.scatter(x1, y1, z1, color='green', label='totalNods',s=2)

            #Plotting the points from list1 in red and list2 in blue

            ax.scatter(x2, y2, z2, color='red', label='concexNodes',s=2)
           
            plt.title('plot nodes')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            # save the image
            plt.savefig('nodes.png', dpi=300, bbox_inches='tight')
 
            # Show plot
            plt.show() 
def plot_node_bond_distribution(particles, nnodes_filename='nnodes_distribution.png', nbonds_filename='nbonds_distribution.png', bins=None):
    nnodes_vec = []
    nbonds_vec = []
    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            part = particles[sh][count]
            nnodes_vec.append(len(part.pos))
            nbonds_vec.append(len(part.NArr))
    plt.hist(nnodes_vec, bins=bins)
    plt.title('Number of nodes: distribution')
    plt.xlabel('nnodes')
    plt.ylabel('frequency')
    # save the image
    plt.savefig(nnodes_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.hist(nbonds_vec, bins=bins)
    plt.title('Number of bonds: distribution')
    plt.xlabel('nbonds')
    plt.ylabel('frequency')
    # save the image
    plt.savefig(nbonds_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_setup(particles, dotsize=0.5, contact_radius=None, delta=None, linewidth=1, wall=None, show_plot=True, adaptive_dotsize=False, show_particle_index=False, save_filename='setup.png'):
    """plots the particle setup
    :particles: array of particles
    """

    dim = len(particles[0][0].pos[0])

    min_vol = 1
    if adaptive_dotsize:
        slot_shape = np.zeros(len(particles))
        for sh in range(len(particles)):
            shape = particles[sh]
            slot_count = np.zeros(len(shape))
            for count in range(len(shape)):
                part = particles[sh][count]
                slot_count[count] = np.min(part.vol)
            slot_shape[sh] = np.min(slot_count)
        min_vol = np.min(slot_shape)

    # print('min vol', min_vol)

    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            part = particles[sh][count]
            P = part.pos
            if adaptive_dotsize:
                dval = np.sqrt(part.vol/min_vol) * float(dotsize)
            else:
                dval = dotsize

            plt.scatter(P[:,0], P[:,1], s=dval, linewidths=0)

            # show clamped node
            cc = part.clamped_nodes
            for j in range(len(cc)):
                c = cc[j]
                plt.scatter(P[c,0], P[c,1], c='r', s=dval, linewidths=0)

            if show_particle_index:
                # # text
                plt.annotate(str(count), np.mean(P, axis = 0))

    if contact_radius:
        part = particles[0][0]
        node = 0
        x0 = part.pos[node][0]
        y0 = part.pos[node][1]
        tt = np.linspace(0, 2*np.pi, endpoint=True)
        xx = contact_radius * np.cos(tt)
        yy = contact_radius * np.sin(tt)
        plt.plot( xx + x0, yy + y0, linewidth=linewidth)

    if delta:
        part = particles[0][0]
        node = 0
        x0 = part.pos[node][0]
        y0 = part.pos[node][1]
        tt = np.linspace(0, 2*np.pi, endpoint=True)
        xx = delta * np.cos(tt)
        yy = delta * np.sin(tt)
        plt.plot( xx + x0, yy + y0, 'r', linewidth=linewidth)

    if wall:
        wi = wall.get_lrtp()
        a = [wi[0],  wi[3]]
        b = [wi[1], wi[3]]
        c = [wi[1], wi[2]]
        d = [wi[0],  wi[2]]
        ls = [ [a, b], [b,c], [c,d], [d,a] ]
        lc = LineCollection(ls, linewidths=1, colors='b')
        plt.gca().add_collection(lc)

    plt.axis('scaled')
                
    plt.grid()
    print('Saving experiment setup png to', save_filename)
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

def plot3d_setup(particles, dotsize=0.5, contact_radius=None, delta=None, wall=None, show_plot=True, adaptive_dotsize=False, show_particle_index=False, save_filename='setup.png', trisurf=False, trisurf_transparent=False, trisurf_linewidth=0.1, trisurf_alpha=0.6, noscatter=False):
    """plots the particle setup
    :particles: array of particles
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_vol = 1
    if adaptive_dotsize:
        slot_shape = np.zeros(len(particles))
        for sh in range(len(particles)):
            shape = particles[sh]
            slot_count = np.zeros(len(shape))
            for count in range(len(shape)):
                part = particles[sh][count]
                slot_count[count] = np.min(part.vol)
            slot_shape[sh] = np.min(slot_count)
        min_vol = np.min(slot_shape)

    for sh in range(len(particles)):
        shape = particles[sh]
        for count in range(len(shape)):
            part = particles[sh][count]
            P = part.pos
            if adaptive_dotsize:
                dval = np.sqrt(part.vol/min_vol) * float(dotsize)
            else:
                dval = dotsize

            if not noscatter: 
                ax.scatter(P[:,0], P[:,1], P[:,2], s=dval, marker = '.', linewidth=0, cmap='viridis')

            # show clamped node
            cc = part.clamped_nodes
            for j in range(len(cc)):
                c = cc[j]
                if not noscatter: 
                    ax.scatter(P[c,0], P[c,1], P[c,2], c='r', s=dval, linewidths=0)

            if show_particle_index:
                meanP = np.mean(P, axis=0)
                ax.text(meanP[0], meanP[1], meanP[2], str(count))

            if trisurf:
                # color = None

                if trisurf_transparent:
                    color = (0,0,0,0)
                    ax.plot_trisurf(P[:,0], P[:,1], P[:,2], triangles=particles[sh][count].bdry_edges, color=color, linewidth=trisurf_linewidth, antialiased=True, edgecolor='k') 
                else:
                    ax.plot_trisurf(P[:,0], P[:,1], P[:,2], triangles=particles[sh][count].bdry_edges, alpha=trisurf_alpha, linewidth=trisurf_linewidth, antialiased=True, edgecolor='k') 

    if contact_radius:
        part = particles[0][0]
        node = 0
        x0 = part.pos[node][0]
        y0 = part.pos[node][1]
        z0 = part.pos[node][2]
        tt = np.linspace(0, 2*np.pi, endpoint=True)
        xx = contact_radius * np.cos(tt)
        yy = contact_radius * np.sin(tt)
        plt.plot( xx + x0, yy + y0, z0)

    if delta:
        part = particles[0][0]
        node = 0
        x0 = part.pos[node][0]
        y0 = part.pos[node][1]
        z0 = part.pos[node][2]
        tt = np.linspace(0, 2*np.pi, endpoint=True)
        xx = delta * np.cos(tt)
        yy = delta * np.sin(tt)
        plt.plot( xx + x0, yy + y0, z0, 'r')

    if wall:
        wall_alpha = 0.8
        wall_linewidth = 1
        ls = wall.get_lines()
        lc = Line3DCollection(ls, linewidths=wall_linewidth, colors='k', alpha=wall_alpha)
        ax.add_collection(lc)

    # mx = np.amax(np.abs(P))
    mx = np.amax(np.abs(wall.get_lrtp()))
    XYZlim = [-mx, mx]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_box_aspect((1, 1, 1))
                
    plt.grid()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()



class Experiment(object):

    """Experiment setup"""

    def __init__(self, particles, wall, contact):
        """Collection of all

        :particles: TODO
        :wall: TODO
        :contact: TODO

        """
        self.particles = particles
        self.wall = wall
        self.contact = contact

        
        ## compute the boundary nodes, for each particle count
        # print('Computing boundary nodes: ')
        # for sh in range(len(self.particles)):
            # count = len(self.particles[sh])
            # for i in range(count):
                # particle = self.particles[sh][i]
                # temp = []
                # for p in range(len(particle.edge_nodes)):
                    # e_node = particle.edge_nodes[p]
                    # pos_node = particle.pos[e_node]
                    # for q in range(len(particle.pos)):
                        # pos_other = particle.pos[q]
                        # d = np.sqrt(np.sum(np.square(pos_node - pos_other))) #norm
                        # # if (d <= self.contact.contact_radius):
                        # if (d <= self.contact.contact_radius*1.1):
                            # temp.append(q)
                # particle.nonlocal_bdry_nodes = list(set(temp))
                ## particle.nonlocal_bdry_nodes = list(set(range(len(particle.pos))))

        
                # print('edge nodes: ', len(particle.edge_nodes))
                # print('bdry nodes: ', len(particle.nonlocal_bdry_nodes))
                # print(i, [len(particle.edge_nodes), len(particle.nonlocal_bdry_nodes)], 
                # print(i, end = ' ', flush=True)

        # print('')



    def save(self, filename):
        """Save experiment setup to file

        :filename: filename with path
        :returns: TODO

        """
        print('Saving to', filename,  'Univ particle: ')
        with h5py.File(filename, "w") as f:
            # universal index starts at 1 to be compatible with matlab
            # j = 1
            # universal index starts at 0 to be compatible with C++
            j = 0

            for sh in range(len(self.particles)):
                count = len(self.particles[sh])
                for i in range(count):
                    particle = self.particles[sh][i]
                    p_ind = ('P_%05d' % j)
                    f.create_dataset(p_ind + '/Pos', data=particle.pos)
                    f.create_dataset(p_ind + '/Vol', data=particle.vol)
                    f.create_dataset(p_ind + '/Connectivity', data=particle.NArr)
                    # needs to be a column matrix
                    f.create_dataset(p_ind + '/bdry_nodes', data=np.array([particle.nonlocal_bdry_nodes]).transpose())
                    f.create_dataset(p_ind + '/bdry_edges', data=np.array(particle.bdry_edges))

                    f.create_dataset(p_ind + '/clamped_nodes', data=np.array(particle.clamped_nodes))

                    # Initial conditions
                    f.create_dataset(p_ind + '/disp', data=particle.disp)
                    f.create_dataset(p_ind + '/vel', data=particle.vel)
                    f.create_dataset(p_ind + '/acc', data=particle.acc)
                    f.create_dataset(p_ind + '/extforce', data=particle.extforce)

                    # Material properties
                    f.create_dataset(p_ind + '/delta', data=[[particle.material.delta]])
                    f.create_dataset(p_ind + '/rho', data= [[particle.material.rho]])
                    f.create_dataset(p_ind + '/cnot', data= [[particle.material.cnot]])
                    f.create_dataset(p_ind + '/snot', data= [[particle.material.snot]])

                    # torque info
                    f.create_dataset(p_ind + '/torque_axis', data= [[particle.torque_axis]])
                    f.create_dataset(p_ind + '/torque_val', data= [[particle.torque_val]])

                    # extra properties
                    f.create_dataset(p_ind + '/movable', data= [[particle.movable]])
                    f.create_dataset(p_ind + '/breakable', data= [[particle.breakable]])
                    f.create_dataset(p_ind + '/stoppable', data= [[particle.stoppable]])

                    print(j, end = ' ', flush=True)
                    j = j+1

            print('\n')

            # making all scalars a rank-2 data (matrix) to make it compatible with matlab
            # f.create_dataset('total_particles_univ', data=[[j-1]])
            f.create_dataset('total_particles_univ', data=[[j]])

            # contact info
            f.create_dataset('pairwise/contact_radius', data=[[self.contact.contact_radius]])
            f.create_dataset('pairwise/normal_stiffness', data=[[self.contact.normal_stiffness]])
            f.create_dataset('pairwise/damping_ratio', data=[[self.contact.damping_ratio]])
            f.create_dataset('pairwise/friction_coefficient', data=[[self.contact.friction_coefficient]])

            # wall info
            f.create_dataset('wall/allow_wall', data = [[self.wall.allow]])
            f.create_dataset('wall/geom_wall_info', data = np.array([self.wall.get_lrtp()]).transpose() )

#######################################################################

def icetest1():
    """ Particle to collapse under its own weight
    """

    delta = 1e-3
    meshsize = 1e-3/3
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.small_disk(), count=1, meshsize=meshsize, material=material_dict.peridem(delta))

    bulk_l = 7e-3
    # clamp_l = 2e-3
    clamp_l = 4e-3

    bulk_s = 3e-3
    clamp_s = 0.5e-3/2

    SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.01, w_depth_ratio=1), count=1, meshsize=meshsize, material=peri_deformable(delta, Gnot_scale=0.5))

    SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=2, meshsize=meshsize, material=peri_deformable(delta))
    # SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=3, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)
    # particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    particles[1][0].shift([-(bulk_l - clamp_l), +(bulk_s+clamp_s)+contact_radius/2 ])
    particles[1][1].shift([-(bulk_l - clamp_l), -(bulk_s+clamp_s)-contact_radius/2 ])
    
    # one that pushes
    # particles[1][2].shift([-(bulk_l + clamp_l), 0 ])


    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # force in the x-direction to simulate flow
    # particles[0][0].acc += [5e-4, 0]
    # particles[0][0].extforce += [5e4 * particles[0][0].material.rho, 0]


    # particles[0][0].acc += [0, -5e4]
    # particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # plank does not move
    particles[1][0].movable = 0
    particles[1][1].movable = 0

    # one that pushes
    # particles[1][2].acc += [5e4, 0]
    # particles[1][2].extforce += [5e4 * particles[0][0].material.rho, 0]


    # wall info
    wall_left   = -25e-3
    wall_right  = 25e-3
    wall_top    = 15e-3
    wall_bottom = -15e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def icetest2():
    """ Particle to collapse under its own weight
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    bulk_l = 7e-3
    clamp_l = 6e-3
    bulk_s = 5e-3
    clamp_s = 0.5e-3/2

    SL = ShapeList()

    SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=True)

    # Initial data
    # particles[0][0].vel += [0, -2]
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    part = particles[0][0]
    # force in the x-direction to simulate flow
    for i in range(len(part.pos)):
        right_edge = bulk_l
        if np.abs(part.pos[i][0] - right_edge) < delta :
            # very high gravity
            acc_val = 5e4
            # height-dependent force, plus if above, minus if below
            # acc_val = part.pos[i][1]/bulk_s * 5e4

            part.acc[i] += [acc_val, 0]
            part.extforce[i] += [acc_val * part.material.rho, 0]

        # clamped nodes
        left_edge = -bulk_l
        if np.abs(part.pos[i][0] - left_edge) < delta :
            part.clamped_nodes.append(i)

    # wall info
    wall_left   = -25e-3
    wall_right  = 25e-3
    wall_top    = 15e-3
    wall_bottom = -15e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)


def shapetest():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3
    contact_radius = 1e-3;

    # wall info
    wall_left   = -30e-3
    wall_right  = 30e-3
    wall_top    = 30e-3
    wall_bottom = -30e-3

    # 15 indices
    indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    ## scaling factor for shapes to make them about the same size
    scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 5
    ny = 5

    poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()


    SL = ShapeList()

    ## single shape, repeated
    # SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=0, scaling=1e-5, downsample=1), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.load_shape_index(ind=indlist[0], scaling=1e-5, downsample=2), count=len(poslist), meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # acc_val = 1e4
    # vel_val = 10
    vel_val = 7

    seed(1)
    for i in range(len(poslist)):
        ## single shape
        part = particles[0][i]
        ## multiple shapes
        # part = particles[i][0]

        part.rotate(0 + (random() * (2*np.pi - 0)) )
        part.shift(poslist[i])

        # centroid = np.mean(part.pos, axis=0)
        # # print(centroid)
        # centroid_dir = centroid/np.sqrt(np.sum(centroid**2, axis=0))
        # print(centroid_dir)

        loc_norm = np.sqrt(np.sum(poslist[i]**2, axis=0))
        if loc_norm == 0:
            centroid_dir = 0
        else:
            centroid_dir = poslist[i]/loc_norm

        # part.acc += [acc_val, 0]
        # part.extforce += [acc_val * part.material.rho, 0]

        # part.acc += -acc_val*centroid_dir
        # part.extforce += (-acc_val*centroid_dir*part.material.rho)

        part.vel += -vel_val*centroid_dir


    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles)

    return Experiment(particles, wall, contact)

def boundarytest():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3*2
    contact_radius = 1e-3;

    # wall info
    wall_left   = -20e-3
    wall_right  = 45e-3
    wall_top    = 50e-3
    wall_bottom = -30e-3

    # 15 indices
    indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    ## scaling factor for shapes to make them about the same size
    scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 3
    ny = 2

    region_left = 5e-3
    region_right = 30e-3
    region_top = 40e-3
    region_bottom = 20e-3

    # poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    poslist = GridDist(region_left, region_right, region_top, region_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()


    SL = ShapeList()

    ## single shape, repeated
    # SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=0, scaling=1e-5, downsample=1), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    # boundary shapes
    for b_ind in range(3):
        SL.append(shape=shape_dict.load_boundary_index(ind=b_ind, scaling=1e-5, downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    SL.append(shape=shape_dict.load_shape_index(ind=indlist[0], scaling=1e-5, downsample=2), count=len(poslist), meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))


    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # acc_val = 1e4
    # vel_val = 10
    vel_val = 7

    b_loc = np.array([
        [1423.00000,4188.57143],
        [3112.00000,2919.57143],
        [772.00000,459.57143]
    ])

    # centered at zero
    b_loc -= np.mean(b_loc, axis=0)

    for b_ind in range(3):
        part = particles[b_ind][0]
        part.shift(b_loc[b_ind]*1e-5)

        # doesn't move
        part.movable = 0


    seed(1)
    for i in range(len(poslist)):
        ## single shape
        part = particles[3][i]
        ## multiple shapes
        # part = particles[i][0]

        part.rotate(0 + (random() * (2*np.pi - 0)) )
        part.shift(poslist[i])

        # centroid = np.mean(part.pos, axis=0)
        # # print(centroid)
        # centroid_dir = centroid/np.sqrt(np.sum(centroid**2, axis=0))
        # print(centroid_dir)

        loc_norm = np.sqrt(np.sum(poslist[i]**2, axis=0))
        if loc_norm == 0:
            centroid_dir = 0
        else:
            centroid_dir = poslist[i]/loc_norm

        # part.acc += [acc_val, 0]
        # part.extforce += [acc_val * part.material.rho, 0]

        # part.acc += -acc_val*centroid_dir
        # part.extforce += (-acc_val*centroid_dir*part.material.rho)

        part.vel += -vel_val*centroid_dir

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles)

    return Experiment(particles, wall, contact)

def recttest():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3
    contact_radius = 1e-3;

    # wall info
    wall_left   = -30e-3
    wall_right  = 30e-3
    wall_top    = 30e-3
    wall_bottom = -30e-3

    # 15 indices
    indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    ## scaling factor for shapes to make them about the same size
    scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 5
    ny = 5

    # poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()

    l = 15e-3
    s = 10e-3

    SL = ShapeList()

    ## single shape, repeated
    # SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=0, scaling=1e-5, downsample=1), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=indlist[0], scaling=1e-5, downsample=2), count=len(poslist), meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=0.1))

    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # acc_val = 1e4
    # vel_val = 10
    vel_val = 7

    # seed(1)
    # for i in range(len(poslist)):
        # ## single shape
        # part = particles[0][i]
        # ## multiple shapes
        # # part = particles[i][0]

        # part.rotate(0 + (random() * (2*np.pi - 0)) )
        # part.shift(poslist[i])

        # # centroid = np.mean(part.pos, axis=0)
        # # # print(centroid)
        # # centroid_dir = centroid/np.sqrt(np.sum(centroid**2, axis=0))
        # # print(centroid_dir)

        # loc_norm = np.sqrt(np.sum(poslist[i]**2, axis=0))
        # if loc_norm == 0:
            # centroid_dir = 0
        # else:
            # centroid_dir = poslist[i]/loc_norm

        # # part.acc += [acc_val, 0]
        # # part.extforce += [acc_val * part.material.rho, 0]

        # # part.acc += -acc_val*centroid_dir
        # # part.extforce += (-acc_val*centroid_dir*part.material.rho)

        # part.vel += -vel_val*centroid_dir

    # clamp nodes on the bottom row
    part = particles[0][0]
    print(part.clamped_nodes)
    for i in range(len(part.pos)):
        if (np.abs(part.pos[i,1] - (-s)) < 1e-12):
            part.clamped_nodes.append(i)
    print(part.clamped_nodes)
    # print('clamped nodes', part.clamped_nodes)


    # plot
    plot_setup(particles)

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def funnel():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3*2
    contact_radius = 1e-3;

    # wall info
    wall_left   = -80e-3
    wall_right  = 80e-3
    wall_top    = 80e-3
    wall_bottom = -80e-3

    # 15 indices
    indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    ## scaling factor for shapes to make them about the same size
    scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 8
    ny = 7

    # borders of particle bulk
    region_left = -40e-3
    region_right = 40e-3
    region_top = 80e-3
    region_bottom = 20e-3

    # poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    poslist = GridDist(region_left, region_right, region_top, region_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()


    SL = ShapeList()

    ## single shape, repeated
    # SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=0, scaling=1e-5, downsample=1), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    # boundary shapes
    bdry_shape_count = 2

    SL.append(shape=shape_dict.load_boundary_index(ind=3, scaling=1, downsample=0), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.load_boundary_index(ind=4, scaling=1, downsample=0), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    SL.append(shape=shape_dict.load_shape_index(ind=indlist[0], scaling=1e-5, downsample=2), count=len(poslist), meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))


    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # acc_val = 1e4
    # vel_val = 10
    vel_val = 10

    # b_loc = np.array([
        # [1423.00000,4188.57143],
        # [3112.00000,2919.57143],
        # [772.00000,459.57143]
    # ])

    # centered at zero
    # b_loc -= np.mean(b_loc, axis=0)

    ## Boundary shapes
    for b_ind in range(bdry_shape_count):
        part = particles[b_ind][0]
        # part.shift(b_loc[b_ind]*1e-5)

        # doesn't move
        part.movable = 0
        part.stoppable = 0


    seed(1)
    for i in range(len(poslist)):
        ## single shape
        part = particles[bdry_shape_count][i]
        ## multiple shapes
        # part = particles[i][0]

        part.rotate(0 + (random() * (2*np.pi - 0)) )
        part.shift(poslist[i])

        # centroid = np.mean(part.pos, axis=0)
        # # print(centroid)
        # centroid_dir = centroid/np.sqrt(np.sum(centroid**2, axis=0))
        # print(centroid_dir)

        loc_norm = np.sqrt(np.sum(poslist[i]**2, axis=0))
        if loc_norm == 0:
            centroid_dir = 0
        else:
            centroid_dir = poslist[i]/loc_norm

        # part.acc += [acc_val, 0]
        # part.extforce += [acc_val * part.material.rho, 0]

        # part.acc += -acc_val*centroid_dir
        # part.extforce += (-acc_val*centroid_dir*part.material.rho)

        part.vel += -vel_val*centroid_dir

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles)

    return Experiment(particles, wall, contact)

def cohesion():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3*2
    contact_radius = 1e-3 * 1.5

    # wall info
    L = 20e-3
    wall_left   = -L
    wall_right  = L
    wall_top    = L
    wall_bottom = -L

    # 15 indices
    indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    ## scaling factor for shapes to make them about the same size
    scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 5
    ny = 5

    # borders of particle bulk
    region_left = wall_left
    region_right = wall_right
    region_top = wall_top
    region_bottom = wall_bottom

    # poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    poslist = GridDist(region_left, region_right, region_top, region_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()


    SL = ShapeList()

    ## single shape, repeated
    # SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_shape_index(ind=0, scaling=1e-5, downsample=1), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    # boundary shapes
    bdry_shape_count = 0

    # SL.append(shape=shape_dict.load_boundary_index(ind=3, scaling=1, downsample=0), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.load_boundary_index(ind=4, scaling=1, downsample=0), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    SL.append(shape=shape_dict.load_shape_index(ind=indlist[0], scaling=1e-5, downsample=2), count=len(poslist), meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))


    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # acc_val = 1e4
    # vel_val = 10
    vel_val = 7

    # b_loc = np.array([
        # [1423.00000,4188.57143],
        # [3112.00000,2919.57143],
        # [772.00000,459.57143]
    # ])

    # centered at zero
    # b_loc -= np.mean(b_loc, axis=0)

    ## Boundary shapes
    for b_ind in range(bdry_shape_count):
        part = particles[b_ind][0]
        # part.shift(b_loc[b_ind]*1e-5)
        # doesn't move
        part.movable = 0
        part.stoppable = 0


    seed(1)
    for i in range(len(poslist)):
        ## single shape
        part = particles[bdry_shape_count][i]
        ## multiple shapes
        # part = particles[i][0]

        part.rotate(0 + (random() * (2*np.pi - 0)) )
        part.shift(poslist[i])

        # centroid = np.mean(part.pos, axis=0)
        # # print(centroid)
        # centroid_dir = centroid/np.sqrt(np.sum(centroid**2, axis=0))
        # print(centroid_dir)

        loc_norm = np.sqrt(np.sum(poslist[i]**2, axis=0))
        if loc_norm == 0:
            centroid_dir = 0
        else:
            centroid_dir = poslist[i]/loc_norm

        # part.acc += [acc_val, 0]
        # part.extforce += [acc_val * part.material.rho, 0]

        # part.acc += -acc_val*centroid_dir
        # part.extforce += (-acc_val*centroid_dir*part.material.rho)

        # part.vel += -vel_val*centroid_dir

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def cohesion_disks():
    """ Load shapes
    """

    delta = 5e-3
    meshsize = 1e-3
    contact_radius = 1e-3

    # wall info
    L = 20e-3
    wall_left   = -L
    wall_right  = L
    wall_top    = L
    wall_bottom = -L

    # 15 indices
    # indlist = [0, 114, 120, 14, 150, 15, 16, 22, 2, 33, 44, 5, 68, 99, 9]
    # ## scaling factor for shapes to make them about the same size
    # scalinglist = [1, 1, 3, 3, 3, 2, 5, 3, 3, 8, 1, 5, 8, 7, 10]

    nx = 5
    ny = 5

    # borders of particle bulk
    region_left = wall_left
    region_right = wall_right
    region_top = wall_top
    region_bottom = wall_bottom

    # poslist = GridDist(wall_left, wall_right, wall_top, wall_bottom, nx, ny).gridloc()
    poslist = GridDist(region_left, region_right, region_top, region_bottom, nx, ny).gridloc()
    # plt.scatter(poslist[:,0], poslist[:,1])
    # plt.show()

    min_particle_spacing = contact_radius*1.001
    c = min_particle_spacing/2
    P = np.array([
        [region_left + c, region_bottom + c ],
        [region_right - c, region_bottom + c],
        [region_right - c, region_top/2 - c],
        [region_left + c, region_top - c]
        ])
    # particle generation spacing
    P_meshsize = 4e-3
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )
    # bringing them in contact
    msh.incircle_rad -= min_particle_spacing/2 * 0.9

    SL = ShapeList()

    # boundary shapes
    bdry_shape_count = 0

    for i in range(msh.count()):
        SL.append(shape=shape_dict.small_disk(scaling=msh.incircle_rad[i], steps=10), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))


    ## multiple shapes
    # for i in range(len(indlist)):
        # ind = indlist[i]
        # SL.append(shape=shape_dict.load_shape_index(ind=ind, scaling=1e-5*scalinglist[i], downsample=2), count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=1, K_scale=1, Gnot_scale=1.2, rho_scale=1))


    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=True)


    seed(1)
    for i in range(len(msh.incircle_rad)):
        ## single shape
        # part = particles[bdry_shape_count][i]
        part = particles[i][0]

        part.rotate(0 + (random() * (2*np.pi - 0)) )
        part.shift(msh.incenter[i])

        g_val = -1e5
        part.acc += g_val
        part.extforce += [0, g_val*part.material.rho]

        # part.vel += -vel_val*centroid_dir

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def cohesion_two():
    """ Load shapes
    """

    delta = 1e-3
    meshsize = delta/5
    contact_radius = delta/2
    rad = 1e-3

    # wall info
    L = 5e-3
    wall_left   = -L
    wall_right  = L
    wall_top    = L
    wall_bottom = -L

    SL = ShapeList()

    SL.append(shape=shape_dict.small_disk(scaling=rad, steps=30), count=2, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)


    dist = 2*rad + contact_radius * 0.9
    particles[0][1].shift([dist, 0])

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def wheel():
    """ Wheel on flat surface
    """

    # whether or not to add particles
    add_particles = 0
    wheel_rad = 3e-3

    # delta = 1e-3
    # delta = 10e-3
    delta = 1e-3
    meshsize = delta/2
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 70e-3
    blade_s = 5e-3

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    cl = 10e-3 # 10 cm

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl/2
    wall_bottom = -cl/2
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=2, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.plank(l=wheel_rad, s=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))


    if add_particles:
        # particle generation spacing
        # P_meshsize = 3.5e-3
        P_meshsize = 35e-3 # cm order
        msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )

        msh.trim(min_rad = 6e-3 + min_particle_spacing/2)

        # reduce radius to avoid contact
        msh.incircle_rad -= min_particle_spacing/2
        msh.info()
        msh.plot(plot_edge = True)
        ###########################
        # particles
        # append each shape with own scaling
        for i in range(msh.count()):
            SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=msh.incircle_rad[i]), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=rho_scale, Gnot_scale=Gnot_scale))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    # wheel location
    move_diag = 0e-3
    particles[0][0].shift( [move_diag+wall_left+wheel_rad+contact_radius*2, move_diag+wall_bottom + wheel_rad + contact_radius])

    # wheel-2
    move_diag = 0e-3
    particles[0][1].shift( [wall_right-wheel_rad-move_diag-contact_radius*2, move_diag+wall_bottom + wheel_rad + contact_radius])

    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1

    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    v_val = -600

    perp = np.array([[0, -1], [1, 0]])

    # r = np.sqrt( np.sum(blade.pos[0]**2) )
    # u_dir = blade.pos[0] / r
    # t_dir = perp @ u_dir 

    # print(blade.vel[0])
    # print(u_dir)
    # print(t_dir)

    #gravity
    g_val = -1e3

    centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    print('centroid', centroid)

    # initial angular velcity
    # for j in range(len(wheel.pos)):
        # r_vec = (wheel.pos[j] - centroid)
        # r = np.sqrt( np.sum(r_vec**2) )
        # u_dir = r_vec / r
        # t_dir = perp @ u_dir
        # wheel.vel[j] += v_val * r * t_dir

    #torque
    wheel.torque_axis = 2
    wheel.torque_val = -1e6
    # gravity
    wheel.extforce += [0, g_val * wheel.material.rho]


    # wheel-2 torque
    particles[0][1].torque_axis = 2
    particles[0][1].torque_val = 1e6
    # gravity
    particles[0][1].extforce += [0, g_val * wheel.material.rho]

    # if add_particles:
        # ## Apply transformation
        # seed(1)
        # g_val = -5e4
        # for i in range(msh.count()):
            # ## scaling is done while generating the mesh
            # ## generate random rotation
            # particles[i+1][0].rotate(0 + (random() * (2*np.pi - 0)) )
            # particles[i+1][0].shift(msh.incenter[i])

            # Initial data
            # particles[0][1].vel += [0, -20]
            # particles[i][0].acc += [0, g_val]
            # particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]
        
        ###########################

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def wheel_on_inclined():
    """ Wheel on inclined plane
    """

    # whether or not to add particles
    wheel_rad = 3e-3

    # delta = 1e-3
    # delta = 10e-3
    delta = 1e-3
    # meshsize = delta/2
    meshsize = delta/3
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 20e-3
    blade_s = 0.5e-3

    # inclination 
    # blade_angle = -np.pi/10
    blade_angle = 0

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    # cl = 10e-3 # 10 cm
    cl = blade_l*1.5

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl
    wall_bottom = -cl
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.pygmsh_geom_test(scaling=wheel_rad, meshsize=meshsize) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.gmsh_test(scaling=wheel_rad, meshsize=meshsize/2) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.wheel_annulus(scaling=wheel_rad, inner_circle_ratio=0.7, meshsize=meshsize, filename_suffix='00') , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.annulus() , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.wheel_ring(scaling=2e-3) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.plank(l=blade_l, s=blade_s) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))


    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    print('Here')

    # plank, rotation
    particles[1][0].rotate(blade_angle)
    particles[1][0].movable =0
    particles[1][0].stoppable =0

    # wheel location
    l_offset = -2e-3
    s_offset = 0e-3

    radial_dir = np.array([-np.cos(-blade_angle) , -np.sin(-blade_angle)])
    tang_dir = np.array([-np.sin(-blade_angle) , np.cos(-blade_angle)])
    shift_vec = (blade_l + l_offset)  * radial_dir + (blade_s + wheel_rad + contact_radius+s_offset) * tang_dir
    # shift_vec = blade_l  * tang_dir
    print('shift_val', shift_vec)
    particles[0][0].shift( shift_vec)


    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1
    #torque
    wheel.torque_val = -5e5
    #gravity
    g_val = -1e3
    wheel.extforce += [0, g_val * wheel.material.rho]

    # initial angular velcity
    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    v_val = -600
    perp = np.array([[0, -1], [1, 0]])
    centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    print('centroid', centroid)
    for j in range(len(wheel.pos)):
        r_vec = (wheel.pos[j] - centroid)
        r = np.sqrt( np.sum(r_vec**2) )
        u_dir = r_vec / r
        t_dir = perp @ u_dir
        wheel.vel[j] += v_val * r * t_dir


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def fixed_vel_angular():
    """ Wheel on inclined plane
    """

    # whether or not to add particles
    wheel_rad = 3e-3

    # delta = 1e-3
    # delta = 10e-3
    delta = 1e-3
    # meshsize = delta/2
    meshsize = delta/3
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 20e-3
    blade_s = 0.5e-3

    # inclination 
    # blade_angle = -np.pi/10
    blade_angle = 0

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    # cl = 10e-3 # 10 cm
    cl = blade_l*1.5

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl
    wall_bottom = -cl
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.pygmsh_geom_test(scaling=wheel_rad, meshsize=meshsize) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.gmsh_test(scaling=wheel_rad, meshsize=meshsize/2) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.wheel_annulus(scaling=wheel_rad, inner_circle_ratio=0.7, meshsize=meshsize, filename_suffix='00') , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.annulus() , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.wheel_ring(scaling=2e-3) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.plank(l=blade_l, s=blade_s) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))


    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    print('Here')

    # plank, rotation
    particles[1][0].rotate(blade_angle)
    particles[1][0].movable =0
    particles[1][0].stoppable =0

    # wheel location
    l_offset = -2e-3
    s_offset = 0e-3

    radial_dir = np.array([-np.cos(-blade_angle) , -np.sin(-blade_angle)])
    tang_dir = np.array([-np.sin(-blade_angle) , np.cos(-blade_angle)])
    shift_vec = (blade_l + l_offset)  * radial_dir + (blade_s + wheel_rad + contact_radius+s_offset) * tang_dir
    # shift_vec = blade_l  * tang_dir
    print('shift_val', shift_vec)
    particles[0][0].shift( shift_vec)


    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1
    #torque
    # wheel.torque_val = -5e5
    #gravity
    g_val = -1e3
    wheel.extforce += [0, g_val * wheel.material.rho]

    # initial angular velcity
    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    # v_val = -400
    # perp = np.array([[0, -1], [1, 0]])
    # centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    # print('centroid', centroid)
    # for j in range(len(wheel.pos)):
        # r_vec = (wheel.pos[j] - centroid)
        # r = np.sqrt( np.sum(r_vec**2) )
        # u_dir = r_vec / r
        # t_dir = perp @ u_dir
        # wheel.vel[j] += v_val * r * t_dir
    # wheel.shift([1e-3, 1e-3])
    # wheel.vel += [1, 1]


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)


def angular_vel():
    """ rotating with angular velocity
    """

    # whether or not to add particles
    add_particles = 0
    wheel_rad = 3e-3

    delta = 1e-3
    meshsize = delta/6
    contact_radius = delta/4

    # particle toughness
    Gnot_scale = 0.7
    rho_scale = 0.7

    # wall info
    cl = 10e-3

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl/2
    wall_bottom = -cl/2
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.wheel_annulus(scaling=wheel_rad, inner_circle_ratio=0.5, meshsize=meshsize, filename_suffix='00') , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1

    ## Angular velocity
    w_val = -600
    perp = np.array([[0, -1], [1, 0]])

    centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    print('centroid', centroid)

    # initial angular velcity
    for j in range(len(wheel.pos)):
        r_vec = (wheel.pos[j] - centroid)
        r = np.sqrt( np.sum(r_vec**2) )
        u_dir = r_vec / r
        t_dir = perp @ u_dir
        wheel.vel[j] += w_val * r * t_dir

    #torque
    # wheel.torque_axis = 2
    # wheel.torque_val = -1e6

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def block_sliding():
    """ Wheel on flat surface
    """

    scaling = 1e-3
    L = 30*scaling
    S = 1*scaling


    angle = np.pi/(6)
    block_l = L/10
    block_s = L/40
    x_offset = 0.8

    delta = 1e-3
    meshsize = delta/4
    meshsize_block = meshsize
    contact_radius = delta/4

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    rho_scale = 1

    wall_left   = -L
    wall_right  = L
    wall_top    = L
    wall_bottom = -L
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # Create a list of shapes
    SL = ShapeList()

    # SL.append(shape=shape_dict.triangle(scaling=1, L=L, angle=angle) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.plank(l=L, s=S) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.plank(l=block_l, s=block_s) , count=1, meshsize=meshsize_block, material=material_dict.peridem_deformable(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=True)

    base = particles[0][0]
    block = particles[1][0]

    # plank base
    base.rotate(-angle)

    block.rotate(-angle)
    # block.shift( [x_offset*L, (contact_radius+block_s)/np.cos(angle) + np.tan(angle) * x_offset * L ])
    block.shift( [x_offset*L/2, (contact_radius+block_s + S)/np.cos(angle) + np.tan(angle) * x_offset * L/2 ])

    #gravity
    g_val = -1e3
    # # gravity
    block.acc += [0, g_val]
    block.extforce += [0, g_val * base.material.rho]

    base.movable = 0
    base.stoppable = 0
        

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=1, contact_radius=contact_radius, wall=wall)

    return Experiment(particles, wall, contact)

def shape():
    """ test new shapes
    """

    delta = 3e-3
    # meshsize = delta/20
    meshsize = delta/30
    contact_radius = delta/10

    # wall info
    wall_left   = -10e-3
    wall_right  = 10e-3
    wall_top    = 10e-3
    wall_bottom = -10e-3


    min_particle_spacing = contact_radius*1.001
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        [wall_right - c, wall_top/2 - c],
        [wall_left + c, wall_top - c]
        ])

    # particle generation spacing
    P_meshsize = 5e-3
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )
    # bringing them in contact
    msh.incircle_rad -= min_particle_spacing/2 * 0.9



    SL = ShapeList()

    # area of the rectangle is (ratio_vol * 2 * scaling^2)
    # ratio_vol = 0.4
    # shape = shape_dict.rect(scaling=delta, ratio_vol=ratio_vol)

    # tau = 0.602
    # shape = shape_dict.wheel_annulus(scaling=delta, inner_circle_ratio=tau, meshsize=meshsize/2, filename_suffix='00')

    theta = 0.4

    from shape_params import Param
    PR = Param(theta)

    # shape = shape_dict.L_symm(scaling=1e-3, c_ratio=c_ratio)
    # shape.plot(plot_bounding_ball=True, bounding_ball_rad=delta, bounding_ball_steps=50)

    # shape = shape_dict.rect(scaling=1e-3, ratio_vol=theta)
    # shape.plot(plot_bounding_ball=True, bounding_ball_rad=delta, bounding_ball_steps=50)

    # print("p", shape.P)
    # shape.plot()


    seed = 3
    steps = 50
    # pertdisk_std = np.sqrt(8/np.pi) -1 
    # phi =  0.0093
    phi =  0.1093
    pertdisk_std = 1 - phi
    incirc_ratio = 0.5
    nu_l = 0.0
    nu_h = 0.0
    # shape = shape_dict.perturbed_disk(seed=seed, steps=steps, scaling=delta, std=pertdisk_std, angle_drift_amp=0.0, angle_drift_std_ratio=0.0 )
    # shape = shape_dict.perturbed_disk(seed=seed, steps=steps, scaling=delta, std=pertdisk_std)
    # shape = shape_dict.pertdisk_incirc(seed=seed, steps=steps, scaling=delta, incirc_ratio=incirc_ratio)
    # shape = shape_dict.star(scaling=delta, incirc_ratio=incirc_ratio)
    # shape = shape_dict.pertdisk_by_mean(seed=seed, steps=steps, scaling=delta, mean_r_ratio=incirc_ratio, noise_unif_low=nu_l, noise_unif_high=nu_h)
    shape = shape_dict.pertdisk_by_mean(seed=seed, steps=steps, scaling=delta, mean_r_ratio=incirc_ratio, noise_unif_low=nu_l, noise_unif_high=nu_h, angle_drift_amp=0.2, angle_drift_supp_ratio=0.1)

    shape.plot(bdry_arrow=False, extended_bdry=False, plot_bounding_ball=True, bounding_ball_rad=delta, bounding_ball_steps=50, plot_included_ball=True, included_ball_rad=delta*incirc_ratio, included_ball_steps=50)

    for param in np.linspace(0.1, 1.0, 10):
        print(param)

        incirc_ratio = param
        seed = 1
        steps = 20
        nu_l = 0.01
        nu_h = 0.01
        supp_rat = incirc_ratio
        # drift_type = 'sin_full'
        # drift_type = 'sin'
        drift_type = 'exp'

        # shape = shape_dict.perturbed_disk(seed=seed, steps=steps, scaling=delta, std=pertdisk_std, angle_drift_amp=0.0, angle_drift_std_ratio=0.0 )
        # shape = shape_dict.perturbed_disk(seed=seed, steps=steps, scaling=delta, std=pertdisk_std)
        shape = shape_dict.pertdisk_incirc(seed=seed, steps=steps, scaling=delta, incirc_ratio=incirc_ratio)
        # shape = shape_dict.star(scaling=delta, incirc_ratio=incirc_ratio)
        # shape = shape_dict.pertdisk_by_mean(seed=seed, steps=steps, scaling=delta, mean_r_ratio=incirc_ratio, noise_unif_low=nu_l, noise_unif_high=nu_h)
        # shape = shape_dict.pertdisk_by_mean(seed=seed, steps=steps, scaling=delta, mean_r_ratio=0.5, noise_unif_low=nu_l, noise_unif_high=nu_h, angle_drift_amp=0.2, angle_drift_supp_ratio=supp_rat, drift_type=drift_type)

        shape.plot(bdry_arrow=False, extended_bdry=False, plot_bounding_ball=True, bounding_ball_rad=delta, bounding_ball_steps=50, plot_included_ball=True, included_ball_rad=delta*incirc_ratio, included_ball_steps=50)
        # shape.plot(bdry_arrow=False, extended_bdry=False, plot_bounding_ball=True, bounding_ball_rad=delta, bounding_ball_steps=50, plot_included_ball=True, included_ball_rad=0.5*delta, included_ball_steps=50)


    # SL.append(shape=shape, count=1, meshsize=meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=0.1))

    for i in range(msh.count()):
        # seed = i
        this_meshsize = meshsize /1e-3 * msh.incircle_rad[i]
        # this_meshsize = meshsize
        # if msh.incircle_rad[i]/1e-3 < 1:
            # this_meshsize = meshsize /1e-3 * msh.incircle_rad[i]
        # else:
            # this_meshsize = meshsize


        # shape = shape_dict.perturbed_disk(seed=seed, steps=steps, scaling=delta, std=pertdisk_std, angle_drift_amp=0.0, angle_drift_std_ratio=0.0 )
        # shape = shape_dict.L_symm(scaling=msh.incircle_rad[i], c_ratio=PR.c_r)
        # shape = shape_dict.wheel_annulus(scaling=msh.incircle_rad[i], inner_circle_ratio=PR.tau, filename_suffix=str(i), meshsize=this_meshsize)
        shape = shape_dict.pertdisk_incirc(seed=seed, steps=steps, scaling=delta, incirc_ratio=incirc_ratio)


        SL.append(shape=shape, count=1, meshsize=this_meshsize, material=peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=0.1))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    g_val = -1e4
    for i in range(msh.count()):
        part = particles[i][0]
        part.rotate(0 + (random() * (2*np.pi - 0)) )
        # part.scale(msh.incircle_rad[i]/1e-3)
        part.shift(msh.incenter[i])
        part.vel += [0, -5]

        part.acc += [0, g_val]
        part.extforce += [0, g_val * part.material.rho]


    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    # plot_setup(particles, wall=wall, dotsize=1)
    plot_setup(particles, wall=wall, dotsize=1.0, adaptive_dotsize=True)

    return Experiment(particles, wall, contact)

def belt():
    """ Belt going over wheel
    """

    wheel_rad = 10e-3
    inner_circle_ratio = 0.9

    steps=20
    small_rad = 2e-3


    delta = 1e-3
    meshsize = 1e-3/6
    contact_radius = 1e-3/4

    show_plot = False

    SL = ShapeList()

    # shape = shape_dict.wheel_annulus(scaling=wheel_rad, inner_circle_ratio=inner_circle_ratio, meshsize=meshsize, filename_suffix='00')

    belt_teethn = 20
    half_thickness_ratio = 0.95
    tooth_gap_ratio = 0.75
    shape = shape_dict.belt_reverse_gear(scaling=wheel_rad,teethn=belt_teethn, half_thickness_ratio=half_thickness_ratio,  inner_circle_ratio=inner_circle_ratio, filename_suffix='belt', meshsize=meshsize, tooth_gap_ratio=tooth_gap_ratio)

    material = material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1)
    SL.append(shape=shape, count=1, meshsize=meshsize, material=material)

    #shape=shape_dict.small_disk(steps=steps, scaling=small_rad)
    # shape=shape_dict.gear_inscribed(steps=steps, scaling=small_rad, teethn=5)
    hollow = True
    gear_inner_circle_ratio = 0.7
    hollow_inner_rad = 0.6
    gear_teethn = 5
    shape = shape_dict.gear_inscribed_par(scaling=small_rad,teethn=gear_teethn, inner_circle_ratio=gear_inner_circle_ratio, filename_suffix='gear', meshsize=meshsize, hollow=hollow, hollow_inner_rad=hollow_inner_rad)

    material = material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1)
    SL.append(shape=shape, count=1, meshsize=meshsize, material=material)

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=True, plot_node_text=True, shapes_in_parallel=False)

    # particles[0][0].movable = 0
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]
    particles[0][0].shift([0, -wheel_rad+small_rad+(1-inner_circle_ratio)*wheel_rad + contact_radius])
    #clamp
    # particles[0][0].clamped_nodes.append(100)

    # particles[1][0].movable = 0
    particles[1][0].stoppable = 0
    #particles[1][0].acc += [0, -5e4]
    #particles[1][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    #particles[1][0].stoppable = 0


    # wall info
    wall_left   = -30e-3
    wall_right  = 30e-3
    wall_top    = 30e-3
    wall_bottom = -30e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=5, contact_radius=None, delta=delta, wall=wall, show_plot=show_plot, adaptive_dotsize=False, show_particle_index=False, save_filename='setup.png')

    return Experiment(particles, wall, contact)

def twogears():
    """ One gear rotating the other
    """

    steps=40

    teethn=8
    teethl=0.8
    inner_circle_ratio = 0.6

    ring=True
    ring_inner_ratio = 0.5

    half_thickness_ratio = 0.8
    small_rad = 2e-3


    # delta = 1e-3
    # debug
    delta = 1e-3
    meshsize = 1e-3/8
    contact_radius = 1e-3/3;

    show_plot = True

    SL = ShapeList()

    shape=shape_dict.gear_inscribed(steps=steps, scaling=small_rad, teethn=teethn, teethl=teethl)
    # shape = shape_dict.gear_inscribed_par(scaling=small_rad,teethn=teethn, inner_circle_ratio=inner_circle_ratio, filename_suffix='00', meshsize=meshsize, ring=ring, ring_inner_rad=ring_inner_ratio)
    # shape = shape_dict.belt_reverse_gear(scaling=small_rad,teethn=teethn, half_thickness_ratio=half_thickness_ratio,  inner_circle_ratio=inner_circle_ratio, filename_suffix='00', meshsize=meshsize)
    material = material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1)
    SL.append(shape=shape, count=2, meshsize=meshsize, material=material)

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, plot_node_text=True, shapes_in_parallel=False)

    # particles[0][0].movable = 0
    # particles[0][0].acc += [0, -5e4]
    # particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]
    particles[0][0].rotate(-2*np.pi/teethn/4)
    particles[0][0].shift([(1 + teethl) * small_rad + contact_radius, 0])
    #clamp
    # particles[0][0].clamped_nodes.append(100)

    # particles[1][0].movable = 0
    # particles[0][1].rotate(-2*np.pi/teethn/4 - 2*np.pi/steps)
    particles[0][1].stoppable = 0
    #particles[1][0].acc += [0, -5e4]
    #particles[1][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    #particles[1][0].stoppable = 0


    # wall info
    wall_left   = -10e-3
    wall_right  = 10e-3
    wall_top    = 10e-3
    wall_bottom = -10e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot_setup(particles, dotsize=5, contact_radius=contact_radius, delta=delta, wall=wall, show_plot=show_plot, adaptive_dotsize=False, show_particle_index=True, save_filename='setup.png')

def test3d(dim=3):
    """ Two particles colliding in 3D
    """

    rad = 1e-3
    delta = rad/2
    meshsize = rad/4
    contact_radius = rad/3;    # conserves momentum better (than delta/3)

    SL = ShapeList()

    shape=shape_dict.sphere_3d(rad=rad, meshsize=meshsize)
    material = material_dict.peridem_3d(delta)
    # material = material_dict.kalthoff3d(delta)

    SL.append(shape=shape, count=2, meshsize=meshsize, material=material)
    # SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
    # SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
    # SL.append(shape=shape_dict.disk_w_hole_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
    # SL.append(shape=shape_dict.plus_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

    material.print()

    particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius, plot_node_text=False)

    # apply transformation
    particles[0][0].rotate3d('z', np.pi/2)
    particles[0][1].rotate3d('z', -np.pi/2)
    particles[0][1].shift([0, 0, 2*rad+contact_radius*1.1])
    # particles[0][1].shift([0, 0, 2.6e-3])

    # Initial data
    particles[0][1].vel += [0, 0, -20]
    # particles[0][1].acc += [0, 0, -16e4]
    # particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]

    # wall info

    L = 4e-3
    x_min = -L
    y_min = -L
    z_min = -L
    x_max = L
    y_max = L
    z_max = L
    wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
    # wall = Wall3d(0)

    # contact properties
    # normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));

    normal_stiffness = material.cnot / contact_radius
    # normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

    damping_ratio = 0.8
    friction_coefficient = 0.8

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    plot3d_setup(particles, dotsize=15, wall=wall, show_particle_index=True, delta=delta, contact_radius=contact_radius)

    return Experiment(particles, wall, contact)

def hopper():
    """ Load shapes
    """

    particle_rad = 8e-3
    # note: delta = (8/3)e-3 and dt = 1e-6 is stable
    delta = particle_rad/2
    particle_meshsize = particle_rad/3
    contact_radius = particle_rad/3

    wall_meshsize = particle_rad/3

    # wall info
    # L = 200e-3
    L = 100e-3

    wall_left   = -L
    wall_right  = L
    wall_top    = 2*L
    wall_bottom = -2*L

    hopper_ratio = 2.5
    hopper_gap = hopper_ratio * particle_rad 
    l = (wall_right - wall_left - 2*hopper_gap)/4
    s = 1e-3

    nx = 10
    ny = 10

    acc_val = -10
    material = peri_deformable(delta, G_scale=0.6, K_scale=0.5, Gnot_scale=0.1, rho_scale=1)

    # show_plot = False
    show_plot = True
    dotsize = 5

    # borders of particle bulk
    region_left = wall_left+contact_radius/2
    region_right = wall_right-contact_radius/2
    region_top = wall_top-contact_radius/2
    region_bottom = 0+contact_radius/2

    poslist = GridDist(region_left, region_right, region_top, region_bottom, nx, ny).gridloc()

    SL = ShapeList()

    # boundary shapes: 0
    shape = shape_dict.plank(l=l, s=s)
    SL.append(shape=shape, count=2, meshsize=wall_meshsize, material=material)
    
    # particle shapes: 1
    shape  = shape_dict.small_disk(scaling=particle_rad)
    SL.append(shape=shape, count=len(poslist), meshsize=particle_meshsize, material=material)

    particles = SL.generate_mesh(dimension=2, contact_radius=contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=True)

    ## Boundary shapes
    particles[0][0].shift([-L+l,-s])
    particles[0][1].shift([L-l,-s])

    particles[0][0].movable = 0
    particles[0][1].movable = 0
    particles[0][0].stoppable = 0
    particles[0][1].stoppable = 0

    for i in range(len(poslist)):
        part = particles[1][i]
        part.shift(poslist[i])

        part.acc += [0, acc_val]
        part.extforce += [0, acc_val * part.material.rho]

    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * peri_deformable(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # plot
    plot_setup(particles, wall=wall, contact_radius=contact_radius, delta=delta, show_plot=show_plot, dotsize=dotsize)

    return Experiment(particles, wall, contact)
