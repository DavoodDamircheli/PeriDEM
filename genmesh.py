import pygmsh
import numpy as np
import matplotlib.pyplot as plt

# import optimesh

# 3d plot
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import meshio

class ReturnValue(object):
    """Returns the position of nodes and nodal volume"""
    def __init__(self, pos, vol, bdry_nodes, T, bdry_edges):
        self.pos = pos
        self.vol = vol
        self.T = T
        self.bdry_edges = bdry_edges

        self.bdry_nodes = bdry_nodes

    def get_edges(self):
        """returns all the edges connecting various nodes
        """
        E1 = np.c_[self.T[:,0], self.T[:,1]]
        E2 = np.c_[self.T[:,1], self.T[:,2]]
        E3 = np.c_[self.T[:,2], self.T[:,0]]

        E = np.r_[E1, E2, E3]
        # sort
        E.sort(axis = 1)
        # select the unique rows
        E = np.unique(E, axis = 0)
        return E

    def plot(self, dotsize=10, plot_node_text=True, highlight_bdry_nodes=True, trisurf=True, transparent=True, volume_proportional=True):
        """plot the mesh
        :returns: TODO

        """
        Pos = self.pos
        print("pos is")
        dimension = len(Pos[0])

        if volume_proportional:
            dotsize = self.vol / np.amin(self.vol) * 5

        if dimension==1:
            # Plot the mesh
            plt.scatter(Pos[:,0], Pos[:,1], s = dotsize, marker = '.', linewidth = 0, cmap='viridis')
        elif dimension==2:
            # Plot the mesh
            plt.scatter(Pos[:,0], Pos[:,1], s = dotsize, marker = '.', linewidth = 0, cmap='viridis')

            # highlight boundary nodes
            if highlight_bdry_nodes:
                dotsize = self.vol[self.bdry_nodes] / np.amin(self.vol) * 5
                plt.scatter(Pos[self.bdry_nodes,0], Pos[self.bdry_nodes,1], s = dotsize,  linewidth = 0 )

            # # text
            if plot_node_text:
                for i in range(0,len(Pos)):
                    plt.annotate(str(i), (Pos[i,0], Pos[i,1]))


            plt.axis('scaled')

        elif dimension==3:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = Axes3D(fig)
            # ax = plt.axes(projection='3d')

            # ax = Axes3D(fig, auto_add_to_figure=False)
            # fig.add_axes(ax, auto_add_to_figure=False)

            # Plot the mesh
            #ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], s = dotsize, marker = '.', linewidth = 0, cmap='viridis')

            ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], s = dotsize, marker = '.', linewidth = 0 )
            # highlight boundary nodes
            if highlight_bdry_nodes:
                dotsize = self.vol[self.bdry_nodes] / np.amin(self.vol) * 5
                ax.scatter(Pos[self.bdry_nodes,0], Pos[self.bdry_nodes,1], Pos[self.bdry_nodes,2], s = dotsize,  linewidth = 0 )

            # # text
            if plot_node_text:
                for i in range(0,len(Pos)):
                    ax.text(Pos[i,0], Pos[i,1], Pos[i,2], str(i))

            if trisurf:
                linewidth = 0.1
                alpha = 0.6
                # color = None

                if transparent:
                    color = (0,0,0,0)
                    ax.plot_trisurf(Pos[:,0], Pos[:,1], Pos[:,2], triangles=self.bdry_edges, color=color, linewidth=linewidth, antialiased=True, edgecolor='k') 
                else:
                    ax.plot_trisurf(Pos[:,0], Pos[:,1], Pos[:,2], triangles=self.bdry_edges, alpha=alpha, linewidth=linewidth, antialiased=True, edgecolor='k') 

            # if triline:
                # edges= self.bdry_edges
                # V1 = edges[:,0]
                # V2 = edges[:,1]
                # P1 = Pos[V1]
                # P2 = Pos[V2]
                # ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 
                # lc = Line3DCollection(ls, linewidths=linewidth, colors='b')
                # ax.add_collection(lc)


            # ax.set_box_aspect((1,1,1))
            # ax.set_aspect('equal')
            # Create cubic bounding box to simulate equal aspect ratio
            # ax.pbaspect = [1.0, 1.0, 0.25]

            ## Fix aspect ratio
            ## def axisEqual3D(ax):
            # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            # sz = extents[:,1] - extents[:,0]
            # centers = np.mean(extents, axis=1)
            # maxsize = max(abs(sz))
            # r = maxsize/2
            # for ctr, dim in zip(centers, 'xyz'):
                # getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

            # print(np.amax(np.abs(Pos)))
            mx = np.amax(np.abs(Pos))

            XYZlim = [-mx, mx]
            ax.set_xlim3d(XYZlim)
            ax.set_ylim3d(XYZlim)
            ax.set_zlim3d(XYZlim)
            # ax.set_aspect('equal')
            ax.set_box_aspect((1, 1, 1))

            ## Plot properties
            ax.grid(False)
            # plt.axis('off')

            # plt.savefig('mesh.png', dpi=200, bbox_inches='tight')

        plt.show()


        
# def genmesh(P_bdry, meshsize, pygmsh_geom=None, msh_file = None, do_plot = True, dimension = 2, dotsize = 10, mesh_optimize=True):
def genmesh(P_bdry, meshsize, pygmsh_geom=None, msh_file = None, dimension = 2, mesh_optimize=True):
    """Generate a mesh from given polygon
    :P_bdry: an array of boundary points
    :meshsize: 
    :returns: pos and vol
    """

    # mesh
    if msh_file is None:
        if pygmsh_geom is None:
            # print('Generating mesh from poly')
            # with pygmsh.geo.Geometry() as geom:
            with pygmsh.occ.Geometry() as geom:
                polygon1 = geom.add_polygon(
                    P_bdry,
                    mesh_size= meshsize,
                )
                geom.add_physical(polygon1.surface, 'surface1')
                mesh = geom.generate_mesh()
        else:
            # print(pygmsh_geom)
            print('Generating from provided pygmsh_geom object.')
            mesh = pygmsh_geom.generate_mesh()
            # print(mesh)

    else:
        print('Loading mesh from file: ', msh_file)
        mesh = meshio.read(msh_file)

    if mesh_optimize:
        # print('Optimizing mesh')
        # mesh = pygmsh.optimize(mesh, method="")
        # mesh = pygmsh.optimize(mesh, method="Netgen")
        # mesh = optimesh.optimize(mesh, "CVT (block-diagonal)", 1.0e-5, 100)
        pass

    Pos = mesh.points
    total_nodes = len(Pos)
    # print('Mesh nodes: ', total_nodes)

    # elements
    if dimension==1:
        T = mesh.get_cells_type("line")
    elif dimension==2:
        T = mesh.get_cells_type("triangle")
    elif dimension == 3:
        T = mesh.get_cells_type("tetra")

    # print('Total T: ', len(T))

    area = np.zeros((total_nodes, 1))

    # 2D

    if dimension==1:
        Pos = Pos[:,0:2]
    elif dimension==2:
        Pos = Pos[:,0:2]
    # otherwise gmsh produces 3d anyway

    # Generate nodal volume
    if dimension==1:
        for i in range(len(T)):
            l = np.sqrt(np.sum((Pos[T[i,1]] - Pos[T[i,0]])**2))

            # distribute area evenly over the vertices
            j = T[i,0]
            area[j] += l/2
            j = T[i,1]
            area[j] += l/2

    elif dimension==2:
        for i in range(len(T)):
            u_x = Pos[T[i,1], 0] - Pos[T[i,0], 0]
            u_y = Pos[T[i,1], 1] - Pos[T[i,0], 1]

            v_x = Pos[T[i,2], 0] - Pos[T[i,0], 0]
            v_y = Pos[T[i,2], 1] - Pos[T[i,0], 1]

            # cross product, half
            cp = 0.5 * abs(u_x * v_y - u_y * v_x)

            # distribute area evenly over the vertices
            j = T[i,0]
            area[j] += cp/3
            j = T[i,1]
            area[j] += cp/3
            j = T[i,2]
            area[j] += cp/3

    elif dimension==3:
        for i in range(len(T)):

            x1 = Pos[T[i, 0], 0]
            y1 = Pos[T[i, 0], 1]
            z1 = Pos[T[i, 0], 2]

            x2 = Pos[T[i, 1], 0]
            y2 = Pos[T[i, 1], 1]
            z2 = Pos[T[i, 1], 2]

            x3 = Pos[T[i, 2], 0]
            y3 = Pos[T[i, 2], 1]
            z3 = Pos[T[i, 2], 2]

            x4 = Pos[T[i, 3], 0]
            y4 = Pos[T[i, 3], 1]
            z4 = Pos[T[i, 3], 2]

            cp = 1/6*( (x4-x1) * ((y2-y1)*(z3-z1)-(z2-z1)*(y3-y1)) + (y4-y1) * ((z2-z1)*(x3-x1)-(x2-x1)*(z3-z1)) + (z4-z1) * ((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)) )

            cp = np.abs(cp)

            # print(T[i], 'element volume ', cp)


            # distribute volume evenly over the vertices
            j = T[i,0]
            area[j] += cp/4
            j = T[i,1]
            area[j] += cp/4
            j = T[i,2]
            area[j] += cp/4
            j = T[i,3]
            area[j] += cp/4

        # print('Pos', Pos)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], s = 5, marker = '.', linewidth = 0, cmap='viridis')
        # for i in range(0,len(Pos)):
        #     ax.text(Pos[i,0], Pos[i,1], Pos[i,2], str(i))
        # plt.show()



    # Boundary info
    if dimension==1:
        # all nodes are boundary nodes for 1d mesh in 2d manifold
        bdry_edges = mesh.get_cells_type("line")
        temp = bdry_edges.flatten()
        bdry_nodes = list(set(temp))
    elif dimension==2:
        bdry_edges = mesh.get_cells_type("line")
        temp = bdry_edges.flatten()
        bdry_nodes = list(set(temp))
    elif dimension==3:
        # bdry_edges = mesh.get_cells_type("line")
        bdry_edges = mesh.get_cells_type("triangle")
        # bdry_edges = mesh.get_cells_type("tetra")
        temp = bdry_edges.flatten()
        bdry_nodes = list(set(temp))

    nodelist_zero_vol = np.where(area == 0)[0]
    if len(nodelist_zero_vol) > 0:
        print('Caution: there are nodes with zero volume: ', np.where(area == 0)[0])

        print('Removing nodes with zero volume.')
        print('Total nodes before: ', len(Pos) )
        print('Total elements before: ', len(T))
        # print('bdry nodes before', bdry_nodes)

        # print(bdry_nodes)
        #print(bdry_edges)
        # print(Pos)
        # print(area)
        # print(T)


        ## DEbug
        # nodelist_zero_vol[0] = 3

        orig_len = len(Pos)

        Pos = np.delete(Pos, nodelist_zero_vol, axis=0)
        area = np.delete(area, nodelist_zero_vol, axis=0)
        ## delete all rows containing zero volume node
        bdry_nodes = np.array(bdry_nodes, dtype=int)
        bdry_edges = np.array(bdry_edges, dtype=int)
        T = np.array(T, dtype=int)
        for i, znode in enumerate(nodelist_zero_vol):
            bdry_nodes = bdry_nodes[bdry_nodes != znode]
            bdry_edges = bdry_edges[~np.any(bdry_edges == znode, axis=1)]
            T = T[~np.any(T == znode, axis=1)]

        # print('bdry nodes middle', bdry_nodes)

        # create a map for new indices
        arrmap = np.array(list(range(orig_len)))
        for i, node in enumerate(nodelist_zero_vol):
            arrmap[node] = -1
            arrmap[node+1:] -= 1
            # print([[l,m] for l,m in zip(alist,arrmap)])

        # apply the map
        bdry_nodes = arrmap[bdry_nodes]
        bdry_edges = arrmap[bdry_edges]
        T = arrmap[T]

        # ## iteratively rename indices from reduced list (assuming the lists have no item from nodelist_zero_vol)
        # for i, node in enumerate(nodelist_zero_vol):
        #     # bdry_nodes = [(n-1 if n>node else n) for n  in bdry_nodes]
        #     # np.where(bdry_edges > node, bdry_edges-1, bdry_edges)
        #     bdry_nodes[bdry_nodes > node] -= 1
        #     bdry_edges[bdry_edges > node] -= 1
        #     T[T > node] -= 1
        #     print('removed', node, bdry_nodes)

        print('Total nodes after: ', len(Pos))
        print('Total elements after: ', len(T))
        # print('bdry nodes after', bdry_nodes)
        # print('T after', T)
        # print(area)
        #print(bdry_edges)
        # print(Pos)
        # print(bdry_nodes)
        # print(Pos[79], area[79])
        # print(T)


    return ReturnValue(Pos, area, bdry_nodes, T, bdry_edges)
