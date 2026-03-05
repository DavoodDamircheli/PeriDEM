# this routin is implementing a cube
import gmsh 

import sys

import pdb
ax =  0.2
ay =  0.2
az =  0.2
if __name__ =="__main__":
            gmsh.initialize()
            gmsh.model.add("cube")

            # Create corner points for the cube
            lc = 0.050
            p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
            p2 = gmsh.model.geo.addPoint(ax, 0, 0, lc)
            p3 = gmsh.model.geo.addPoint(ax, ay, 0, lc)
            p4 = gmsh.model.geo.addPoint(0, ay, 0, lc)

            # Create lines connecting the points
            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)
            


            curve_loop1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surface1 = gmsh.model.geo.addPlaneSurface([curve_loop1])
           

            gmsh.model.geo.extrude([(2,surface1)],0,0,az)

            #pdb.set_trace()
            # Create volume
            # gmsh.model.geo.addVolume([surface1, surface2, surface3, surface4, surface5, surface6])

            gmsh.model.geo.synchronize()

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Save the mesh
            gmsh.write("excube.msh")

            gmsh.finalize()
