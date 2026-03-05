# this routin is implementing a cube
import gmsh 

import sys

import pdb

if __name__ =="__main__":
            gmsh.initialize()
            gmsh.model.add("cube")

            # Create corner points for the cube
            lc = 0.1
            p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
            p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
            p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
            p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
            p5 = gmsh.model.geo.addPoint(0, 0, 1, lc)
            p6 = gmsh.model.geo.addPoint(1, 0, 1, lc)
            p7 = gmsh.model.geo.addPoint(1, 1, 1, lc)
            p8 = gmsh.model.geo.addPoint(0, 1, 1, lc)

            # Create lines connecting the points
            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)
            l5 = gmsh.model.geo.addLine(p5, p6)
            l6 = gmsh.model.geo.addLine(p6, p7)
            l7 = gmsh.model.geo.addLine(p7, p8)
            l8 = gmsh.model.geo.addLine(p8, p5)
            l9 = gmsh.model.geo.addLine(p1, p5)
            l10 = gmsh.model.geo.addLine(p2, p6)
            l11 = gmsh.model.geo.addLine(p3, p7)
            l12 = gmsh.model.geo.addLine(p4, p8)



            curve_loop1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            gmsh.model.geo.addPlaneSurface([curve_loop1],1)
            curve_loop2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
            gmsh.model.geo.addPlaneSurface([curve_loop2],2)

            curve_loop3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
            gmsh.model.geo.addPlaneSurface([curve_loop3],3)

            curve_loop4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
            gmsh.model.geo.addPlaneSurface([curve_loop4],4)

            curve_loop5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
            gmsh.model.geo.addPlaneSurface([curve_loop5],5)

            curve_loop6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
            gmsh.model.geo.addPlaneSurface([curve_loop6],6)
           

            #Create surfaces
            '''
            curve_loop1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surface1 = gmsh.model.geo.addPlaneSurface([curve_loop1])
            curve_loop2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
            surface2 = gmsh.model.geo.addPlaneSurface([curve_loop2])

            curve_loop3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
            surface3 = gmsh.model.geo.addPlaneSurface([curve_loop3])

            curve_loop4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
            surface4 = gmsh.model.geo.addPlaneSurface([curve_loop4])

            curve_loop5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
            surface5 = gmsh.model.geo.addPlaneSurface([curve_loop5])

            curve_loop6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
            surface6 = gmsh.model.geo.addPlaneSurface([curve_loop6])
            '''
            #pdb.set_trace()
            # Create volume
            # gmsh.model.geo.addVolume([surface1, surface2, surface3, surface4, surface5, surface6])

            gmsh.model.geo.addVolume([1,2,3,4,5,6],1)
            gmsh.model.geo.synchronize()

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Save the mesh
            gmsh.write("cube.msh")

            gmsh.finalize()
