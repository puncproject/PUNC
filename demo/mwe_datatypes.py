from dolfin import *
import numpy as np
import sys

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
bnd  = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")

if sys.argv[1] == 'True':
    int_bnd_id = np.int64(59)
else:
    int_bnd_id = 59

print(type(int_bnd_id))

dss = Measure('ds', domain=mesh, subdomain_data=bnd)
n = FacetNormal(mesh)

print(assemble(1 * dss(int_bnd_id)))
