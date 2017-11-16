from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
import os
import sys
import importlib


r_inner = 0.2
r_outer = 1.0

phi_inner = 1.0
phi_outer = 10.0

cap_matrix = 4.0*np.pi*r_inner*r_outer/(r_outer-r_inner)

fname = "../mesh/3D/sphere_in_sphere_res1"

change_tag = True
if not change_tag:
  mesh, boundaries = load_mesh(fname)
  ext_bnd_id, int_bnd_ids = get_mesh_ids(boundaries)
else:

  mesh, bnd = load_mesh(fname)
  ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)

  # Rename boundary numbers from GMSH
  boundaries = df.FacetFunction("size_t", mesh)
  boundaries.set_all(0)
  boundaries.array()[bnd.array() == ext_bnd_id] = 10
  boundaries.array()[bnd.array() == int_bnd_ids[0]] = 20
  ext_bnd_id = 10 
  int_bnd_ids= [20]

boundaryMarkers = [ext_bnd_id, int_bnd_ids[0]]
surface_area = [4 * np.pi * r_outer**2, 4 * np.pi * r_inner**2]
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

# Surfaces
for i, marker in enumerate(boundaryMarkers):
  print("surface area: ", marker, " ", surface_area[i], "  ", df.assemble(df.Constant(1.) * ds(marker)))

# df.File("bnd.pvd") << boundaries

V = df.FunctionSpace(mesh, 'CG', 1)

bc = df.DirichletBC(V, df.Constant(phi_outer), boundaries, ext_bnd_id)
objects = [Object(V, j, boundaries) for j in int_bnd_ids]

poisson = PoissonSolver(V, bc)

inv_cap = capacitance_matrix(V, poisson, objects, boundaries, ext_bnd_id)

print("capacitance (numerical): ", 1.0/inv_cap[0,0])
print("capacitance (analytical): ", cap_matrix)
print("Error: ", np.abs(cap_matrix - 1.0/inv_cap[0, 0]))

