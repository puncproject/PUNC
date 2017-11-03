from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *

def float_object(A, int_bc):
    ind = int_bc.get_boundary_values().keys()

    delete_first = False
    manual_first = True

    # delete (actually zero out) first row belonging to object
    if delete_first:
        A.zero(np.array([ind[0]],dtype=np.intc))

    for i in ind[delete_first:]:
        neighbors = A.getrow(i)[0]

        surface_neighbors = np.array([n for n in neighbors if n in ind])
        values = -np.ones(surface_neighbors.shape)

        self_index = np.where(surface_neighbors==i)[0][0]
        num_of_neighbors = len(surface_neighbors)-1
        values[self_index] = num_of_neighbors

        A.setrow(i, surface_neighbors, values)

        A.apply('insert')

    if manual_first:
        cols = np.array([  0, 247, 248, 256, 257, 265], dtype=np.uintp)
        block = np.array([ 0.75043089,  0.01699319, -0.81002774, -0.1727715 ,  1.64745977,
           -0.68165371],dtype=np.float_)
        A.setrow(257, cols, block)
        A.apply('insert')

# def delete_first(A, b, int_bc):
#     remove = int_bc.get_boundary_values().keys()[0]
#     num_of_rows = A.size(1)
#     An = df.Matrix()
#     bn = df.Vector()
#
#     for i in list(range(remove))+list(range(remove+1,num_of_rows))
#         cols, block = A.getrows(i)
#         An.setrow()



# Upload the mesh and the bouandaries
mesh = df.Mesh("mesh/circle_fine.xml")
boundaries = df.MeshFunction("size_t", mesh, "mesh/circle_fine_facet_region.xml")
ext_bnd_id = 1
int_bnd_id = 2

# Check the numbering of the boundaries
# df.plot(boundaries)

# Create the single circular object
r = 0.5
tol = 0.00099
s = np.array([np.pi, np.pi])

def func(x): return np.dot(x - s, x - s) <= r**2 + tol

class Circle(df.SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and func(x)

objects = Circle()

cell = mesh.ufl_cell()  # ufl cell
V = df.FiniteElement("Lagrange", cell, 1)  # CG elements of order 1
R = df.FiniteElement("Real", cell, 0)     # Real elements of order 0

# Create the mixed function space
W = df.FunctionSpace(mesh, df.MixedElement([V, R]))

# Create DirichletBC of value 0 on the exterior boundaries
ext_bc = df.DirichletBC(W.sub(0), df.Constant(0), boundaries, ext_bnd_id)
int_bc = df.DirichletBC(W.sub(0), df.Constant(0), boundaries, int_bnd_id)
int_bc2 =   FloatingBC(W.sub(0), boundaries, int_bnd_id)

# Create trial and test functions
u, c = df.TrialFunctions(W)
v, d = df.TestFunctions(W)

# The object charge
Q = df.Constant(10.)

# Charge density in the domain
# rho = df.Constant(0.)
rho = df.Expression("sin(x[0]-x[1])", degree=3)
# rho = df.Expression("(x[0]-pi)*(x[0]-pi)+(x[1]-pi)*(x[1]-pi)<=r*r ? 0.0 : sin(x[0]-x[1])", pi=np.pi, r=r, degree=2)

# The normal vector to the facets
n = df.FacetNormal(mesh)

# The measure on exterior boundaries
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

# The surface area of the object
surface_integral = df.assemble(df.Constant(1) * ds(int_bnd_id))
surf_inv = df.Constant(1. / surface_integral)
sig = df.Constant(Q / surface_integral)

# Bilinear form
a = df.inner(df.grad(u), df.grad(v)) * df.dx +\
    df.inner(c, df.dot(df.grad(v), n)) * ds(int_bnd_id) +\
    df.inner(d, df.dot(df.grad(u), n)) * ds(int_bnd_id) -\
    df.inner(v, df.dot(df.grad(u), n)) * ds(int_bnd_id)


# Linear form
L = df.inner(rho, v) * df.dx + df.inner(sig, d) * ds(int_bnd_id)

# Create the solution function
wh = df.Function(W)


# Assemble the system and apply boundary condition (exterior boundaries)
# A, b = df.assemble_system(a, L, [ext_bc,int_bc])
# A, b = df.assemble_system(a, L, [ext_bc])
A = df.assemble(a)
b = df.assemble(L)
ext_bc.apply(A)
ext_bc.apply(b)
# int_bc.apply(A)
# int_bc.apply(b)
# float_object(A, int_bc)
int_bc2.apply(A)
int_bc2.apply(b)


# Solve the linear system with a direct solver
df.solve(A, wh.vector(), b)

# (Alternative) Solve the linear system with a Krylov solver
# solver = df.PETScKrylovSolver('gmres', 'hypre_amg')
# solver.parameters['absolute_tolerance'] = 1e-14
# solver.parameters['relative_tolerance'] = 1e-12
# solver.parameters['maximum_iterations'] = 1000
# # solver.set_operator(A)
# # solver.solve(wh.vector(), b)
# solver.solve(A, wh.vector(), b)

# Split the solution function
uh, ph = wh.split(deepcopy=True)

# Tests and plots
# print("Tangential component of E on object: ", df.assemble(df.dot(-df.grad(uh), tt) * ds(int_bnd_id)))
print("Object charge: ", df.assemble(df.dot(-df.grad(uh), -n) * ds(int_bnd_id)))
print("Surface integral of object: ",
      df.assemble(df.Constant(1) * ds(int_bnd_id)), ", analytical:", 2*np.pi*r)

df.plot(uh, interactive=True)
df.File("phi.pvd") << uh

#---------------------Tests-----------------------------------------------------
# Create a DirichletBC on the object boundary

# A.getrow(1188)

uu = df.Function(W)
uu.vector()[:] = 1
int_bc.apply(uu.vector())
# df.plot(uu[0])

inds = np.where(uu.vector().array() == 0)[0]
print("indices: ", inds)
#-------------------------------------------------------------------------------
# df.interactive()
