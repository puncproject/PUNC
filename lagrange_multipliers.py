from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *

# Upload the mesh and the bouandaries
mesh = df.Mesh("mesh/circle_fine.xml")
boundaries = df.MeshFunction("size_t", mesh, "mesh/circle_fine_facet_region.xml")
ext_bnd_id = 1
object_id = 2

# Check the numbering of the boundaries
df.plot(boundaries)

# Create the single circular object
r = 0.5
tol = 1e-5
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
W = df.FunctionSpace(mesh, df.MixedElement([V, R, R]))

# Create DirichletBC of value 0 on the exterior boundaries
ext_bc = df.DirichletBC(W.sub(0), df.Constant(0), boundaries, ext_bnd_id)

# Create trial and test functions
u, c1, c2 = df.TrialFunctions(W)
v, d1, d2 = df.TestFunctions(W)

# The object charge
Q = df.Constant(1)

# Charge density in the domain
rho = df.Expression("sin(x[0]-x[1])", degree=3)
# rho = df.Expression("(x[0]-pi)*(x[0]-pi)+(x[1]-pi)*(x[1]-pi)<=r*r ? 0.0 : sin(x[0]-x[1])", pi=np.pi, r=r, degree=2)

# The normal vector to the facets
n = df.FacetNormal(mesh)

# The measure on exterior boundaries
ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

# The surface area of the object
surface_integral = df.assemble(df.Constant(1) * ds(object_id))
surf_inv = df.Constant(1. / surface_integral)

# Define cross product in 2D
def Cross(a, b): return a[0] * b[1] - a[1] * b[0]

# Create tangent vector in 2D
tt = df.as_vector((-n[1], n[0]))

# Bilinear form
a = df.inner(df.grad(u), df.grad(v)) * df.dx +\
    df.inner(c1, df.dot(df.grad(v), n)) * ds(object_id) +\
    df.inner(d1, df.dot(df.grad(u), n)) * ds(object_id) +\
    df.inner(c2, df.dot(df.grad(v), tt)) * ds(object_id) +\
    df.inner(d2, df.dot(df.grad(u), tt)) * ds(object_id) -\
    df.inner(v, df.dot(df.grad(u), n)) * ds(object_id)
#inner(c2, dot(Cross(grad(v), n),Cross(grad(v), n)))*ds(object_id)+\
#inner(d2, dot(Cross(grad(u), n),Cross(grad(u), n)))*ds(object_id)

# Linear form
L = df.inner(rho, v) * df.dx + surf_inv * df.inner(Q, d1) * ds(object_id)

# Assemble the system and apply boundary condition (exterior boundaries)
A, b = df.assemble_system(a, L, ext_bc)

# Create the solution function
wh = df.Function(W)

# Solve the linear system with a direct solver
df.solve(A, wh.vector(), b)

# (Alternative) Solve the linear system with a Krylov solver
# solver = df.PETScKrylovSolver('minres', 'hypre_amg')
# solver.set_operator(A)
# solver.solve(wh.vector(), b)

# Split the solution function
uh, ph1, ph2 = wh.split(deepcopy=True)

# Tests and plots
print("Tangential component of E on object: ", df.assemble(df.dot(-df.grad(uh), tt) * ds(object_id)))
print("Object charge: ", df.assemble(df.dot(-df.grad(uh), -n) * ds(object_id)))
print("Surface integral of object: ",
      df.assemble(df.Constant(1) * ds(object_id)), ", analytical:", 2*np.pi*r)

df.plot(uh)
df.File("phi.pvd") << uh

#---------------------Tests-----------------------------------------------------
# Create a DirichletBC on the object boundary
obj_bnd = df.DirichletBC(W.sub(0), df.Constant(0), objects)

# A.getrow(1188)

uu = df.Function(W)
uu.vector()[:] = 1
obj_bnd.apply(uu.vector())
df.plot(uu[0])

inds = np.where(uu.vector().array() == 0)
print("indices: ", inds)
#-------------------------------------------------------------------------------
df.interactive()


