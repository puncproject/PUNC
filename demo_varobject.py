from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
# V = FunctionSpace(mesh, "CG", 1)
# R = FunctionSpace(mesh, "R", 0)
# W = V * R
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
M = MixedElement([R, R])
W = FunctionSpace(mesh, P1 * M)

# Define variational problem
(u, c1, c2) = TrialFunction(W)
(v, d1, d2) = TestFunction(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=2)
g = Expression("0",degree=2)
a = (inner(grad(u), grad(v)) + c1*v + u*d1)*dx + (c2*v + u*d2)*ds
L = f*v*dx + g*v*ds

# Compute solution
w = Function(W)
solve(a == L, w)
(u, c1) = w.split()

# Plot solution
plot(u, interactive=True)
