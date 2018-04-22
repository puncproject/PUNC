from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import *
import time

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
boundaries = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

# Simulation settings
Q = Constant(-10.) # Object 1 charge
grounded = False # Wether objects should be grounded (Dirichlet)
direct = True # Use direct solver instead of iterative solver?
method = 'gmres'
preconditioner = 'ilu'

if grounded:
    W = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(W)
    v = TestFunction(W)

    ext_bc = DirichletBC(W, Constant(0), boundaries, ext_bnd_id)
    int_bc = DirichletBC(W, Constant(0), boundaries, int_bnd_id)

else:
    cell = mesh.ufl_cell()
    V = FiniteElement("Lagrange", cell, 1)
    R = FiniteElement("Real", cell, 0)

    W = FunctionSpace(mesh, MixedElement([V, R]))
    u, c = TrialFunctions(W)
    v, d = TestFunctions(W)

    ext_bc = DirichletBC(W.sub(0), Constant(0), boundaries, ext_bnd_id)
    int_bc = FloatingBC(W.sub(0), boundaries, int_bnd_id)

# rho = Constant(1.)
rho = df.Expression("100*x[1]", degree=3)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

if grounded:

    a = inner(grad(u), grad(v)) * dx
    L = inner(rho, v) * dx

else:

    # TWO OBJECTS BIASED WRT ONE ANOTHER
    a = inner(grad(u), grad(v)) * dx -\
        inner(v, dot(grad(u), n)) * dsi +\
        inner(c, dot(grad(v), n)) * dsi +\
        inner(d, dot(grad(u), n)) * dsi

    L = inner(rho, v) * dx +\
        inner(Q/S, d) * dsi

wh = df.Function(W)

print("Assembling matrix")
A = df.assemble(a)
b = df.assemble(L)
print("Applying boundary conditions")
ext_bc.apply(A)
ext_bc.apply(b)
int_bc.apply(A)
int_bc.apply(b)

if direct:
    solve(A, wh.vector(), b)
else:
    solver = PETScKrylovSolver(method,preconditioner)
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = True
    solver.parameters['nonzero_initial_guess'] = True

    print("Setting operator (computing preconditioning?)")
    solver.set_operator(A)
    solver.set_reuse_preconditioner(True)

    print("Started solving 1st time")
    t0 = time.time()
    solver.solve(wh.vector(), b)
    # solver.solve(A, wh.vector(), b)
    t1 = time.time()
    print("Time:",t1-t0)

    print("Started solving 2nd time")
    t0 = time.time()
    # solver.solve(A, wh.vector(), b)
    solver.solve(wh.vector(), b)
    t1 = time.time()
    print("Time:",t1-t0)

if grounded:
    uh = wh
else:
    uh, ph = wh.split(deepcopy=True)

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

# df.plot(uh, interactive=True)
# df.File("phi.pvd") << uh
