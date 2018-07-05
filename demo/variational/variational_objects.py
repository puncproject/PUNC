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

mesh = Mesh("../mesh/2D/circle_and_square_in_square_res2.xml")
boundaries = MeshFunction("size_t", mesh,
                          "../mesh/2D/circle_and_square_in_square_res2_facet_region.xml")
ext_bnd_id = 17
int1_bnd_id = 18
int2_bnd_id = 19

# Simulation settings
Q1 = Constant(15.) # Object 1 charge
Q2 = Constant(20.) # Object 2 charge
V12 = Constant(3.)
connected = True # Connect voltage source of voltage V12 between objects
grounded = True # Wether objects should be grounded (Dirichlet)
direct = False # Use direct solver instead of iterative solver?
method = 'bicgstab'
preconditioner = 'ilu'

# New Benchmark results on Sigvald's computer (07.03.18)
# (Not quite sure what changed, but I get faster results)
#
# Connected (not grounded) objects:
# BICGSTAB/ILU  0.16s
# TFQMR/ILU     0.18s
# GMRES/ILU     0.35s
# BICGSTAB/ICC  0.57s
# TFQMR/none    0.72s
# TFQMR/ICC     1.15s
# TFQMR/Jacobi  2.02s
# GMRES/Jacobi  3.72s
# GMRES/None   12.20s
# GMRES/ICC    18.50s
#
# Non-connected, floating objects didn't seem to alter the results
# significantly for TFQMR/none.
#
# Grounded (Dirichlet) objects (as for Capacitance Matrix method).
# GMRES/hypre_AMG   0.03s
# BICGSTAB/ILU      0.09s
# GMRES/ILU         0.13s

# Old Benchmark results on Sigvald's computer.
#
# Connected (not grounded) objects:
# TFQMR/none    2.20s
# BICGSTAB/ICC  2.25s
# TFQMR/ICC     6.95s
# TFQMR/Jacobi 13.60s
# GMRES/Jacobi 13.80s
# GMRES/None   30.10s
# GMRES/ICC    38.50s
#
# Non-connected, floating objects didn't seem to alter the results
# significantly for TFQMR/none.
#
# Grounded (Dirichlet) objects (as for Capacitance Matrix method).
# GMRES/hypre_AMG   0.16s
# BICGSTAB/ICC      0.41s
# TFQMR/none        0.43s


if grounded:
    W = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(W)
    v = TestFunction(W)

    ext_bc = DirichletBC(W, Constant(0), boundaries, ext_bnd_id)
    int1_bc = DirichletBC(W, Constant(0), boundaries, int1_bnd_id)
    int2_bc = DirichletBC(W, Constant(0), boundaries, int2_bnd_id)

else:
    cell = mesh.ufl_cell()
    V = FiniteElement("Lagrange", cell, 1)
    R = FiniteElement("Real", cell, 0)

    W = FunctionSpace(mesh, MixedElement([V, R, R]))
    u, c1, c2 = TrialFunctions(W)
    v, d1, d2 = TestFunctions(W)

    ext_bc = DirichletBC(W.sub(0), Constant(0), boundaries, ext_bnd_id)
    int1_bc = FloatingBC(W.sub(0), boundaries, int1_bnd_id)
    int2_bc = FloatingBC(W.sub(0), boundaries, int2_bnd_id)

# rho = Constant(0.)
rho = df.Expression("100*x[1]", degree=3)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
ds1 = dss(int1_bnd_id)
ds2 = dss(int2_bnd_id)
dsi = ds1+ds2

S1 = assemble(Constant(1.)*ds1)
S2 = assemble(Constant(1.)*ds2)

if grounded:

    # TWO OBJECTS GROUNDED
    a = inner(grad(u), grad(v)) * dx
    L = inner(rho, v) * dx

elif connected:

    # TWO OBJECTS BIASED WRT ONE ANOTHER
    a = inner(grad(u), grad(v)) * dx -\
        inner(v, dot(grad(u), n)) * dsi +\
        inner(c1, dot(grad(v), n)) * dsi +\
        inner(d1, dot(grad(u), n)) * dsi +\
        inner(c2, (1./S2)*v) * ds2 - inner(c2, (1./S1)*v) * ds1 +\
        inner(d2, (1./S2)*u) * ds2 - inner(d2, (1./S1)*u) * ds1

    L = inner(rho, v) * dx +\
        inner((Q1+Q2)/(S1+S2), d1) * dsi +\
        inner(V12/(S1+S2), d2) * dsi

else:

    # TWO INDEPENDENTLY FLOATING OBJECTS
    a = inner(grad(u), grad(v)) * dx -\
        inner(v, dot(grad(u), n)) * dsi +\
        inner(c1, dot(grad(v), n)) * ds1 +\
        inner(d1, dot(grad(u), n)) * ds1 +\
        inner(c2, dot(grad(v), n)) * ds2 +\
        inner(d2, dot(grad(u), n)) * ds2

    L = inner(rho, v) * dx +\
        inner(Q1/S1, d1) * ds1 +\
        inner(Q2/S2, d2) * ds2

wh = df.Function(W)

A = df.assemble(a)
b = df.assemble(L)
ext_bc.apply(A)
ext_bc.apply(b)
int1_bc.apply(A)
int1_bc.apply(b)
int2_bc.apply(A)
int2_bc.apply(b)

if direct:
    solve(A, wh.vector(), b)
else:
    solver = KrylovSolver(method,preconditioner)
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-12 #e-12
    solver.parameters['maximum_iterations'] = 100000
    # solver.parameters['monitor_convergence'] = True
    # solver.parameters['nonzero_initial_guess'] = True

    print("Setting operator (computing preconditioning?)")
    solver.set_operator(A)

    for it in range(3):
        t0 = time.time()
        solver.solve(wh.vector(), b)
        t1 = time.time()
        print("Solving %dst time: %.5f"%(it+1,t1-t0))

if grounded:
    uh = wh
else:
    uh, ph, qh = wh.split(deepcopy=True)

Q1m = assemble(dot(grad(uh), n) * ds1)
Q2m = assemble(dot(grad(uh), n) * ds2)
print("Object 1 charge: ", Q1m)
print("Object 2 charge: ", Q2m)
print("Total charge:", Q1m+Q2m)
print("Potential difference:", uh(0.4,0)-uh(-0.4,0))

df.plot(uh)
df.File("phi.pvd") << uh
