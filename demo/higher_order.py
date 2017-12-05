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

mesh, bnd = load_mesh("../mesh/2D/circle_and_square_in_square_res1")
ext_bnd_id = 17
int1_bnd_id = 18
int2_bnd_id = 19

for order in [1,2]:

    V = FunctionSpace(mesh, "CG", order)
    u = TrialFunction(V)
    v = TestFunction(V)

    esolver = ESolver(V)

    ext_bc = DirichletBC(V, Constant(0), bnd, ext_bnd_id)
    int1_bc = DirichletBC(V, Constant(0), bnd, int1_bnd_id)
    int2_bc = DirichletBC(V, Constant(0), bnd, int2_bnd_id)

    rho = df.Expression("100*x[1]", degree=3)

    a = inner(grad(u), grad(v)) * dx
    L = inner(rho, v) * dx

    uh = df.Function(V)

    A = df.assemble(a)
    b = df.assemble(L)
    ext_bc.apply(A)
    ext_bc.apply(b)
    int1_bc.apply(A)
    int1_bc.apply(b)
    int2_bc.apply(A)
    int2_bc.apply(b)

    solver = KrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-12 #e-12
    solver.parameters['maximum_iterations'] = 100000

    solver.solve(A, uh.vector(), b) # dummy for precond

    t0 = time.time()
    solver.solve(A, uh.vector(), b)
    t1 = time.time()

    if order==2:
        E = df.nabla_grad(uh)
    else:
        E = esolver.solve(uh)

    t2 = time.time()

    t_phi = 1000*(t1-t0)
    t_E = 1000*(t2-t1)
    t_tot = 1000*(t2-t0)

    print("Order: %d, t_phi: %f, t_E: %f, t_tot: %f"%(order, t_phi, t_E, t_tot))

    ux = df.Constant((1,0))
    uy = df.Constant((0,1))
    Ex = df.project(df.inner(ux, E))
    Ey = df.project(df.inner(uy, E))
    Em = df.project(df.inner(Ex, Ex)+df.inner(Ey,Ey))

    df.File("phi%d.pvd"%order) << uh
    df.File("Ex%d.pvd"%order) << Ex
    df.File("Ey%d.pvd"%order) << Ey
    df.File("Em%d.pvd"%order) << Em
