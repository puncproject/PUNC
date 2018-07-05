from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import FloatingBC
import time

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
boundaries = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

# Simulation settings
Q = Constant(-10.) # Object 1 charge

cell = mesh.ufl_cell()
V = FiniteElement("Lagrange", cell, 2)
R = FiniteElement("Real", cell, 0)

W = FunctionSpace(mesh, MixedElement([V, R]))
u, c = TrialFunctions(W)
v, d = TestFunctions(W)

ext_bc = DirichletBC(W.sub(0), Constant(0), boundaries, ext_bnd_id)
int_bc = FloatingBC(W.sub(0), boundaries, int_bnd_id)

rho = Expression("100*x[1]", degree=2)
# rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = inner(grad(u), grad(v)) * dx -\
    inner(v, dot(grad(u), n)) * dsi +\
    inner(c, dot(grad(v), n)) * dsi +\
    inner(d, dot(grad(u), n)) * dsi

L = inner(rho, v) * dx +\
    inner(Q/S, d) * dsi

wh = Function(W)

print("Assembling matrix")
A = assemble(a)
b = assemble(L)

print("Applying boundary conditions 1")
ext_bc.apply(A)
print("Applying boundary conditions 2")
ext_bc.apply(b)
print("Applying boundary conditions 3")
int_bc.apply(A)
print("Applying boundary conditions 4")
int_bc.apply(b)


# BICGSTAB/ILU : 1.48s
# TFQMR   /NONE: 13.6s
# GMRES   /ILU : 18.0s

method = 'tfqmr'
preconditioner = 'none'

solver = PETScKrylovSolver(method,preconditioner)
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10 #e-12
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = False
# solver.parameters['nonzero_initial_guess'] = True

solver.set_operator(A)
solver.set_reuse_preconditioner(True)

print("Solving Equations")
t0 = time.time()

# solve(A, wh.vector(), b)
solver.solve(wh.vector(), b)

t1 = time.time()
print("Time: ",t1-t0)

print("Splitting Equations")
uh, ph = wh.split(deepcopy=True)

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << uh
