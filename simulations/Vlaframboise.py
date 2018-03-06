from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import *

fname = "../mesh/3D/sphere_in_sphere_res1"
mesh, bnd = load_mesh(fname)
ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)
int_bnd_id = int_bnd_ids[0]

# Simulation settings
Q = Constant(-10.)

cell = mesh.ufl_cell()
V = FiniteElement("Lagrange", cell, 1)
R = FiniteElement("Real", cell, 0)

W = FunctionSpace(mesh, MixedElement([V, R]))
u, c = TrialFunctions(W)
v, d = TestFunctions(W)

ext_bc = DirichletBC(W.sub(0), Constant(0), bnd, ext_bnd_id)
int_bc = FloatingBC(W.sub(0), bnd, int_bnd_id)

# rho = Expression("10*x[1]", degree=2)
rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
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

print("Applying boundary conditions")
ext_bc.apply(A)
ext_bc.apply(b)
int_bc.apply(A)
int_bc.apply(b)

solve(A, wh.vector(), b)
uh, ph = wh.split(deepcopy=True)

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

File("uh.pvd") << uh

V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, Constant(0.0), bnd, ext_bnd_id)

objects = [Object(V, bnd, i) for i in int_bnd_ids]
poisson = PoissonSolver(V, bc)
esolver = ESolver(V)
objects[0].charge = Q.values()[0]

# object_e_field = solve_laplace(V, poisson, objects, bnd, ext_bnd_id)
# ds = Measure('ds', domain=mesh, subdomain_data=bnd)
# n = FacetNormal(mesh)
# fluxi = inner(object_e_field[0], -1 * n) * ds(objects[0].id)
# print(assemble(fluxi))
# fluxd = dot(object_e_field[0], -1 * n) * ds(objects[0].id)
# print(assemble(fluxd))

inv_cap_matrix = capacitance_matrix(V, poisson, objects, bnd, ext_bnd_id)

reset_objects(objects)
phi = poisson.solve(rho, objects)
E = esolver.solve(phi)
compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
phi = poisson.solve(rho, objects)

Qm = assemble(dot(grad(phi), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << phi
