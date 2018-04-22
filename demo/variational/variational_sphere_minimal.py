from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import FloatingBC

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
boundaries = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

# Simulation settings
Q = Constant(-10.) # Object 1 charge

cell = mesh.ufl_cell()
V = FiniteElement("Lagrange", cell, 1)
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

print("Applying boundary conditions")
ext_bc.apply(A)
ext_bc.apply(b)
int_bc.apply(A)
int_bc.apply(b)

solve(A, wh.vector(), b)
uh, ph = wh.split(deepcopy=True)

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << uh
