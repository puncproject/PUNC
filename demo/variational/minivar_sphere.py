from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import FloatingBC

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
boundaries = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

# Simulation settings
Q = Constant(10.) # Object 1 charge

ri = 0.2
EPS = 1e-4
EPS = DOLFIN_EPS
class ConstantBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return bool(np.abs(np.linalg.norm(x)-ri)<EPS) and on_bnd

    def map(self, x, y):
        if self.inside(x, True):
            y[0] = +ri
            y[1] = 0
            y[2] = 0
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]

W = FunctionSpace(mesh, 'CG', 1, constrained_domain=ConstantBoundary())
u = TrialFunction(W)
v = TestFunction(W)

ext_bc = DirichletBC(W, Constant(0), boundaries, ext_bnd_id)
int_bc = FloatingBC(W, boundaries, int_bnd_id)

rho = Expression("100*x[0]", degree=2)
# rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = inner(grad(u), grad(v)) * dx
L = inner(rho, v) * dx + inner(v, Q/S)*dsi

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

Qm = assemble(dot(grad(wh), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << wh
