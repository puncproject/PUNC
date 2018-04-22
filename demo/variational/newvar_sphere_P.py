from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ConstantBC import ConstantBC
import time

mesh = Mesh("../../mesh/3D/sphere_in_sphere_res1.xml")
bnd = MeshFunction("size_t", mesh, "../../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

mesh.init()
facet_on_bnd_id = bnd.where_equal(int_bnd_id)[0]
facet_on_bnd = list(facets(mesh))[facet_on_bnd_id]
vertex_on_bnd_id = facet_on_bnd.entities(0)[0]
vertex_on_bnd = mesh.coordinates()[vertex_on_bnd_id]

cell = mesh.ufl_cell()
RE = FiniteElement("Real", cell, 0)
R = FunctionSpace(mesh, RE)

print(vertex_on_bnd)

# Simulation settings
Q = Constant(5.) # Object 1 charge

ri = 0.2
EPS = DOLFIN_EPS

class ConstantBoundary(SubDomain):

    def inside(self, x, on_bnd):
        on_vertex = np.linalg.norm(x-vertex_on_bnd[:len(x)])<EPS
        # on_sphere = np.linalg.norm(x)-1*ri<EPS
        # is_inside = on_bnd and on_sphere and on_vertex
        # return is_inside
        return on_vertex

    def map(self, x, y):
        on_sphere = np.linalg.norm(x)-1*ri<EPS
        on_vertex = np.linalg.norm(x-vertex_on_bnd[:len(x)])<EPS
        if on_sphere and not on_vertex:
            y[0] = ri
            y[1] = 0
            y[2] = 0
        else:
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2]


W = FunctionSpace(mesh, 'CG', 1, constrained_domain=ConstantBoundary())
#W = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(W)
v = TestFunction(W)

r = TrialFunction(R)
d = TestFunction(R)

ext_bc = DirichletBC(W, Constant(0), bnd, ext_bnd_id)
# ext_bc = DirichletBC(W, Expression("x[0]", degree=1), bnd, ext_bnd_id)

rho = Expression("100*x[0]", degree=2)
# rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = inner(grad(u), grad(v))*dx
L = inner(rho, v)*dx
a0 = inner(d, dot(grad(u), n))*dsi

wh = Function(W)

print("Assembling matrix")
A = assemble(a)
b = assemble(L)
A0 = assemble(a0)

print("Applying boundary conditions")
ext_bc.apply(A)
ext_bc.apply(b)

print("Setting final dof on boundary to enforce inner(dot(grad(u), n))*dsi = Q")
int_bc = DirichletBC(W, 2, bnd, int_bnd_id)
ww = Function(W)
int_bc.apply(ww.vector())
con_dof = np.where(ww.vector().array() == 2)[0][0] # The dof all constrained are mapped into

row, col = A0.getrow(0)
row = row[np.where(abs(col) > 1e-10)[0]]
col = col[row]
r0, c0 = A.getrow(con_dof)
A.setrow(con_dof, r0, np.zeros_like(c0))  # Nullify row
A.apply('insert')
A.setrow(con_dof, row, col)               # Enforce inner(dot(grad(u), n))*dsi = Q
b[con_dof] = Q(0)
A.apply('insert')

print("Solving equation using iterative solver")
solver = PETScKrylovSolver('gmres','hypre_amg')
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10 #e-12
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = True

solver.set_operator(A)
t0 = time.time()
solver.solve(wh.vector(), b)
t1 = time.time()
print(t1-t0)

# solve(A, wh.vector(), b)

Qm = assemble(dot(grad(wh), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << wh
