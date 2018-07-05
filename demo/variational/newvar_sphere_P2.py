from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ConstantBC import ConstantBC

mesh = Mesh("../ConstantBC/mesh/sphere_in_sphere_res3.xml")
bnd = MeshFunction("size_t", mesh, "../ConstantBC/mesh/sphere_in_sphere_res3_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

mesh.init()
facet_on_bnd_id = bnd.where_equal(int_bnd_id)[0]
facet_on_bnd = list(facets(mesh))[facet_on_bnd_id]
vertex_on_bnd_id = facet_on_bnd.entities(0)[0]
vertex_on_bnd = mesh.coordinates()[vertex_on_bnd_id]

cell = mesh.ufl_cell()
VE = FiniteElement("Lagrange", cell, 1)
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


W = FunctionSpace(mesh, MixedElement([VE, RE]), constrained_domain=ConstantBoundary())

u, r = TrialFunctions(W)
v, d = TestFunctions(W)

#r = TrialFunction(R)
#d = TestFunction(R)

ext_bc = DirichletBC(W.sub(0), Constant(0), bnd, ext_bnd_id)
#ext_bc = DirichletBC(W.sub(0), Expression("x[0]", degree=1), bnd, ext_bnd_id)

#rho = Expression("100*x[0]", degree=2)
rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = dot(grad(u), grad(v))*dx - v*dot(grad(u), n)*dsi + d*dot(grad(u), n)*dsi + r*dot(grad(v), n)*dsi
L = rho*v*dx + Q*d/S*dsi

wh = Function(W)

print("Assembling matrix")
A = assemble(a)
b = assemble(L)

print("Applying boundary conditions")
ext_bc.apply(A, b)
#ext_bc.apply(b)

print("Solving equation using iterative solver")
solver = PETScKrylovSolver('bicgstab','ilu')
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10 #e-12
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = True

solver.set_operator(A)
solver.solve(wh.vector(), b)

# solve(A, wh.vector(), b)
uh, ph = wh.split(deepcopy=True)

Qm = assemble(dot(grad(uh), n) * dsi)
print("Object charge: ", Qm)

File("phi.pvd") << uh

