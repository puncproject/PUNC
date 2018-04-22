from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ConstantBC import ConstantBC
import time

def on_bnd_id(x, mesh, bnd, bnd_id, tol=DOLFIN_EPS):
    """
    Returns `True` if a point `x` is on the boundary with id `bnd_id` as given by
    the `FacetFunction` named `bnd`.
    """

    # If changing this function, keep in mind that it should work for all
    # points on boundary facets. Not just the vertices.

    bnd_facets = bnd.where_equal(bnd_id)

    bmesh = BoundaryMesh(mesh, 'exterior')
    facet_dim = bmesh.topology().dim()
    cell_map = bmesh.entity_map(facet_dim)

    for i, facet in enumerate(cells(bmesh)):
        if cell_map[i] in bnd_facets:
            if facet.distance(Point(x))<tol:
                return True

    # Slower but more straight-forward implementation:

    # for facet in facets(mesh):
    #     if facet.index() in bnd_facets:
    #         if facet.collides(Point(x)):
    #             return True


    return False

resolution = 4
order = 3

fname = "../../mesh/3D/sphere_in_sphere_res"+str(resolution)
mesh = Mesh(fname+".xml")
bnd = MeshFunction("size_t", mesh, fname+"_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

mesh.init()
facet_on_bnd_id = bnd.where_equal(int_bnd_id)[0]
facet_on_bnd = list(facets(mesh))[facet_on_bnd_id]
vertex_on_bnd_id = facet_on_bnd.entities(0)[0]
vertex_on_bnd = mesh.coordinates()[vertex_on_bnd_id]

print(vertex_on_bnd)

# Simulation settings
Q = Constant(5.) # Object 1 charge

ri = 0.2
ro = 1.0
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

class ConstantBoundary2(SubDomain):
    """
    Enforces constant values for both `TrialFunction` and `TestFunction` on a
    boundary with id `bnd_id` as given by the `FacetFunction` named `bnd`.
    Assumes some sort of elements where the vertices on the boundary are nodes,
    but not necessarily the only nodes. E.g. CG1, CG2, ... should be fine.

    Usage::
        mesh = Mesh("mesh.xml")
        bnd  = MeshFunction('size_t', mesh, "mesh_facet_region.xml")
        cb   = ConstantBoundary(mesh, bnd, bnd_id)
        V    = FunctionSpace(mesh, 'CG', 2, constrained_domain=cb)
    """

    def __init__(self, mesh, bnd, bnd_id, tol=DOLFIN_EPS):

        SubDomain.__init__(self)
        self.mesh   = mesh
        self.bnd    = bnd
        self.bnd_id = bnd_id
        self.tol    = tol

        # Pick a random vertex on the bnd (assuming this vertex is a node)
        facet_id    = bnd.where_equal(bnd_id)[0]
        facet       = list(facets(mesh))[facet_id]
        vertex_id   = facet.entities(0)[0]
        self.vertex = mesh.coordinates()[vertex_id]

        self.bnd_facets = bnd.where_equal(bnd_id)

        self.bmesh = BoundaryMesh(mesh, 'exterior')
        facet_dim = self.bmesh.topology().dim()
        self.cell_map = self.bmesh.entity_map(facet_dim)

    def on_bnd_id(self, x):

        # If changing this function, keep in mind that it should work for all
        # points on boundary facets. Not just the vertices.

        for i, facet in enumerate(cells(self.bmesh)):
            if self.cell_map[i] in self.bnd_facets:
                if facet.distance(Point(x)) < self.tol:
                    return True

        return False

    def inside(self, x, on_bnd):
        # Some FEniCS functions (not all) will pass 3D x even in 2D problems
        x = x[:self.mesh.geometry().dim()]
        return np.linalg.norm(x-self.vertex) < self.tol

    def map(self, x, y):
        if self.on_bnd_id(x):
            y[:] = self.vertex
        else:
            y[:] = x[:]

# cb = ConstantBoundary2(mesh, bnd, int_bnd_id)
cb = ConstantBoundary()
print("Making constrained function space")
W = FunctionSpace(mesh, 'CG', order, constrained_domain=cb)
print("done")
u = TrialFunction(W)
v = TestFunction(W)

ext_bc = DirichletBC(W, Constant(0), bnd, ext_bnd_id)
int_bc = ConstantBC(W, bnd, int_bnd_id)

rho = Expression("100*x[0]", degree=2)
rho = Constant(0.0)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = inner(grad(u), grad(v))*dx
# L = inner(rho, v)*dx + inner(v, Q/S)*dsi
L = inner(rho, v)*dx + v*(Q/S)*dsi

wh = Function(W)

print("Assembling matrix")
# A = assemble(a)
# b = assemble(L)

# print("Applying boundary conditions")
# ext_bc.apply(A)
# ext_bc.apply(b)
# int_bc.apply(A)
# int_bc.apply(b)

A, b = assemble_system(a, L, ext_bc)

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

print("Making plots")
line = np.linspace(ri,ro,10000, endpoint=False)
uh_line = np.array([wh(x,0,0) for x in line])
ue_line = (Q.values()[0]/(4*np.pi))*(line**(-1)-ro**(-1))

dr = line[1]-line[0]
e_abs = np.sqrt(dr*np.sum((uh_line-ue_line)**2))
e_rel1 = e_abs/np.sqrt(dr*np.sum(ue_line**2))
e_rel2 = np.sqrt(dr*np.sum(((uh_line-ue_line)/ue_line)**2))
hmin = mesh.hmin()
hmax = mesh.hmax()

with open("convergence.txt", "a") as myfile:
    myfile.write("{} {} {} {} {} {} {} {} {}".format(
        str(resolution),
        str(order),
        str(Qm),
        str(e_abs),
        str(e_rel1),
        str(e_rel2),
        str(hmax),
        str(hmin),
        "\n"
    ))

plt.plot(line, uh_line, label='Numerical')
plt.plot(line, ue_line, '--', label='Exact')
plt.legend(loc='lower left')
# plt.show()

# File("phi.pvd") << wh
