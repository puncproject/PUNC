from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *
import time

def plot_vertices(mesh, meshfunc):
    mvertices = [x.point().array() for x in list(vertices(mesh))]
    xs = [p[0] for p in mvertices]
    ys = [p[1] for p in mvertices]
    zs = meshfunc.array()
    plot(mesh, zorder=0)
    plt.scatter(xs, ys, c=zs, s=10, zorder=1)
    plt.axis('equal')
    plt.colorbar()
    plt.show()

def inside_meshfunc(mesh, constrained_domain):
    meshfunc = MeshFunction('size_t', mesh, 0)
    meshfunc.set_all(0)
    mvertices = [x.point().array() for x in list(vertices(mesh))]
    for i, p in enumerate(mvertices):
        if constrained_domain.inside(p, True):
            meshfunc.array()[i] = 1
    return meshfunc

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

pbcomp = PeriodicBoundaryComputation()

#
# PERIODIC BOUNDARY
#

mesh = UnitSquareMesh(8,8)

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_bnd):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_bnd)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

pbc = PeriodicBoundary()

# mesh.init()
# meshfunc = inside_meshfunc(mesh, pbc)
# plot_vertices(mesh,meshfunc)
# meshfunc = pbcomp.masters_slaves(mesh, pbc, 0)
# plot_vertices(mesh,meshfunc)

#
# CONSTANT BOUNDARY
#
ri = 0.2
EPS = DOLFIN_EPS
# EPS = 1E-1

domain = Circle(Point(0,0),1)-Circle(Point(0,0),ri)
mesh = generate_mesh(domain,10)
vertex_on_bnd = np.array([ri, 0, 0])

class ConstantBoundary(SubDomain):

    def inside(self, x, on_bnd):
        on_vertex = np.linalg.norm(x-vertex_on_bnd[:len(x)])<EPS
        return on_vertex
        # on_sphere = np.linalg.norm(x)-1*ri<EPS
        # is_inside = on_bnd and on_sphere and on_vertex
        # return is_inside

    def map(self, x, y):
        on_sphere = np.linalg.norm(x)-1*ri<EPS
        on_vertex = np.linalg.norm(x-vertex_on_bnd[:len(x)])<EPS
        if on_sphere and not on_vertex:
            y[0] = ri
            y[1] = 0
        else:
            y[0] = x[0]
            y[1] = x[1]

cbc = ConstantBoundary()

# mesh.init()
# meshfunc = inside_meshfunc(mesh, cbc)
# plot_vertices(mesh,meshfunc)
# meshfunc = pbcomp.masters_slaves(mesh, cbc, 0)
# plot_vertices(mesh,meshfunc)

#
# CONSTANT BOUNDARY (GENERIC)
#
mesh = Mesh("../../mesh/2D/circle_and_square_in_square_res1.xml")
bnd = MeshFunction("size_t", mesh, "../../mesh/2D/circle_and_square_in_square_res1_facet_region.xml")
int_bnd_id = 18

class ConstantBoundary(SubDomain):
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

    def inside(self, x, on_bnd):
        # Some FEniCS functions (not all) will pass 3D x even in 2D problems
        x = x[:self.mesh.geometry().dim()]
        return np.linalg.norm(x-self.vertex) < self.tol

    def map(self, x, y):
        if on_bnd_id(x, self.mesh, self.bnd, self.bnd_id, self.tol):
            y[:] = self.vertex
        else:
            y[:] = x[:]

cbc = ConstantBoundary(mesh, bnd, int_bnd_id)

meshfunc = inside_meshfunc(mesh, cbc)
plot_vertices(mesh,meshfunc)
meshfunc = pbcomp.masters_slaves(mesh, cbc, 0)
plot_vertices(mesh,meshfunc)
