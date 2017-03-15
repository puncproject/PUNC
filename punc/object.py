from __future__ import print_function
import dolfin as df
import numpy as np
from marker import *


def objects_bcs(objects, inv_cap_matrix):

    bcs = [0.0]*len(objects)
    for i, o in enumerate(objects):
        phi_object = 0.0
        for j, p in enumerate(objects):
            phi_object += (p.charge - p.q_rho)*inv_cap_matrix[i,j]
        o.set_phi(phi_object)
        bcs[i] = o.bc()

    return bcs

def initiate_objects(V, objects, Ld):

    dim = len(Ld)
    L = np.zeros(2*dim)
    L[dim:] = Ld
    n_components = len(objects)

    mesh = V.mesh()
    v2d = df.vertex_to_dof_map(V)

    #---------------------------------------------------------------------------
    #        Create facet and cell functions to to mark the boundaries
    #---------------------------------------------------------------------------
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+2*dim)

    cell_f = df.CellFunction('size_t', mesh)
    cell_f.set_all(n_components)
    #---------------------------------------------------------------------------
    #           Create the objects
    #---------------------------------------------------------------------------
    for i in range(n_components):
        objects[i] = Object(objects[i], i, V, mesh, facet_f, cell_f, v2d)

    #---------------------------------------------------------------------------
    #       Mark the exterior boundaries of the simulation domain
    #---------------------------------------------------------------------------
    for i in range(2*dim):
        boundary = 'near((x[i]-l), 0, tol)'
        boundary = df.CompiledSubDomain(boundary,
                                        i = i%dim,
                                        l = L[i],
                                        tol = 1E-8)
        boundary.mark(facet_f, (n_components+i))

    for o in objects:
        o.mark_facets()

    return facet_f, objects

class Object:
    """
    This class is used to keep track of an object with a given id and
    geometry. It finds whether a particle is inside the object or not.
    If the particle is inside the object it returns 'True' and adds the charge
    of the particle to the accumulated charge of the object.

    Object is also capable of marking the facets of the surface boundary of the
    object, as well as the adjecent cells to the object.

    It returns the dofs, the adjecent cells and the vertices of the boundary
    nodes of the objects' surface.
    """
    def __init__(self, geometry, id, V, mesh, facet_f, cell_f, v2d):
        self.geometry = geometry
        self.id = id
        self.V = V
        self.mesh = mesh
        self.facet_f = facet_f
        self.cell_f = cell_f
        self.v2d = v2d
        self.charge = 0.0
        self.q_rho = 0.0
        self.phi = df.Constant(0.0)

    def inside(self, x):
        return self.geometry(x)

    def add_charge(self, q):
        self.charge += q

    def set_q_rho(self, q):
        self.q_rho = np.sum(q.vector()[self.dofs()])

    def set_phi(self, phi):
        self.phi.assign(phi)

    def bc(self):
        return df.DirichletBC(self.V, self.phi, self.facet_f, self.id)

    def dofs(self):
        """
        Returns the dofs of the object
        """
        itr_facet = df.SubsetIterator(self.facet_f, self.id)
        object_dofs = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_dofs.add(self.v2d[v.index()])
        return list(object_dofs)

    def vertices(self):
        """
        Returns the vertices of the surface of the object
        """
        itr_facet = df.SubsetIterator(self.facet_f, self.id)
        object_vertices = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_vertices.add(v.index())
        return list(object_vertices)

    def cells(self):
        """
        Returns the cells adjecent to the surface of the object
        """
        D = self.mesh.topology().dim()
        self.mesh.init(D-1,D) # Build connectivity between facets and cells
        itr_facet = df.SubsetIterator(self.facet_f, self.id)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        return object_adjacent_cells

    def mark_facets(self):
        """
        Marks the surface facets of the object
        """
        object_boundary = df.AutoSubDomain(lambda x: self.inside(x))
        object_boundary.mark(self.facet_f, self.id)

    def mark_cells(self):
        """
        Marks the cells adjecent to the object
        """
        cells = self.cells()
        for c in cells:
            self.cell_f[int(c)] = self.id
