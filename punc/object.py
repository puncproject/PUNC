from __future__ import print_function
import dolfin as df
import numpy as np

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
    def __init__(self, geometry, id, mesh, facet_f, cell_f, v2d):
        self.geometry = geometry
        self.id = id
        self.mesh = mesh
        self.facet_f = facet_f
        self.cell_f = cell_f
        self.v2d = v2d
        self.charge = 0.0

    def inside(self, p, q = 0.0):
        if self.geometry(p):
            self.charge += q
            return True
        else:
            return False

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
