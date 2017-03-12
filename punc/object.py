from __future__ import print_function
import dolfin as df
import numpy as np

class Object:
    """This class is used to find whether a particle is inside an object or not.
    If the particle is inside the object it returns 'True' and adds the charge
    of the particle to the accumulated charge of the object.
    """
    def __init__(self, geometry, index):
        self.geometry = geometry
        self.index = index
        self.charge = 0.0

    def inside(self, p, q = 0.0):
        if self.geometry(p):
            self.charge += q
            return True
        else:
            return False

    def dofs(self, V, facet_f):
        """This function returns the dofs of the object

        Args:
            V           : FunctionSpace(mesh, "CG", 1)
            facet_f     : contains the facets of each surface component

        returns:
            A list of all object dofs
        """
        v2d = df.vertex_to_dof_map(V)
        itr_facet = df.SubsetIterator(facet_f, self.index)
        object_dofs = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_dofs.add(v2d[v.index()])
        return list(object_dofs)

    def vertices(self, facet_f):
        """This function returns the vertices of the object

        Args:
            facet_f : contains the facets of each surface component

        returns:
            A list of all object vertices
        """
        itr_facet = df.SubsetIterator(facet_f, self.index)
        object_vertices = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_vertices.add(v.index())
        return list(object_vertices)

    def cells(self, mesh, facet_f):
        """This function returns the cells adjecent to the object

        Args:
            mesh     : The mesh
            facet_f  : contains the facets of each surface component

        returns:
            A list of all cells adjecent to the surface component
        """
        D = mesh.topology().dim()
        mesh.init(D-1,D) # Build connectivity between facets and cells
        itr_facet = df.SubsetIterator(facet_f, self.index)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        return object_adjacent_cells
