from __future__ import print_function
import dolfin as df
import numpy as np

class Marker:
    """ This class marks the exterior boundaries, as well as all the objects
    inside the simulation domain.

    Args:
        mesh        : mesh of the domain
        domain_info : complete infor about the exterior domain
        objects     : a list containing all the objects inside the domain
    """
    def __init__(self, mesh, domain_info, objects):
        self.mesh = mesh
        self.domain_info = domain_info
        self.objects = objects
        self.n_components = len(self.objects)

    def markers(self):
        """This function marks the exterior boundaries, as well as all the
        objects inside the simulation domain.

        returns:
            facet_f: marked facets of all exterior and object boundaries.
        """
        facet_f = df.FacetFunction('size_t', self.mesh)
        facet_f.set_all(self.n_components+len(self.domain_info))
        df.DomainBoundary().mark(facet_f, 0)
        if self.n_components > 1:
            for i, o in enumerate(self.objects[1:]):
                object_boundary = df.AutoSubDomain(lambda x: o.inside(x))
                object_boundary.mark(facet_f, (i+1))
        facet_f = self.exterior_boundaries(facet_f)
        return facet_f

    def exterior_boundaries(self, facet_f):
        """This function marks the exteror boundaries of the simulation domain
        Args:
            facet_f     : contains the facets of each surface component

        returns:
            facet_f: marked facets of the exterior boundaries of the domain
        """
        d = len(self.domain_info)/2
        for i in range(2*d):
            boundary = 'near((x[i]-l), 0, tol)'
            boundary = df.CompiledSubDomain(boundary,
                                            i = i%d,
                                            l = self.domain_info[i],
                                            tol = 1E-8)
            boundary.mark(facet_f, (self.n_components+i))
        return facet_f

    def object_adjecent_cells(self, cells):
        """This function marks the cells adjecent to all objects

        Args:
            cells: a list of indices of the adjecent cells for each object
                   (obtained from the Object class).

        returns:
            Marks the cells adjecent to objects
        """
        adjecent_cells = df.CellFunction('size_t', self.mesh)
        adjecent_cells.set_all(self.n_components)

        for i in range(self.n_components):
            for c in cells[i]:
                adjecent_cells[int(c)] = i
        return adjecent_cells
