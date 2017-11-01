from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np

class FloatingBC(df.DirichletBC):

    def __init__(self, V, sub_domains, sub_domain, method="topological"):
        df.DirichletBC.__init__(self, V, df.Constant(0), sub_domains, sub_domain, method)

    def __init__(self, V, sub_domain, method="topological"):
        df.DirichletBC.__init__(self, V, df.Constant(0), sub_domain, method)

    def apply(self, A):

        if isinstance(A, df.GenericVector):
            df.DirichletBC.apply(self, A)

        else:
            ind = self.get_boundary_values().keys()

            for i in ind[1:]:
                neighbors = A.getrow(i)[0]
                A.zero(np.array([i], dtype=np.intc))

                surface_neighbors = np.array([n for n in neighbors if n in ind])
                values = -np.ones(surface_neighbors.shape)

                self_index = np.where(surface_neighbors==i)[0][0]
                num_of_neighbors = len(surface_neighbors)-1
                values[self_index] = num_of_neighbors

                A.setrow(i, surface_neighbors, values)

                A.apply('insert')
