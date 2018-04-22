# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of ConstantBC.
#
# ConstantBC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# ConstantBC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# ConstantBC.  If not, see <http://www.gnu.org/licenses/>.

import dolfin as df
import numpy as np

class ConstantBC(df.DirichletBC):
    """
    Enforces a constant but unknown boundary. The (single) unknown value at the
    boundary must be determined from the variational formaulation, typically by
    means of a Lagrange multiplier. See examples in the demos.

    Tested for 1st and 2nd order Lagrange finite elements but should in principe
    work for higher orders as well.

    This class is in most ways similar to Dolfin's own DirichletBC class, which
    it inherits. Its constructor is similar to DirichletBC's except that the
    second argument (the value on the boundary) must be omitted, e.g.:

        bc = ConstantBC(V, sub_domain)
        bc = ConstantBC(V, sub_domain, method)
        bc = ConstantBC(V, sub_domains, sub_domain)
        bc = ConstantBC(V, sub_domains, sub_domain, method)

    where sub_domain, sub_domains and method has the same meanings as for
    DirichletBC.
    """

    def __init__(self, *args, **kwargs):

        # Adds the missing argument (the value on the boundary) before calling
        # the parent constructor. The value must be zero to set the
        # corresponding elements in the load vector to zero.

        args = list(args)
        args.insert(1, df.Constant(0.0))
        monitor = False

        df.DirichletBC.__init__(self, *args, **kwargs)

    def monitor(self, monitor):
        self.monitor = monitor

    def apply(self, *args):

        for A in args:

            if isinstance(A, df.GenericVector):
                # Applying to load vectory.
                # Set all elements to zero but leave the first.

                ind = self.get_boundary_values().keys()
                first_ind = list(ind)[0]
                first_element = A[first_ind][0]

                df.DirichletBC.apply(self, A)

                A[first_ind] = first_element

            else:
                # Applying to stiffness matrix.
                # Leave the first row on the boundary node, but change the
                # remaining to be the average of it's neighbors also on the
                # boundary.

                ind = self.get_boundary_values().keys()

                length = len(list(ind))-2

                for it, i in enumerate(list(ind)[1:]):

                    if self.monitor:
                        print("ConstantBC iteration", it, "of", length)

                    neighbors = A.getrow(i)[0]
                    A.zero(np.array([i], dtype=np.intc))

                    surface_neighbors = np.array([n for n in neighbors if n in ind])
                    values = -np.ones(surface_neighbors.shape)

                    self_index = np.where(surface_neighbors==i)[0][0]
                    num_of_neighbors = len(surface_neighbors)-1
                    values[self_index] = num_of_neighbors

                    A.setrow(i, surface_neighbors, values)

                    A.apply('insert')
