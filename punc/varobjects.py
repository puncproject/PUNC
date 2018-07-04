# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of PUNC.
#
# PUNC is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PUNC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC. If not, see <http://www.gnu.org/licenses/>.

import dolfin as df
import numpy as np

# NB: Leave as-is! Currently used in demo.
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

            # for i in ind[1:]: # Old method stopped working
            for i in list(ind)[1:]:
                neighbors = A.getrow(i)[0]
                A.zero(np.array([i], dtype=np.intc))

                surface_neighbors = np.array([n for n in neighbors if n in ind])
                values = -np.ones(surface_neighbors.shape)

                self_index = np.where(surface_neighbors==i)[0][0]
                num_of_neighbors = len(surface_neighbors)-1
                values[self_index] = num_of_neighbors

                A.setrow(i, surface_neighbors, values)

                A.apply('insert')

class VObject(df.DirichletBC):

    def __init__(self, V, bnd, bnd_id, floating=True, potential=0, charge=0):
        if floating and potential != 0:
            potential=0
            print("Potential cannot be set when using floating")
        self.charge = charge
        self._potential = df.Constant(potential)
        self.floating = df.Constant(floating)
        self.V = V
        self.id = bnd_id
        df.DirichletBC.__init__(self, V, df.Constant(potential), bnd, bnd_id, "topological")

    def set_potential(self, potential):
        if self.floating and potential != 0:
            potential=0
            print("Potential cannot be set when using floating")
        self._potential = potential
        self.set_value(df.Constant(self._potential))

    def apply(self, A):
        if not self.floating or isinstance(A, df.GenericVector):
            df.DirichletBC.apply(self, A)

        else:
            print('APPLYING CONSTANT BOUNDARY CONSTRAINT')
            ind = self.get_boundary_values().keys()

            for i in list(ind)[1:]:
                neighbors = A.getrow(i)[0]
                A.zero(np.array([i], dtype=np.intc))

                surface_neighbors = np.array([n for n in neighbors if n in ind])
                values = -np.ones(surface_neighbors.shape)

                self_index = np.where(surface_neighbors==i)[0][0]
                num_of_neighbors = len(surface_neighbors)-1
                values[self_index] = num_of_neighbors

                A.setrow(i, surface_neighbors, values)

                A.apply('insert')
