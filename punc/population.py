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
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC.  If not, see <http://www.gnu.org/licenses/>.
#
# Loosely based on fenicstools/LagrangianParticles by Mikeal Mortensen and
# Miroslav Kuchta.

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import scipy.constants as constants
from mpi4py import MPI as pyMPI
from collections import defaultdict
from itertools import count
from punc.poisson import get_mesh_size
from punc.injector import create_mesh_pdf, Flux, maxwellian, random_domain_points, locate

comm = pyMPI.COMM_WORLD

class Particle(object):
    __slots__ = ('x', 'v', 'q', 'm')
    def __init__(self, x, v, q, m):
        assert q!=0 and m!=0
        self.x = np.array(x) # Position vector
        self.v = np.array(v) # Velocity vector
        self.q = q           # Charge
        self.m = m           # Mass

class Species(object):
    __slots__ = ('q', 'm', 'n', 'vth', 'vd', 'num', 'flux')
    def __init__(self, q, m, n, vth, vd, num, ext_bnd):
        # Parameters apply for normalized simulation particles
        self.q        = q                      # Charge
        self.m        = m                      # Mass
        self.n        = n                      # Density
        self.vth      = vth                    # Thermal velocity
        self.vd       = vd                     # Drift velocity
        self.num      = num                    # Initial number of particles
        self.flux     = Flux(vth, vd, ext_bnd) # Flux object

class SpeciesList(list):
    def __init__(self, mesh, ext_bnd, X, T=None):

        elementary_charge = constants.value('elementary charge')

        self.volume = df.assemble(1*df.dx(mesh))
        self.num_cells = mesh.num_cells()
        self.ext_bnd = ext_bnd

        self.X = X                     # Characteristic length
        self.T = T                     # Characteristic time
        self.Q = elementary_charge     # Characteristic charge
        self.M = None                  # Characteristic mass
        self.D = mesh.geometry().dim() # Number of dimensions

    def append(self, species, n, vth, vd=0.0, num=16, **args):
        
        epsilon_0         = constants.value('electric constant')
        elementary_charge = constants.value('elementary charge')
        electron_mass     = constants.value('electron mass')
        proton_mass       = constants.value('proton mass')
        amu               = constants.value('atomic mass constant')

        # Read in mass and charge and give them SI units

        if   species == 'electron': q, m = -1.0, 1.0
        elif species == 'positron': q, m =  1.0, 1.0
        elif species == 'proton':   q, m =  1.0, proton_mass/electron_mass
        else:
            assert isinstance(species,tuple) and len(species)==2 ,\
                "species must be a valid keyword or a (charge,mass)-tuple"

            q, m = species
            if 'amu' in args: m *= amu/electron_mass

        q *= elementary_charge # [C]
        m *= electron_mass     # [kg]
        
        # Set normalization parameters if not done (all parameters are now SI)

        if self.T==None:
            wp = np.sqrt(n*q**2/(epsilon_0*m))
            self.T = wp**(-1)

        if self.M==None:
            self.M = (self.T*self.Q)**2 / (epsilon_0 * self.X**self.D)

        # Normalize parameters

        q   /= self.Q
        m   /= self.M
        n   *= self.X**self.D
        vth /= (self.X/self.T)
        vd  /= (self.X/self.T)

        # Simulation particle scaling

        if 'num total' not in args: num *= self.num_cells
        w = (n/num)*self.volume
        
        q *= w
        m *= w
        n /= w # This now equals num/self.volume

        # Add to list
        list.append(self, Species(q, m, n, vth, vd, num, self.ext_bnd))

class Population(list):
    """
    Represents a population of particles. self[i] is a list of all Particle
    objects belonging to cell i in the DOLFIN mesh. Note that particles, when
    moved, do not automatically appear in the correct cell. Instead relocate()
    must be invoked to relocate the particles.
    """

    def __init__(self, mesh, bnd, periodic=None):
        self.mesh = mesh
        self.Ld = get_mesh_size(mesh)
        self.periodic = periodic

        # Allocate a list of particles for each cell
        for cell in df.cells(self.mesh):
            self.append(list())

        # Create a list of sets of neighbors for each cell
        self.t_dim = self.mesh.topology().dim()
        self.g_dim = self.mesh.geometry().dim()

        self.mesh.init(0, self.t_dim)
        self.tree = self.mesh.bounding_box_tree()
        self.neighbors = list()
        for cell in df.cells(self.mesh):
            neigh = sum([vertex.entities(self.t_dim).tolist() for vertex in df.vertices(cell)], [])
            neigh = set(neigh) - set([cell.index()])
            self.neighbors.append(neigh)

        self.init_localizer(bnd)

    def init_localizer(self, bnd):
        # self.facet_adjacents[cell_id][facet_number] is the id of the adjacent cell
        # self.facet_normals[cell_id][facet_number] is the normal vector to a facet
        # self.facet_mids[cell_id][facet_number] is the midpoint on a facet
        # facet_number is a number from 0 to t_dim
        # TBD: Now all facets are stored redundantly (for each cell)
        # Storage could be reduced, but would the performance hit be significant?

        self.mesh.init(self.t_dim-1, self.t_dim)
        self.facet_adjacents = []
        self.facet_normals = []
        self.facet_mids = []
        facets = list(df.facets(self.mesh))
        for cell in df.cells(self.mesh):
            facet_ids = cell.entities(self.t_dim-1)
            adjacents = []
            normals = []
            mids = []

            for facet_number, facet_id in enumerate(facet_ids):
                facet = facets[facet_id]

                adjacent = set(facet.entities(self.t_dim))-{cell.index()}
                adjacent = list(adjacent)
                if adjacent == []:
                    # Travelled out of bounds through the following boundary
                    # Minus indicates through boundary
                    adjacent = -int(bnd.array()[facet_id])

                else:
                    adjacent = int(adjacent[0])

                assert isinstance(adjacent,int)


                # take normal from cell rather than from facet to make sure it is outwards-pointing
                normal = [cell.normal(facet_number, i) for i in range(self.t_dim)]

                mid = facet.midpoint()
                mid = np.array([mid.x(), mid.y(), mid.z()])
                mid = mid[:self.t_dim]

                adjacents.append(adjacent)
                normals.append(normal)
                mids.append(mid)


            self.facet_adjacents.append(adjacents)
            self.facet_normals.append(normals)
            self.facet_mids.append(mids)

    def add_particles(self, xs, vs=None, qs=None, ms=None):
        """
        Adds particles to the population and locates them on their home
        processor. xs is a list/array of position vectors. vs, qs and ms may
        be lists/arrays of velocity vectors, charges, and masses,
        respectively, or they may be only a single velocity vector, mass
        and/or charge if all particles should have the same value.
        """

        if vs is None or qs is None or ms is None:
            assert isinstance(xs,list)
            if len(xs)==0:
                return
            assert isinstance(xs[0],Particle)
            ps = xs
            xs = [p.x for p in ps]
            vs = [p.v for p in ps]
            qs = [p.q for p in ps]
            ms = [p.m for p in ps]
            self.add_particles(xs, vs, qs, ms)
            return

        # Expand input to lists/arrays if necessary
        if len(np.array(vs).shape)==1: vs = np.tile(vs,(len(xs),1))
        if not isinstance(qs, (np.ndarray,list)): qs *= np.ones(len(xs))
        if not isinstance(ms, (np.ndarray,list)): ms *= np.ones(len(xs))

        # Keep track of which particles are located locally and globally
        my_found = np.zeros(len(xs), np.int)
        all_found = np.zeros(len(xs), np.int)

        for i, x, v, q, m in zip(count(), xs, vs, qs, ms):
            cell_id = self.locate(x)
            if cell_id >=0:
                self[cell_id].append(Particle(x, v, q, m))

    def locate(self, x):
        return locate(self.mesh, x)

    def relocate(self, p, cell_id):

        cell = df.Cell(self.mesh, cell_id)
        if cell.contains(df.Point(*p)):
            return cell_id
        else:
            x = p - np.array(self.facet_mids[cell_id])

            # The projection of x on each facet normal. Negative if behind facet.
            # If all negative particle is within cell
            proj = np.sum(x*self.facet_normals[cell_id], axis=1)
            projarg = np.argmax(proj)
            new_cell_id = self.facet_adjacents[cell_id][projarg]
            if new_cell_id>=0:
                return self.relocate(p, new_cell_id)
            else:
                return new_cell_id # crossed a boundary

    def update(self, objects = None):

        if objects == None: objects = []

        # TBD: Could possibly be placed elsewhere
        object_domains = [o.id for o in objects]
        object_ids = dict()
        for o,d in enumerate(object_domains):
            object_ids[d] = o

        # This times dt is an accurate measurement of collected current
        # collected_charge = np.zeros(len(objects))

        for cell_id, cell in enumerate(self):

            to_delete = []

            for particle_id, particle in enumerate(cell):

                new_cell_id = self.relocate(particle.x, cell_id)

                if new_cell_id != cell_id:

                    # Particle has moved out of cell.
                    # Mark it for deletion
                    to_delete.append(particle_id)

                    if new_cell_id < 0:
                        # Particle has crossed a boundary, either external
                        # or internal (into an object) and do not reappear
                        # in a new cell.

                        if -new_cell_id in object_ids:
                            # Particle entered object. Accumulate charge.
                            # collected_charge[object_ids[-new_cell_id]] += particle.q
                            obj = objects[object_ids[-new_cell_id]]
                            obj.charge += particle.q
                    else:
                        # Particle has moved to another cell
                        self[new_cell_id].append(particle)

            # Delete particles in reverse order to avoid altering the id
            # of particles yet to be deleted.
            for particle_id in reversed(to_delete):

                if particle_id==len(cell)-1:
                    # Particle is the last element
                    cell.pop()
                else:
                    # Delete by replacing it by the last element in the list.
                    # More efficient then shifting the whole list.
                    cell[particle_id] = cell.pop()

        # for o, c in zip(objects, collected_charge):
        #     o.charge += c

    def num_of_particles(self):
        'Return number of particles in total.'
        return sum([len(x) for x in self])

    def num_of_positives(self):
        return np.sum([np.sum([p.q>0 for p in c],dtype=int) for c in self])

    def num_of_negatives(self):
        return np.sum([np.sum([p.q<0 for p in c],dtype=int) for c in self])

    def num_of_conditioned(self, cond):
        '''
        Number of particles satisfying some condition.
        E.g. pop.num_of_conditions(lambda p: p.q<0)
        is equivalent to pop.num_of_negatives()
        '''
        return np.sum([np.sum([cond(p) for p in c],dtype=int) for c in self])

    def save_file(self, fname):
        with open(fname, 'w') as datafile:
            for cell in self:
                for particle in cell:
                    x = '\t'.join([str(x) for x in particle.x])
                    v = '\t'.join([str(v) for v in particle.v])
                    q = particle.q
                    m = particle.m
                    datafile.write("%s\t%s\t%s\t%s\n"%(x,v,q,m))

    def load_file(self, fname):
        nDims = len(self.Ld)
        with open(fname, 'r') as datafile:
            for line in datafile:
                nums = np.array([float(a) for a in line.split('\t')])
                x = nums[0:nDims]
                v = nums[nDims:2*nDims]
                q = nums[2*nDims]
                m = nums[2*nDims+1]
                self.add_particles([x],v,q,m)
