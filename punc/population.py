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
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
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
from itertools import count
from punc.injector import locate, ORS, ShiftedMaxwellian

class Particle(object):
    __slots__ = ('x', 'v', 'q', 'm')
    def __init__(self, x, v, q, m):
        assert q!=0 and m!=0
        self.x = np.array(x) # Position vector
        self.v = np.array(v) # Velocity vector
        self.q = q           # Charge
        self.m = m           # Mass

class Species(object):
    __slots__ = ('q', 'm', 'n', 'vth', 'vd', 'num', 'pdf', 'pdf_max', 
                 'vdf_type', 'vdf', 'num_particles', 'flux')

    def __init__(self, q, m, n, vth, vd, num, pdf, pdf_max, ext_bnd, maxwellian):
        # Parameters apply for normalized simulation particles
        self.q       = q                     # Charge
        self.m       = m                     # Mass
        self.n       = n                     # Density
        self.num     = num                   # Initial number of particles
        self.vth     = vth                   # Thermal velocity
        self.vd      = vd                    # Drift velocity
        self.pdf     = pdf                   # Position distribution function
        self.pdf_max = pdf_max               # Maximum value of pdf
        if maxwellian:
            self.set_vdf_type(ShiftedMaxwellian(vth, vd), ext_bnd)

    def set_vdf_type(self, vdf_type, ext_bnd):
        self.vdf_type = vdf_type
        self.vdf = vdf_type.get_vdf()
        self.num_particles = vdf_type.get_num_particles(ext_bnd)
        self.flux = ORS(vdf_type, ext_bnd)

class SpeciesList(list):
    """
    Whereas the Population class is unaware of any species (it only keeps track
    of individual particles), this is a list of all the species and their
    parameters, for instance their thermal velocity. It is useful for functions
    generating new particles, e.g. load_particles() and inject_particles().

    It also keeps track of the characteristic length (X), time (T), charge (Q),
    mass (M) and the number of dimensions of the simulation (D) which may help
    dimensionalize the output through simple dimensional analysis.

    Example:
        Let's say you have the normalized current into an object in a 2D
        simulation. The SI unit of this is A/m, or equivalently C/(m*s). Then
        this normalized current must be multiplied by Q/(X*T) to get a unit of
        A/m.
    """
    def __init__(self, mesh, X, T=None):
        """
        'mesh' is DOLFIN mesh, while 'ext_bnd' is ExteriorBoundary object.
        'X' is characteristic length while 'T' is characteristic time (SI units).
        If T==None it will be set to the reciprocal of the plasma angular
        frequency of the first species added to the list.
        """

        elementary_charge = constants.value('elementary charge')

        self.volume = df.assemble(1*df.dx(mesh))
        self.num_cells = mesh.num_cells()

        self.X = X                     # Characteristic length
        self.T = T                     # Characteristic time
        self.Q = elementary_charge     # Characteristic charge
        self.M = None                  # Characteristic mass
        self.D = mesh.geometry().dim() # Number of dimensions

    def append_raw(self, q, m, n, vth=None, vd=None, npc=16, ext_bnd=None, 
                   num=None, pdf=lambda x: 1, pdf_max=1, maxwellian=True):
        """
        Like append() but without normalization. Can be use to run simulations
        in non-normalized SI units, use this function instead, and set eps_0
        in the Poisson solver equal to its true value.
        """

        # Simulation particle scaling

        if num==None: num = npc*self.num_cells
        w = (n/num)*self.volume

        q *= w
        m *= w
        n /= w # Equals num/self.volume

        list.append(self, Species(q, m, n, vth, vd, num, pdf, pdf_max, ext_bnd, maxwellian))

    def append(self, q, m, n, vth=None, vd=None, npc=16, ext_bnd=None, num=None,
               pdf=lambda x:1, pdf_max=1, maxwellian=True):
        """
        Appends a species with given parameters:

            q   - Charge
            m   - Mass
            n   - Plasma density
            vth - Thermal speed (scalar)
            vd  - Drift velocity (vector, default: 0)
            npc - Initial number of particles per cell (default: 16)
            num - Initial number of particles in total (overrides npc if set)

        All paremeters are in SI units. The parameters will be appropriately
        normalized and scaled to simulation particles.
        """

        epsilon_0 = constants.value('electric constant')

        # Set missing normalization scales

        if self.T==None:
            wp = np.sqrt(n*q**2/(epsilon_0*m))
            self.T = wp**(-1)

        if self.M==None:
            self.M = (self.T*self.Q)**2 / (epsilon_0 * self.X**self.D)

        # Normalize input

        q   /= self.Q
        m   /= self.M
        n   *= self.X**self.D

        if vth is not None:
            if vth == 0: vth = np.finfo(float).eps
            vth /= (self.X/self.T)
        if vd is None:
            vd = np.zeros(self.D)
        else:
            vd  = [vd_i/(self.X/self.T) for vd_i in vd]

        # Add to list
        self.append_raw(q, m, n, vth, vd, npc, ext_bnd, num, pdf, pdf_max, maxwellian)

class Population(list):
    """
    Represents a population of particles. self[i] is a list of all Particle
    objects belonging to cell i in the DOLFIN mesh. Note that particles, when
    moved, do not automatically appear in the correct cell. Instead relocate()
    must be invoked to relocate the particles.
    """

    def __init__(self, mesh, bnd):
        self.mesh = mesh

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


                # take normal from cell rather than from facet to make sure it
                # is outwards-pointing
                normal = cell.normal(facet_number).array()[:self.g_dim]

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
        nDims = self.g_dim
        with open(fname, 'r') as datafile:
            for line in datafile:
                nums = np.array([float(a) for a in line.split('\t')])
                x = nums[0:nDims]
                v = nums[nDims:2*nDims]
                q = nums[2*nDims]
                m = nums[2*nDims+1  ]
                self.add_particles([x],v,q,m)
