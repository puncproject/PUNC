# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
#
# Loosely based on fenicstools/LagrangianParticles by Mikeal Mortensen and
# Miroslav Kuchta. Released under same license.

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from mpi4py import MPI as pyMPI
from collections import defaultdict
from itertools import count
from punc.poisson import get_mesh_size
from punc.injector import create_mesh_pdf, SRS, Maxwellian


comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle(object):
    __slots__ = ('x', 'v', 'q', 'm')
    def __init__(self, x, v, q, m):
        assert q!=0 and m!=0
        self.x = np.array(x)    # Position vector
        self.v = np.array(v)    # Velocity vector
        self.q = q              # Charge
        self.m = m              # Mass

    def send(self, dest):
        comm.Send(self.x, dest=dest)
        comm.Send(self.v, dest=dest)
        comm.Send(self.q, dest=dest)
        comm.Send(self.m, dest=dest)

    def recv(self, source):
        comm.Recv(self.x, source=source)
        comm.Recv(self.v, source=source)
        comm.Recv(self.q, source=source)
        comm.Recv(self.m, source=source)

class Specie(object):
    """
    A specie with q elementary charges and m electron masses is specified as
    follows:

        s = Specie((q,m))

    Alternatively, electrons and protons may be specified by an 'electron' or
    'proton' string instead of a tuple:

        s = Specie('electron')

    The following keyword arguments are accepted to change default behavior:

        v_drift
            Drift velocity of specie. Default: 0.

        v_thermal
            Thermal velocity of specie. Default: 0.
            Do not use along with temperature.

        temperature
            Temperature of specie. Default: 0.
            Do not use along with v_thermal.

        num_per_cell
            Number of particles per cell. Default: 16.

        num_total
            Number of particles in total.
            Overrides num_per_cell if specified.

    E.g. to specify electrons with thermal and drift velocities:

        s = Specie('electron', v_thermal=1, v_drift=[1,0])

    Note that the species have to be normalized before being useful. Species
    are typically put in a Species list and normalized before being used. See
    Species.
    """

    def __init__(self, specie, **kwargs):

        # Will be set during normalization
        self.charge = None
        self.mass = None
        self.v_thermal = None
        self.v_drift = None

        self.v_thermal_raw = 0
        self.temperature_raw = None
        self.v_drift_raw = 0

        self.num_total = None
        self.num_per_cell = 16

        if specie == 'electron':
            self.charge_raw = -1
            self.mass_raw = 1

        elif specie == 'proton':
            self.charge_raw = 1
            self.mass_raw = 1836.15267389

        else:
            assert isinstance(specie,tuple) and len(specie)==2 ,\
                "specie must be a valid keyword or a (charge,mass)-tuple"

            self.charge_raw = specie[0]
            self.mass_raw = specie[1]

        if 'num_per_cell' in kwargs:
            self.num_per_cell = kwargs['num_per_cell']

        if 'num_total' in kwargs:
            self.num_total = kwargs['num_total']

        if 'v_thermal' in kwargs:
            self.v_thermal_raw = kwargs['v_thermal']

        if 'v_drift' in kwargs:
            self.v_drift_raw = kwargs['v_drift']

        if 'temperature' in kwargs:
            self.temperature_raw = kwargs['temperature']

class Species(list):
    """
    Just a normal list of Specie objects except that the method append_specie()
    may be used to append species to the list and normalize them.
    append_specie() takes the same argumets as the Specie() constructor.

    Two normalization schemes are implemented as can be chosen using the
    'normalization' parameter in the constructor:

        'plasma params' (default):
            The zeroth specie in the list (i.e. the first appended one) is
            normalized to have an angular plasma frequency of one and a thermal
            velocity of 1 (and hence also a Debye length of one). If the specie
            is cold the thermal velocity is 0 and the Debye length does not act
            as a characteristic length scale in the simulations.

        'none':
            The specified charge, mass, drift and thermal velocities are used
            as specified without further normalization.

    E.g. to create isothermal electrons and ions normalized such that the
    electron parameters are all one:

        species = Species(mesh)
        species.append_specie('electron', temperature=1) # reference
        species.append_specie('proton'  , temperature=1)

    """

    def __init__(self, mesh, normalization='plasma params'):
        self.volume = df.assemble(1*df.dx(mesh))
        self.num_cells = mesh.num_cells()

        assert normalization in ('plasma params', 'none')

        if normalization == 'plasma params':
            self.normalize = self.normalize_plasma_params

        if normalization == 'none':
            self.normalize = self.normalize_none

    def append_specie(self, specie, **kwargs):
        self.append(Specie(specie, **kwargs))
        self.normalize(self[-1])

    def normalize_none(self, s):
        if s.num_total == None:
            s.num_total = s.num_per_cell * self.num_cells

        s.charge = s.charge_raw
        s.mass = s.mass_raw
        s.v_thermal = s.v_thermal_raw
        s.v_drift = s.v_drift_raw
        self.weight = 1

    def normalize_plasma_params(self, s):
        if s.num_total == None:
            s.num_total = s.num_per_cell * self.num_cells
            print("num_tot: ", s.num_total)

        ref = self[0]
        w_pe = 1
        self.weight = (w_pe**2) \
               * (self.volume/ref.num_total) \
               * (ref.mass_raw/ref.charge_raw**2)

        s.charge = self.weight*s.charge_raw
        s.mass = self.weight*s.mass_raw

        if ref.temperature_raw != None:
            assert s.temperature_raw != None, \
                "Specify temperature for all or none species"

            ref.v_thermal = 1
            for s in self:
                s.v_thermal = ref.v_thermal*np.sqrt( \
                    (s.temperature_raw/ref.temperature_raw) * \
                    (ref.mass_raw/s.mass_raw) )
        elif s.v_thermal_raw == 0:
            s.v_thermal = 0
        else:
            s.v_thermal = s.v_thermal_raw/ref.v_thermal_raw

        if (isinstance(s.v_drift_raw, np.ndarray) and \
           all(i == 0 for i in s.v_drift_raw) ):
            s.v_drift = np.zeros((s.v_drift_raw.shape))
        elif isinstance(s.v_drift_raw, (float,int)) and s.v_drift_raw==0:
            s.v_drift = 0
        else:
            s.v_drift = s.v_drift_raw/ref.v_thermal_raw

class Population(list):
    """
    Represents a population of particles. self[i] is a list of all Particle
    objects belonging to cell i in the DOLFIN mesh. Note that particles, when
    moved, do not automatically appear in the correct cell. Instead relocate()
    must be invoked to relocate the particles.
    """

    def __init__(self, mesh, periodic, normalization='plasma params'):
        self.mesh = mesh
        self.Ld = get_mesh_size(mesh)
        self.periodic = periodic
        # --------Suggestion---------
        self.vel = []
        self.plasma_density = []
        self.volume = df.assemble(1*df.dx(mesh))
        # -------------------------------

        # Species
        self.species = Species(mesh, normalization)

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

        # Allocate some MPI stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = list(range(self.num_processes))
        self.other_processes = list(range(self.num_processes))
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = np.zeros(1, dtype='I')
        self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
        # Dummy particle for receiving/sending at [0, 0, ...]
        v_zero = np.zeros(self.g_dim)
        self.particle0 = Particle(v_zero,v_zero,1,1)

    def init_new_specie(self, specie, **kwargs):
        """
        To initialize a new specie within a population use this function, e.g.
        to uniformly populate the domain with 16 (default) cold electrons and
        protons per cell:

            pop = Population(mesh)
            pop.init_new_specie('electron')
            pop.init_new_specie('proton')

        Here, the normalization is such that the electron plasma frequency and
        Debye lengths are set to one. The electron is used as a reference
        because that specie is initialized first.

        All species is represented as a Species object internally in the
        population and consequentially, the init_new_specie() method takes the
        same arguments as the append_specie() method in the Species class. See
        that method for information of how to tweak specie properties.

        In addition, init_new_specie() takes two additional keywords:

            pdf:
                A probability density function of how to distribute particles.

            pdf_max:
                An upper bound for the values in the pdf.

        E.g. to initialize cold langmuir oscillations (where the initial
        electron density is sinusoidal) in the x-direction in a unit length
        domain:

            pop = Population(mesh)
            pdf = lambda x: 1+0.1*np.sin(2*np.pi*x[0])
            pop.init_new_specie('electron', pdf=pdf, pdf_max=1.1)
            pop.init_new_specie('proton')

        """

        self.species.append_specie(specie, **kwargs)

        if 'pdf' in kwargs:
            pdf = kwargs['pdf']
        else:
            pdf = lambda x: 1

        if pdf != None:

            pdf = create_mesh_pdf(pdf, self.mesh)

            if 'pdf_max' in kwargs:
                pdf_max = kwargs['pdf_max']
            else:
                pdf_max = 1

        m = self.species[-1].mass
        q = self.species[-1].charge
        v_thermal = self.species[-1].v_thermal
        v_drift = self.species[-1].v_drift
        num_total = self.species[-1].num_total

        # --------Suggestion---------
        rs = SRS(pdf, pdf_max=pdf_max, Ld=self.Ld)
        mv = Maxwellian(v_thermal, v_drift, self.periodic)
        self.vel.append(mv)
        self.plasma_density.append(num_total/self.volume)

        xs = rs.sample(num_total)
        vs = mv.load(num_total)
        #---------------------------
        # xs = random_points(pdf, self.Ld, num_total, pdf_max)
        # vs = maxwellian(v_drift, v_thermal, xs.shape)
        self.add_particles(xs,vs,q,m)

    def add_particles_of_specie(self, specie, xs, vs=None):
        q = self.species[specie].charge
        m = self.species[specie].mass
        self.add_particles(xs, vs, q, m)

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
            cell = self.locate(x)
            if not (cell == -1 or cell == __UINT32_MAX__):
                my_found[i] = True
                self[cell].append(Particle(x, v, q, m))

        # All particles must be found on some process
        comm.Reduce(my_found, all_found, root=0)

        if self.myrank == 0:
            n_missing = len(np.where(all_found == 0)[0])
            assert n_missing==0,'%d particles are not located in mesh'%n_missing

    def relocate(self, objects = [], open_bnd = False):
        """
        Relocate particles on cells and processors
        map such that map[old_cell] = [(new_cell, particle_id), ...]
        i.e. new destination of particles formerly in old_cell
        """
        new_cell_map = defaultdict(list)
        for df_cell in df.cells(self.mesh):
            c_index = df_cell.index()
            cell = self[c_index]
            for i, particle in enumerate(cell):
                point = df.Point(*particle.x)
                # Search only if particle moved outside original cell
                if not df_cell.contains(point):
                    found = False
                    # Check neighbor cells
                    for neighbor in self.neighbors[df_cell.index()]:
                        if df.Cell(self.mesh,neighbor).contains(point):
                            new_cell_id = neighbor
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        new_cell_id = self.locate(particle)
                    # Record to map
                    new_cell_map[df_cell.index()].append((new_cell_id, i))

        # Rebuild locally the particles that end up on the process. Some
        # have cell_id == -1, i.e. they are on other process
        list_of_escaped_particles = []
        for old_cell_id, new_data in new_cell_map.items():
            # We iterate in reverse becasue normal order would remove some
            # particle this shifts the whole list!
            for (new_cell_id, i) in sorted(new_data,
                                           key=lambda t: t[1],
                                           reverse=True):
#               particle = p_map.pop(old_cell_id, i)

                particle = self[old_cell_id][i]
                # Delete particle in old cell, fill gap by last element
                if not i==len(self[old_cell_id])-1:
                    self[old_cell_id][i] = self[old_cell_id].pop()
                else:
                    self[old_cell_id].pop()

                if new_cell_id == -1 or new_cell_id == __UINT32_MAX__ :
                    list_of_escaped_particles.append(particle)
                else:
#                   p_map += self.mesh, new_cell_id, particle
                    self[new_cell_id].append(particle)

        # Create a list of how many particles escapes from each processor
        self.my_escaped_particles[0] = len(list_of_escaped_particles)
        # Make all processes aware of the number of escapees
        comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)

        # Send particles to root
        if self.myrank != 0:
            for particle in list_of_escaped_particles:
                particle.send(0)

        # Receive the particles escaping from other processors
        if self.myrank == 0:
            for proc in self.other_processes:
                for i in range(self.tot_escaped_particles[proc]):
                    self.particle0.recv(proc)
                    list_of_escaped_particles.append(copy.deepcopy(self.particle0))

        """
        The escaped particles are handled in the following way:
        For each particle in the list, if it's outside the simulation domain,
        it's removed from the population. Otherwise, the particle must be inside
        one of the objects. For each object if the particle is inside or at the
        boundary of the object, the electric charge of the particle is added to
        the accumulated charge of the object, and then it is removed from the
        simulation.
        """
        if ((len(objects) != 0) or open_bnd):
            # print("WTF")
            particles_outside_domain = set()
            for i in range(len(list_of_escaped_particles)):
                # print("missing: ", len(list_of_escaped_particles))
                particle = list_of_escaped_particles[i]
                x = particle.x
                q = particle.q

                for j in range(self.g_dim):
                    if self.periodic[j]:
                        x[j] %= self.Ld[j]
                    elif x[j] < 0.0 or x[j] > self.Ld[j]:
                        particles_outside_domain.update([i])
                        break

                # if open_bnd:
                #     for (j,l) in enumerate(self.Ld):
                #         if x[j] < 0.0 or x[j] > l:
                #             particles_outside_domain.update([i])
                #             break

                for o in objects:
                    if o.inside(x, True):
                        o.add_charge(q)
                        particles_outside_domain.update([i])
                        break

            particles_outside_domain = list(particles_outside_domain)

            # Remove particles inside the object
            for i in reversed(particles_outside_domain):
                p = list_of_escaped_particles[i]
                list_of_escaped_particles.remove(p)

        # Put all travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        self.add_particles(travelling_particles)


    def total_number_of_particles(self):
        'Return number of particles in total and on process.'
        num_p = sum([len(x) for x in self])
        tot_p = comm.allreduce(num_p)
        return (tot_p, num_p)

    def locate(self, particle):
        'Find mesh cell that contains particle.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
                    # Convert particle to point
            point = df.Point(*particle.x)
        else:
            point = df.Point(*particle)
        return self.tree.compute_first_entity_collision(point)

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
