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

#import dolfin as df
import dolfin as df
import numpy as np
from mpi4py import MPI as pyMPI
from collections import defaultdict
from itertools import count

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle(object):
    __slots__ = ['x', 'v', 'q', 'm']
    def __init__(self, x, v, q, m):
        assert(q!=0 and m!=0)
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

class Population(list):

    def __init__(self, mesh):
        self.mesh = mesh

        # Allocate a list of particles for each cell
        for cell in df.cells(self.mesh):
            self.append(list())

        # Create a list of sets of neighbors for each cell
        self.t_dim = self.mesh.topology().dim()
        self.g_dim = self.mesh.geometry().dim()

        self.Ld = []
        for j in range(self.g_dim):
            self.Ld.append(self.mesh.coordinates()[:,j].max())

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
        for old_cell_id, new_data in new_cell_map.iteritems():
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
            particles_outside_domain = set()
            for i in range(len(list_of_escaped_particles)):
                particle = list_of_escaped_particles[i]
                x = particle.x
                q = particle.q

                if open_bnd:
                    for (j,l) in enumerate(self.Ld):
                        if x[j] < 0.0 or x[j] > l:
                            particles_outside_domain.update([i])
                            break

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
