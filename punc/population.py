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

class Particle:
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

    def __init__(self, mesh, object_type = None, object_info = []):
        self.mesh = mesh
        self.object_info = object_info
        self.object_type = object_type
        # Allocate a list of particles for each cell
        for cell in df.cells(self.mesh):
            self.append(list())

        # Create a list of sets of neighbors for each cell
        self.tDim = self.mesh.topology().dim()
        self.gDim = self.mesh.geometry().dim()
        self.mesh.init(0, self.tDim)
        self.tree = self.mesh.bounding_box_tree()
        self.neighbors = list()
        for cell in df.cells(self.mesh):
            neigh = sum([vertex.entities(self.tDim).tolist() for vertex in df.vertices(cell)], [])
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
        vZero = np.zeros(self.gDim)
        self.particle0 = Particle(vZero,vZero,1,1)

    def addParticles(self, xs, vs=None, qs=None, ms=None):
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
            addParticles(xs, vs, qs, ms)
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
            nMissing = len(np.where(all_found == 0)[0])
            assert nMissing==0,'%d particles are not located in mesh'%nMissing

    def relocate(self, q_object):
        """
        Relocate particles on cells and processors
        map such that map[old_cell] = [(new_cell, particle_id), ...]
        i.e. new destination of particles formerly in old_cell
        """
        new_cell_map = defaultdict(list)
        for dfCell in df.cells(self.mesh):
            cindex = dfCell.index()
            cell = self[cindex]
            for i, particle in enumerate(cell):
                point = df.Point(*particle.x)
                # Search only if particle moved outside original cell
                if not dfCell.contains(point):
                    found = False
                    # Check neighbor cells
                    for neighbor in self.neighbors[dfCell.index()]:
                        if df.Cell(self.mesh,neighbor).contains(point):
                            new_cell_id = neighbor
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        new_cell_id = self.locate(particle)
                    # Record to map
                    new_cell_map[dfCell.index()].append((new_cell_id, i))

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
        # What to do with escaped particles
        particles_inside_object = []
        particles_outside_domain = []

        for i in range(len(list_of_escaped_particles)):
            particle = list_of_escaped_particles[i]
            x = particle.x
            q = particle.q
            d = len(x)

            if self.object_type == 'spherical_object':
                if d == 2:
                    s0 = [self.object_info[0], self.object_info[1]]
                    r0 = self.object_info[2]
                if d == 3:
                    s0 = [self.object_info[0], self.object_info[1], self.object_info[2]]
                    r0 = self.object_info[3]
                if np.dot(x-s0, x-s0) < r0**2:
                    particles_inside_object.append(i)

        particles_outside_domain = set(particles_outside_domain)
        particles_outside_domain  = list(particles_outside_domain)

        particles_to_be_removed = []
        particles_to_be_removed.extend(particles_inside_object)
        particles_to_be_removed.extend(particles_outside_domain)
        particles_to_be_removed.sort()

        print("particles inside object: ", len(particles_inside_object))
        print("particles_outside_domain: ", len(particles_outside_domain))
        print("particles_to_be_removed: ", len(particles_to_be_removed))

        # print("list_of_escaped_particles: ", list_of_escaped_particles)
        if (not self.object_type is None):

            # Remove particles inside the object and accumulate the charge
            for i in reversed(particles_to_be_removed):
                p = list_of_escaped_particles[i]
                if i in particles_inside_object:
                    q_object[0] += p.q
                list_of_escaped_particles.remove(p)
        # Put all travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        self.addParticles(travelling_particles)
        return q_object

    def total_number_of_particles(self):
        'Return number of particles in total and on process.'
        num_p = self.particle_map.total_number_of_particles()
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

def stdSpecie(mesh, Ld, q, m, N, q0=-1.0, m0=1.0, wp0=1.0, count='per cell'):
    """
    Returns standard normalized particle parameters to use for a specie. The
    inputs q and m are the charge and mass of the specie in elementary charges
    and electron masses, respectively. N is the number of particles per cell
    unless count='total' in which case it is the total number of simulation
    particles in the domain. For instance to create 8 ions per cell (having
    mass of 1836 electrons and 1 positive elementary charge):

        q, m, N = stdSpecie(mesh, Ld, 1, 1836, 8)

    The returned values are the normalized charge q, mass m and total number of
    simulation particles N. The normalization is such that the angular electron
    plasma frequency will be 1.

    Alternatively, the reference charge q0 and mass m0 which should yield a
    angular plasma frequency of 1, or for that sake any other frequency wp0,
    could be set through the kwargs. For example, in this case the ions will
    have a plasma frequency of 0.1:

        q, m, N = stdSpecie(mesh, Ld, 1, 1836, 8, q0=1, m0=1836, wp0=0.1)
    """

    assert count in ['per cell','total']

    if count=='total':
        Np = N
    else:
        Np = N*mesh.num_cells()

    mul = (np.prod(Ld)/np.prod(Np))*(wp0**2)*m0/(q0**2)
    return q*mul, m*mul, Np

#==============================================================================
# PERHAPS STUFF BELOW THIS SHOULD GO INTO A SEPARATE MODULE? E.G. INIT. CONDS.?
#------------------------------------------------------------------------------



def randomPoints(pdf, Ld, N, pdfMax=1):
    # Creates an array of N points randomly distributed according to pdf in a
    # domain size given by Ld (and starting in the origin) using the Monte
    # Carlo method. The pdf is assumed to have a max-value of 1 unless
    # otherwise specified. Useful for creating arrays of position/velocity
    # vectors of various distributions.

    Ld = np.array(Ld) # In case it's a list
    dim = len(Ld)

    assert not isinstance(pdf(np.ones(dim)),(list,np.ndarray)) ,\
        "pdf returns non-scalar value"

    points = np.array([]).reshape(0,dim)

    while len(points)<N:

        # Creates missing points
        n = N-len(points)
        newPoints = np.random.rand(n,dim+1) # Last value to be compared with pdf
        newPoints *= np.append(Ld,pdfMax)   # Stretch axes

        # Only keep points below pdf
        newPoints = [x[0:dim] for x in newPoints if x[dim]<pdf(x[0:dim])]
        newPoints = np.array(newPoints).reshape(-1,dim)
        points = np.concatenate([points,newPoints])

    return points

def maxwellian(vd, vth, N):
    """
    Returns N maxwellian velocity vectors with drift and thermal velocity
    vectors vd and vth, respectively, in an (N,d) array where d is the number
    of geometric dimensions. If you have an (N,d) array xs (e.g. position
    vectors) the following call would also be valid:

        maxwellian(vd, vth, xs.shape)

    or even

        np.random.normal(vd, vth, xs.shape)

    where np stands for numpy. However, np.random.normal fails when one of the
    elements in the vector vth is zero. maxwellian replaces zeroes with the
    machine epsilon.

    Either vd or vth may be a scalar, or both if N is a shape-tuple.
    """

    if isinstance(N,int):
        d = max(len(vd),len(vth))
    else:
        (N,d) = N

    # Replace zeroes in vth for machine epsilon
    if isinstance(vth,(float,int)): vth = [vth]
    vth = np.array(vth,dtype=float)
    indices = np.where(vth==0.0)[0]
    vth[indices] = np.finfo(float).eps

    return np.random.normal(vd, vth, (N,d))

def initLangmuir(pop, Ld, vd, vth, A, mode, Npc):
    """
    Initializes population pop with particles corresponding to electron-ion
    plasma oscillations with amplitude A in density, thermal velocity vth[0]
    and vth[1] for electrons and ions, respectively, and drift velocity vector
    vd. mesh and Ld is the mesh and it's size, while Npc is the number of
    simulation particles per cell per specie. Normalization is such that
    angular electron plasma frequency is one. mode is the number of half
    periods per domain length in the x-direction.
    """

    mesh = pop.mesh

    # Adding electrons
    q, m, N = stdSpecie(mesh, Ld, -1, 1, Npc)
    pdf = lambda x: 1+A*np.sin(mode*2*np.pi*x[0]/Ld[0])
    xs = randomPoints(pdf, Ld, N, pdfMax=1+A)
    vs = maxwellian(vd, vth[0], xs.shape)
    pop.addParticles(xs,vs,q,m)

    # Adding ions
    q, m, N = stdSpecie(mesh, Ld, 1, 1836, Npc)
    pdf = lambda x: 1
    xs = randomPoints(pdf, Ld, N)
    vs = maxwellian(vd, vth[1], xs.shape)
    pop.addParticles(xs,vs,q,m)
