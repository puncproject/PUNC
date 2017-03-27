from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from math import erf
from itertools import combinations
import copy

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

def random_points(pdf, Ld, N, pdf_max=1):
    """
    Creates an array of N points randomly distributed according to pdf in a
    domain size given by Ld (and starting in the origin) using the Monte
    Carlo method. The pdf is assumed to have a max-value of 1 unless
    otherwise specified. Useful for creating arrays of position/velocity
    vectors of various distributions.

    If the domain contains objects, the points inside the objects are removed.
    """

    Ld = np.array(Ld) # In case it's a list
    dim = len(Ld)

    assert not isinstance(pdf(np.ones(dim)),(list,np.ndarray)) ,\
        "pdf returns non-scalar value"

    points = np.array([]).reshape(0,dim)
    while len(points)<N:

        # Creates missing points
        n = N-len(points)
        new_points = np.random.rand(n,dim+1) # Last value to be compared with pdf
        new_points *= np.append(Ld,pdf_max)   # Stretch axes

        # Only keep points below pdf
        new_points = [x[0:dim] for x in new_points if x[dim]<pdf(x[0:dim])]
        new_points = np.array(new_points).reshape(-1,dim)
        points = np.concatenate([points,new_points])

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

# No longer in use due to create_mesh_pdf() but nice to have
def create_object_pdf(pdf, objects):
    """
    Given a pdf for the hole simulation domain, creates a new pdf that is zero
    inside the objects.

    Args:
        pdf: A probability distribution function
        objects: A list of Object objects

    returns:
        A new pdf with value zero inside the objects domain.
    """
    object_pdf = lambda x, pdf=pdf:\
                 0 if any(c.inside(x, True) for c in objects) else pdf(x)

    return object_pdf

def create_mesh_pdf(pdf, mesh):
    """
    Takes a probability density function (pdf) and a DOLFIN mesh and returns a
    new pdf which is zero outside the mesh (such as inside objects) but
    otherwise identical to the original pdf.
    """
    mesh.init(0, mesh.topology().dim())
    tree = mesh.bounding_box_tree()

    def mesh_pdf(x):
        cell_id = tree.compute_first_entity_collision(df.Point(*x))
        inside_mesh = int(cell_id!=__UINT32_MAX__ and cell_id!=-1)
        return inside_mesh*pdf(x)

    return mesh_pdf

def num_injected_particles(A, dt, n_p, v_n, alpha):
    """
    This function calculates the number of particles that needs to be injected
    through a surface of the exterior boundary with area, A, based on a drifting
    Maxwellian distribution, at each time step.
    """
    N = n_p*A*dt*( (alpha/(np.sqrt(2*np.pi)) * np.exp(-v_n**2/(2*alpha**2))) +\
                    0.5*v_n*(1. + erf(v_n/(alpha*np.sqrt(2)))) )
    return N

class Initialize(object):
    """
    Initializes population pop with particles according to a prespecified
    probability distribution function, pdf, for each particle species
    (currently, only electrons and hydrogen ions) in the background plasma with
    thermal velocity vth and drift velocity vector vd. mesh and Ld is the mesh
    and it's size, while Npc is the number of simulation particles per cell per
    specie. Normalization is such that angular electron plasma frequency is one.

    """
    def __init__(self, pop, pdf, Ld, vd, vth, Npc, pdf_max = 1, dt = 0.1,
                 charge = [-1,1], mass = [1, 1836], objects = None):

        self.pop = pop
        self.mesh = pop.mesh
        self.pdf = pdf
        self.Ld = Ld
        self.vd = vd
        self.vth = vth
        self.Npc = Npc
        self.pdf_max = pdf_max

        self.dt = dt
        self.dim = len(self.Ld)
        self.objects = objects

        assert len(charge)==len(mass)
        self.num_species = len(charge)
        self.q = copy.deepcopy(charge)
        self.m = copy.deepcopy(mass)

        self.normalize()

    def inject(self):
        for i in range(self.num_species):
            # Note: This is a pdf for a plane. The pdf for the initializer part
            # is a pdf for a volume. That it even works using the same both
            # places is just because it equals "lambda x: 1". Use separate
            # pdf for injector.
            xs = self.get_xs(self.pdf[i], self.n_particles[i], self.surfaces)
            vs = maxwellian(self.vd, self.vth[i], xs.shape)
            xs, vs = self.inside(xs, vs)
            self.pop.add_particles(xs,vs,self.q[i],self.m[i])

    def get_xs(self, pdf, n_particles, surfaces):
        n_tot = np.sum(n_particles)
        xs = np.zeros((n_tot, self.dim))
        tmp = np.array([np.sum(n_particles[:j+1]) for j in range(2*self.dim)])

        for j in range(self.dim):
            xs[tmp[2*j]:tmp[2*j+1], j] = self.Ld[j]
        tmp = np.insert(tmp, 0, 0, axis=0)

        slices = \
        [list(j) for j in reversed(list(combinations(np.arange(0,self.dim),
                                                                 self.dim-1)))]
        k = 0
        for j in range(2*self.dim):
            pos = random_points(pdf, surfaces[j+k], n_particles[j])
            if n_particles[j] != 0:
                for l in range(len(slices[j+k])):
                    xs[tmp[j]:tmp[j+1], slices[j+k][l]] = pos[:,l]
            if j%(len(n_particles)/self.dim)==0:
                k -= 1
        return xs

    def inside(self, xs, vs):
        w_random = np.random.rand(len(vs))
        if len(xs) != 0:
            for j in range(self.dim):
                xs[:,j]  += self.dt*w_random*vs[:,j]

        outside = []
        for j in range(len(xs)):
            x = xs[j]
            for k in range(self.dim):
                if x[k] < 0.0 or x[k] > self.Ld[k]:
                    outside.append(j)
                    break
        xs = np.delete(xs, outside, axis=0)
        vs = np.delete(vs, outside, axis=0)
        return xs, vs

    def initialize_injection(self):

        self.surfaces = [list(i) for i in reversed(list(combinations(self.Ld,
                                                                self.dim-1)))]
        surface_area = [np.prod(self.surfaces[i]) for i in range(self.dim)]

        # The unit vector normal to outer boundary surfaces
        unit_vec = np.identity(self.dim)
        # Normal components of drift velocity
        vd_normal = np.empty(2*self.dim)
        s = -1 # To insure normal vectors point outward from the domain
        j = 0
        for i in range(2*self.dim):
            si = s**i
            if np.sign(si) == -1:
                j += 1
            vd_normal[i] = np.dot(self.vd, si*unit_vec[i-j])

        self.n_particles = []
        for i in range(self.num_species):
            self.n_particles.append([])
            k = 0
            for j in range(2*self.dim):
                self.n_particles[i].append(\
                num_injected_particles(surface_area[j+k],
                                       self.dt,
                                       self.N/np.prod(self.Ld),
                                       vd_normal[j],
                                       self.vth[i]))
                if j%(len(vd_normal)/self.dim)==0:
                    k -= 1

            diff_e = [(j - int(j)) for j in self.n_particles[i]]
            self.n_particles[i] = [int(j) for j in self.n_particles[i]]
            self.n_particles[i][0] += int(sum(diff_e))
