from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np

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

def random_points(pdf, Ld, N, pdfMax=1, objects=None):
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
        newPoints = np.random.rand(n,dim+1) # Last value to be compared with pdf
        newPoints *= np.append(Ld,pdfMax)   # Stretch axes

        # Only keep points below pdf
        newPoints = [x[0:dim] for x in newPoints if x[dim]<pdf(x[0:dim])]
        newPoints = np.array(newPoints).reshape(-1,dim)

        if objects is not None:
            indices = []
            for i, p in enumerate(newPoints):
                for o in objects:
                    if o.inside(p):
                        indices.append(i)
                        break
            newPoints = np.delete(newPoints, indices, axis=0)
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

class InitialConditions:
    """
    Initializes population pop with particles according to a prespecified
    probability distribution function, pdf, for each particle species
    (currently, only electrons and hydrogen ions) in the background plasma with
    thermal velocity vth and drift velocity vector vd. mesh and Ld is the mesh
    and it's size, while Npc is the number of simulation particles per cell per
    specie. Normalization is such that angular electron plasma frequency is one. 

    """
    def __init__(self, pop, pdf, Ld, vd, vth, Npc, objects=None):
        self.pop = pop
        self.mesh = pop.mesh
        self.pdf = pdf
        self.Ld = Ld
        self.vd = vd
        self.vth = vth
        self.Npc = Npc
        self.objects = objects

        self.num_species = 2
        self.charge = [-1, 1]
        self.mass = [1, 1836]

    def initialize(self):
        for i in range(self.num_species):
            q, m, N = stdSpecie(self.mesh, self.Ld, self.charge[i],
                                self.mass[i], self.Npc)
            xs = random_points(self.pdf[i], self.Ld, N, objects=self.objects)
            vs = maxwellian(self.vd, self.vth[i], xs.shape)
            self.pop.addParticles(xs,vs,q,m)
