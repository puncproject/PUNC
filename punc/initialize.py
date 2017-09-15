from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from scipy.special import erfcinv, erfinv, erf, erfc
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

class SRS(object):
    """
    Standard rejection sampling

    Generates random numbers from a given probability density function, pdf.

    Attributes:
                pdf       : probability density function
                pdf_max   : the maximum value of the pdf
                domain    : the domain where the pdf is defined
                transform (optional) : a function to transform the pdf to its
                                       original form
    """
    def __init__(self, pdf, pdf_max, domain=[0.,1.], transform=lambda t: t):
        """
        Arguments:
                   pdf       : probability density function
                   pdf_max   : the maximum value of the pdf
                   domain    : the domain where the pdf is defined
                   transform (optional) : a function to transform the pdf to its
                                          original form
        """
        self.pdf = pdf
        self.pdf_max = pdf_max
        self.lb = domain[0]
        self.ub = domain[1]
        self.transform = transform

    def sample(self, N):
        """
        Samples N random numbers, vs, from the pdf.

        Arguments:
                  N: Number of random numbers to be generated

        Returns:
                 vs: Array of N random numbers generated from the pdf
        """
        vs = np.array([])
        while len(vs) < N:
            n = N - len(vs)
            r1  = np.random.uniform(self.lb, self.ub, n)
            r2 = self.pdf_max*np.random.rand(n)
            v = [self.transform(r1[i]) for i in range(n) \
                                           if r2[i] < self.pdf(r1[i])]
            v = np.array(v)
            vs = np.concatenate([vs,v])
        return vs

class ARS(object):
    """
    Adaptive rejection sampling

    Given a strictly convex potential function, V(t), of a probability density
    function, p(t), i.e.,

                        V(t) = -log(p(t)),

    it generates random numbers whose distribution function is p(t).

    Attributes:
                Vt     : potential function, V(t)
                dVdt   : derivative of potential function, V(t)
                sp     : list of support points for construction of piecewise
                         linear functions
                domain : domian of the potential function, V(t)
                nsp    : total number of support points to be used to construct
                         piecewise-linear functions

                transform (optional) : a function to transform the pdf to its
                                       original form

                y      : ordinates corresponding to support points
                dy     : derivatives of the potential function at support points

    """
    def __init__(self, Vt, dVdt, sp=[0.3,0.7,0.9], domain=[0,1],
                 nsp=50, transform = lambda t: t):
        """
        Arguments:
                    Vt     : potential function, V(t)
                    dVdt   : derivative of potential function, V(t)
                    sp     : list of support points for construction of
                             piecewise linear functions
                    domain : domian of the potential function, V(t)
                    nsp    : total number of support points to be used to
                             construct piecewise-linear functions

                    transform (optional) : a function to transform the pdf to
                                           its original form
        """
        self.Vt = Vt
        self.dVdt = dVdt
        self.sp = np.array(sp)
        self.lb = domain[0]
        self.ub = domain[1]
        self.transform = transform

        self.ns = nsp
        self.y = self.Vt(self.sp)
        self.dy = self.dVdt(self.sp)

        self.exponentiated_lower_hull()

    def tangent_intersections(self):
        """
        Given a list of support points and the corresponding ordinates and the
        derivative of the potential function, it finds the point of intersection
        between the piecewise-linear functions
        """
        self.z = np.zeros(self.sp.__len__() + 1)
        self.z[0] = self.lb
        self.z[1:-1] = (np.diff(self.y) - np.diff(self.sp*self.dy))/\
                       -np.diff(self.dy)
        self.z[-1] = self.ub

    def lower_hull(self, s_new, y_new, dy_new):
        """
        Contructs the lower hull, Y(t), of the potential by piecewise-linear
        functions at the support points.
        """
        if s_new.__len__() > 0:
            sx = np.hstack([self.sp, s_new])
            reorder = np.argsort(sx)
            self.sp = sx[reorder]
            self.y = np.hstack([self.y, y_new])[reorder]
            self.dy = np.hstack([self.dy, dy_new])[reorder]

        self.tangent_intersections()

        N = self.y.__len__()
        self.Yt = self.dy[[0]+range(N)]*(self.z-self.sp[[0]+range(N)]) + \
                   self.y[[0]+range(N)]

    def exponentiated_lower_hull(self, s_new = [], y_new = [], dy_new = []):
        """
        Constructs a piecwise-exponential lower hull of the potential.
        Moreover, it calculates the cumulative distribution function for the
        piecewise-expontential density function.

        Here, 'integrand' is the integrand in the expression for the cdf
        'exp_cdf' is the cumulative distribution function for the
        piecewise-expontential density function. 'c_i' is the normalization
        factor of the exponentiated lower hull.
        """
        self.lower_hull(s_new, y_new, dy_new)
        integrand = np.diff(np.exp(-self.Yt))/(-self.dy)
        self.exp_cdf = np.hstack([0,np.cumsum(integrand)])
        self.c_i = self.exp_cdf[-1]

    def inv_exp_cdf(self):
        """
        The inverse function of the cumulative distribution function for the
        piecwise-exponential function.

        First samples a uniform random number, r, and finds the index, i, for
        the largest intersection point, z_i, between the piecewise-linear
        functions, such that

                        cdf(z_i) < r,

        where cdf is the cumulative distribution function for
        piecwise-exponential function. Then, it finds the t-value by using the
        analytical expression of the inverse distribution function.

        Returns:
                t     : generated t-value from the inverse cdf of the
                        piecwise-exponential function
                index : the index of the largest z_i
        """
        r = np.random.rand()

        index = np.nonzero(self.exp_cdf/self.c_i < r)[0][-1]

        t = self.sp[index] + (self.y[index] +\
             np.log(-self.dy[index]*(self.c_i*r - self.exp_cdf[index]) +\
             np.exp(-self.Yt[index]))) / (-self.dy[index])

        return [t, index]

    def sample(self, *args):
        """
        Samples N random numbers, t, from the density distribution function

                        p(t) = exp(-V(t))

        Arguments:
                   N: number of t-values to be sampled

        Returns:
                  ts: Array of N sampled t-values

        """
        N = args[0]
        ts = np.array([])
        while len(ts) < N:
            [t, index] = self.inv_exp_cdf()
            y_t = self.Vt(t)
            dy_t = self.dVdt(t)
            Y_i = self.y[index] + (t-self.sp[index])*self.dy[index]

            r = np.random.rand()
            if r <= np.exp(Y_i - y_t):
                ts = np.concatenate([ts, [t]])

            if self.Yt.__len__() < self.ns:
                self.exponentiated_lower_hull([t],[y_t],[dy_t])

        return self.transform(ts)

class MaxwellianVelocities(object):
    """
    Generates both drifting and non-drifting Maxwellian velocities for loading
    and injection of particles.

    Attributes:
            v_thermal (array) : Thermal velocity of particles
            v_drift (array)   : Drift velocity of particles
            periodic (list)   : A list containing which directions are
                               periodic or not.
            loading (boolean)  : Generates Maxwellian velocities for particle
                                loading
            injection (boolean): Generates Maxwellian velocities for particle
                                particle injection through the outer
                                boundaries
            sampling_method (string): Currently implemented methods are
                                      standard rejection sampling, 'srs', and
                                      adaptive rejection sampling, 'ars'.
    """
    def __init__(self, v_thermal, v_drift, periodic, loading=True,
                 injection=False, sampling_method='srs'):
        """
        Arguments:
            v_thermal (array) : Thermal velocity of particles
            v_drift (array)   : Drift velocity of particles
            periodic (list)   : A list containing which directions are
                               periodic or not.
            loading (boolean)  : Generates Maxwellian velocities for particle
                                loading
            injection (boolean): Generates Maxwellian velocities for particle
                                particle injection through the outer
                                boundaries
            sampling_method (string): currently implemented methods are
                                    standard rejection sampling, 'srs', and
                                    adaptive rejection sampling, 'ars'.
        """
        self.vth = v_thermal
        self.vd = v_drift
        self.periodic = periodic
        self.dim = len(self.vd)

        if loading:
            self.cdf = [None]*self.dim
            for i in range(self.dim):
                if self.vd[i] == 0:
                    self.cdf[i] = self.cdf_maxwellian()
                else:
                    self.cdf[i] = self.cdf_drifting_maxwellian(i)

        if injection:
            if all(v==0 for v in self.vd):
                self.cdf = self.cdf_flux_nondrifting()
            else:
                if sampling_method == 'srs':
                    self.initialize_srs()
                elif sampling_method == 'ars':
                    self.initialize_ars()
                self.cdf = self.cdf_flux_drifting()

    def initialize_ars(self):
        """
        Initializes the adaptive rejection sampling for a drifting Maxwellian
        density function.
        """
        ars_inward = []
        ars_backward = []
        for i in range(self.dim):
            d = self.vd[i]/self.vth
            if self.vd[i] != 0:
                V_inward = lambda t,d=d: -np.log(t) + 3.*np.log(1.-t) +\
                                               (0.5*(t/(1.-t)-d)**2)

                dV_inward = lambda t,d=d: (-1./t) - (3./(1.-t)) +\
                                      ((t*(1.+d)-d)/(1.-t)**3)

                V_backward = lambda t,d=d: -np.log(t)+3.*np.log(1.-t)+\
                                       (0.5*(t/(t-1.)-d)**2)

                dV_backward = lambda t,d=d: (-1./t) - (3./(1.-t)) +\
                                        ((t*(d-1.)-d)/(t-1.)**3)

                transform_inward = lambda t, vth=self.vth: vth*t/(1.-t)
                transform_backward = lambda t, vth=self.vth: vth*t/(t-1.)

                ars_inward.append(ARS(V_inward, dV_inward,
                                      transform = transform_inward))
                ars_backward.append(ARS(V_backward, dV_backward,
                                        transform = transform_backward))
        self.rs = [ars_inward, ars_backward]

    def initialize_srs(self):
        """
        Initializes the standard rejection sampling for a drifting Maxwellian
        density function.
        """
        rs_inward = []
        rs_backward = []
        for i in range(self.dim):
            if self.vd[i] != 0:
                d = self.vd[i]/self.vth
                s = 1
                pdf_inward = self.pdf_flux(d, s)
                transform_inward = lambda t, vth=self.vth: vth*t/(1.-t)
                pdf_max_inward = \
                                self.pdf_max_drifting_maxwellian(pdf_inward,d,s)

                rs_inward.append(SRS(pdf_inward, pdf_max_inward,
                                     transform=transform_inward))

                s = -1
                pdf_backward = self.pdf_flux(d, s)
                transform_backward = lambda t, vth=self.vth: vth*t/(t-1.)
                pdf_max_backward = \
                              self.pdf_max_drifting_maxwellian(pdf_backward,d,s)

                rs_backward.append(SRS(pdf_backward, pdf_max_backward,
                                       transform=transform_backward))

        self.rs = [rs_inward, rs_backward]

    def pdf_max_drifting_maxwellian(self, pdf, d, s):
        """
        Finds the root of the cubic equation corresponding to the maximum value
        of the transformed drifting-Maxwellian distribution function.

        Returns:
                The maximum value of the transformed drifting-Maxwellian
                distribution function
        """
        ca = 2.0
        cb = -4.0 - s*d
        cc = s*d
        cd = 1.0

        p = (3.0*ca*cc - cb*cb)/(3.0*ca*ca)
        q = (2.0*cb*cb*cb - 9.0*ca*cb*cc + 27*ca*ca*cd) / (27*ca*ca*ca)

        sqrt_term = np.sqrt(-p/3.0)
        root = 2.0*sqrt_term*\
               np.cos(np.arccos(3.0*q/(2.0*p*sqrt_term))/3.0 - 2.0*np.pi/3.0)
        return pdf(root - cb/(3.0*ca))

    def pdf_flux(self, d, s):
        """
        The transformed distribution function of the flux for the
        drifting-Maxwellian. The transformation is given by

                          x = t/(1-t),

        for inward flux, and

                          x = t/(t-1),

        for backward flux.

        Returns:
                Transformed distribution function of the flux for the
                drifting-Maxwellian distribution function.
        """
        pdf = lambda t, d=d, s=s: s*t*np.exp(-.5*(t/(s*(1.-t))-d)**2)/\
                                      ((s*(1.-t))**3)
        return pdf

    def cdf_maxwellian(self):
        """
        The inverse of the cumulative distribution function for a nondrifting-
        Maxwellian distribution function.
        """
        cdf = lambda x, vth=self.vth: np.sqrt(2.)*vth*erfinv(2*x-1.)
        return cdf

    def cdf_drifting_maxwellian(self, indx):
        """
        The inverse of the cumulative distribution function for a drifting-
        Maxwellian distribution function.
        """
        cdf = lambda x,vd=self.vd[indx],vth=self.vth: vd -np.sqrt(2.)*vth*\
                                                                 erfcinv(2*x)
        return cdf

    def cdf_inward_flux_nondrifting(self):
        """
        The inverse of the cumulative distribution function for the inward flux
        of a nondrifting-Maxwellian distribution function.
        """
        return lambda x, vth=self.vth: vth*np.sqrt(-2.*np.log(1.-x))

    def cdf_backward_flux_nondrifting(self):
        """
        The inverse of the cumulative distribution function for the backward
        flux of a drifting-Maxwellian distribution function.
        """
        return lambda x, vth=self.vth: -vth*np.sqrt(-2.*np.log(x))

    def cdf_flux_nondrifting(self):
        """
        Generates all the inverse cumulative distribution functions for
        nondrifting-Maxwellian distributed particles, needed to generate
        velocities for the injection of particles thourgh the exterior surface
        boundaries.
        """
        maxwellian_flux = self.cdf_maxwellian()
        inward_flux     = self.cdf_inward_flux_nondrifting()
        backward_flux   = self.cdf_backward_flux_nondrifting()

        inward = [inward_flux, maxwellian_flux]
        backward = [backward_flux, maxwellian_flux]
        if self.dim == 3:
            inward.append(maxwellian_flux)
            backward.append(maxwellian_flux)

        def rotate(l, x):
            return l[-x%len(l):] + l[:-x%len(l)]

        l = [i for i in range(self.dim)]
        indices = [rotate(l, i) for i in range(self.dim)]

        inward_cdfs = [[inward[indices[i][j]] for j in range(self.dim)] \
                                                  for i in range(self.dim)]
        backward_cdfs = [[backward[indices[i][j]] for j in range(self.dim)] \
                                                      for i in range(self.dim)]

        # Extract only those boundaries that are not periodic
        inward_cdf = [inward_cdfs[i] for i in range(self.dim)\
                       if not self.periodic[i]]
        backward_cdf = [backward_cdfs[i] for i in range(self.dim)\
                       if not self.periodic[i]]
        return inward_cdf + backward_cdf

    def cdf_flux_drifting(self):
        """
        Generates all the inverse cumulative distribution functions or the
        sampling method for drifting-Maxwellian distributed particles, needed to
        generate velocities for the injection of particles thourgh the exterior
        surface boundaries.
        """
        def rotate(l, x):
            return l[-x%len(l):] + l[:-x%len(l)]

        l = [i for i in range(self.dim)]
        indx = [rotate(l, i) for i in range(self.dim)]

        k = [rotate([i for i in range(j*self.dim,(j+1)*self.dim)], j) \
                           for j in range(self.dim)]

        inward   = [None]*self.dim**2
        backward = [None]*self.dim**2
        for i in range(self.dim):
            for j in range(self.dim):
                if j == 0:
                    if self.vd[indx[i][j]] != 0:
                        inward[k[i][j]] = lambda n, i=i: self.rs[0][i].sample(n)
                        backward[k[i][j]]=lambda n, i=i: self.rs[1][i].sample(n)
                    if self.vd[indx[i][j]] == 0:
                        inward[k[i][j]] = lambda n: \
                                          self.sample_inward_flux_nondrifting(n)
                        backward[k[i][j]] = lambda n: \
                                        self.sample_backward_flux_nondrifting(n)
                else:
                    if self.vd[indx[i][j]] != 0:
                        inward[k[i][j]] = lambda n, ind=indx[i][j]: \
                                          self.sample_drifting_maxwellian(ind,n)
                        backward[k[i][j]] = lambda n, ind=indx[i][j]: \
                                          self.sample_drifting_maxwellian(ind,n)
                    if self.vd[indx[i][j]] == 0:
                        inward[k[i][j]]   = lambda n: self.sample_maxwellian(n)
                        backward[k[i][j]] = lambda n: self.sample_maxwellian(n)

        # Extract only those boundaries that are not periodic
        inward_cdfs = [inward[i*self.dim:(i+1)*self.dim] \
                       for i in range(self.dim)\
                           if not self.periodic[i]]
        backward_cdfs = [backward[i*self.dim:(i+1)*self.dim] \
                         for i in range(self.dim)\
                             if not self.periodic[i]]

        cdfs_in, cdfs_back = [], []
        for i in range(len(inward_cdfs)):
            cdfs_in += inward_cdfs[i]
            cdfs_back += backward_cdfs[i]

        return cdfs_in + cdfs_back

    def sample_loading(self, N):
        """
        Samples and returns N velocities for drifting- or nondrifting-Maxwellian
        distribution function, used for loading of particles.
        """
        w_r = np.random.rand(N, self.dim)
        vs = np.empty((N, self.dim))
        for j in range(self.dim):
            vs[:,j] = self.cdf[j](w_r[:,j])
        return np.array(vs)

    def sample_maxwellian(self, N):
        """
        Samples and returns N velocities from the nondrifting-Maxwellian
        distribution function.
        """
        w_r = np.random.rand(N)
        cdf = self.cdf_maxwellian()
        return [cdf(w_r[i]) for i in range(N)]

    def sample_drifting_maxwellian(self, indx, N):
        """
        Samples and returns N velocities from the drifting-Maxwellian
        distribution function.
        """
        w_r = np.random.rand(N)
        cdf = self.cdf_drifting_maxwellian(indx)
        return [cdf(w_r[i]) for i in range(N)]

    def sample_inward_flux_nondrifting(self, N):
        """
        Samples and returns N velocities for the inward flux of particles,
        entring the simulation domain, for a nondrifting-Maxwellian distribution
        function.
        """
        w_r = np.random.rand(N)
        cdf = self.cdf_inward_flux_nondrifting()
        vs = [cdf(w_r[i]) for i in range(N)]
        return np.array(vs)

    def sample_backward_flux_nondrifting(self, N):
        """
        Samples and returns N velocities for the backward flux of particles,
        entring the simulation domain, for a nondrifting-Maxwellian distribution
        function.
        """
        w_r = np.random.rand(N)
        cdf = self.cdf_backward_flux_nondrifting()
        vs = [cdf(w_r[i]) for i in range(N)]
        return np.array(vs)

    def sample_flux_nondrifting(self, k, N):
        """
        Samples and returns N velocities for the influx of particles based on a
        nondrifting-Maxwellian distribution function, used for injection of
        particles into the simulation domain through the k-th boundary surface.

        Arguments:
                 k (int): k-th exterior surface boundary
                 N (int): Number of velocities to be sampled.

        Returns:
                vs (array): Array containing N sampled velocities.

        Note: The exterior surface boundaries are numbered as follows (3D):

                k = 0: the surface corresponding to x = 0
                k = 1: ---------''-------------- to y = 0
                k = 2: ---------''-------------- to z = 0
                k = 3: the surface corresponding to x = Ld[0]
                k = 4: ---------''-------------- to y = Ld[1]
                k = 5: ---------''-------------- to z = Ld[2]
        """
        w_r = np.random.rand(N, self.dim)
        vs = [[self.cdf[k][j](w_r[i,j]) for j in range(self.dim)] \
                                         for i in range(N)]
        return np.array(vs)

    def sample_flux_drifting(self, k, N):
        """
        Samples and returns N velocities for the influx of particles based on a
        drifting-Maxwellian distribution function, used for injection of
        particles into the simulation domain through the k-th boundary surface.

        Arguments:
                 k (int): k-th exterior surface boundary
                 N (int): Number of velocities to be sampled.

        Returns:
                vs (array): Array containing N sampled velocities.

        Note: The exterior surface boundaries are numbered as follows (3D):

                k = 0: the surface corresponding to x = 0
                k = 1: ---------''-------------- to y = 0
                k = 2: ---------''-------------- to z = 0
                k = 3: the surface corresponding to x = Ld[0]
                k = 4: ---------''-------------- to y = Ld[1]
                k = 5: ---------''-------------- to z = Ld[2]
        """
        vs = np.empty((N, self.dim))
        for j in range(self.dim):
            vs[:,j] = self.cdf[k+j](N)
        return np.array(vs)

def maxwellian_flux_number(dim, d, v_thermal, v_drift, plasma_density,
                           surface_area, dt, periodic):
    """
    For each boundary surface that are not periodic, calculates the number
    of total particles, based on a drifting or nondrifting-Maxwellian
    distribution function, to be injected through the surface.

    Arguments:
          dim               : Space dimensions
          d                 : The number of non-periodic dimensions
          v_thermal         : Thermal velocity
          v_drift           : Drift velocity
          plasma_density    : Plasma density
          surface_area      : A list containing all the surface areas of the
                              exterior boundaries
          dt                :  Simulation time step
          periodic (boolean): A list indicating the non-periodic boundaries.

    Returns:
            N : A list containing the total number of particles to be injected
                at each non-periodic surface boundary.
    """
    def inward_particle_flux_number(i):
        """
        Calculates the total number of particles to be injected at the exterior
        boundaries for the inward flux.
        """
        N = plasma_density*surface_area[i]*dt*( v_thermal/(np.sqrt(2*np.pi))*\
                                      np.exp(-0.5*(v_drift[i]/v_thermal)**2)+\
                0.5*v_drift[i]*(1. + erf(v_drift[i]/(np.sqrt(2)*v_thermal))) )
        return N

    def backward_particle_flux_number(i):
        """
        Calculates the total number of particles to be injected at the exterior
        boundaries for the backward flux.
        """
        N = np.abs(plasma_density*surface_area[i]*dt*\
           (0.5*v_drift[i]*erfc(v_drift[i]/(np.sqrt(2)*v_thermal)) -\
           v_thermal/(np.sqrt(2*np.pi))*np.exp(-0.5*(v_drift[i]/v_thermal)**2)))
        return N

    N = np.empty(d*2)
    j = 0
    for i in range(dim):
        if not periodic[i]:
            N[j] = inward_particle_flux_number(i)
            N[j+d] = backward_particle_flux_number(i)
            j += 1
    return N

class Injector(object):
    """
    Injects particles from the exterior surface boundaries into the simulation
    domain. For every non-periodic boundary surface, calculates the number of
    particles, N, to be injected into the domain from a probability
    distribution function. Currently, only (drifting or non-drifting) Maxwellian
    distribution function is implemented.
    For each of these N particles a random position xs, and a velocity vs is
    assigned consistent with the predefined distribution function. The particles
    are then placed into the domain at positions xs + r*vs*dt, where dt is the
    time step and r is a random number. The choise of the random number r is to
    account for the fact that the particle could have entered the domain with a
    uniform probability in the range [xs-vs*dt, xs]. After assigning the
    position xs + r*vs*dt, only the particles that have entered the simulation
    domain are kept and injected into the domain. The process of assigning
    positions and velocities continues untill N particles have entered the
    simulation domain, i.e., xs + r*vs*dt must be inside the simulation domain.

    Attributes:
            specie            : particle specie
            v_drift           : drift velocity
            v_thermal         : thermal velocity
            temperature       : temperature of the particles
            mass              : mass of the particles
            Ld                : Simulation domain
            plasma_density    : plasma density
            periodic (boolean): A list containing which directions are periodic.
            weight            : the statistical weight of particles
    """
    def __init__(self, pop, specie, Ld, dt, plasma_density, weight, periodic,
                 maxwellian = True):
        """
        Arguments:
            specie            : particle specie (electron or ion)
            Ld                : Simulation domain
            dt                : Simulation time step
            plasma_density    : plasma density
            weight            : the statistical weight of particles
            periodic (boolean): A list containing which directions are periodic.
            maxwellian (boolean): Probability distribution used to calculate the
                                  particle flux number at each boundary surface,
                                  and to sample velocities for the injected
                                  particles.
        """
        self.pop = pop
        self.specie = specie
        self.v_drift = pop.species[specie].v_drift
        self.v_thermal = pop.species[specie].v_thermal
        self.temperature = pop.species[specie].temperature_raw
        self.mass = pop.species[specie].mass
        self.Ld = Ld
        self.periodic = periodic
        self.dim = len(Ld)
        self.dt = dt
        self.plasma_density = plasma_density
        self.weight = weight

        self.initialize_injection()

        if maxwellian:
            self.num_particles = maxwellian_flux_number(self.dim, self.d,
                                                        self.v_thermal,
                                                        self.v_drift,
                                                        self.plasma_density,
                                                        self.surface_area,
                                                        self.dt,
                                                        self.periodic)
            self.mv = MaxwellianVelocities(self.v_thermal, self.v_drift,
                                           self.periodic, loading=False,
                                           injection=True,
                                           sampling_method='srs')
            if all(v==0 for v in self.v_drift):
                self.sample_vs = lambda j, n: \
                                           self.mv.sample_flux_nondrifting(j, n)
            else:
                self.sample_vs = lambda j, n, dim = self.dim:\
                                           self.mv.sample_flux_drifting(j*dim,n)
        else:
            print("Generation of velocities for non-Maxwellian distributions\
                   are not implemented yet.")
            pass

    def inject(self):
        """
        For each non-periodic boundary surface, injects N particles with
        velocities calculated from a predefined probability distribution
        function.

        N is a real number, therfore to have a statistically correct number of
        injected particles, at each time step a random number r is generated,
        and if r < N - int(N), then the number of injected particles is
        incremented by one, i.e., number of injected particles N' = int(N) + 1.
        If r > N - int(N), then the number of injected particles is N' = int(N).

        For each non-periodic boundary surface and for each of the N' particles
        random positions and velocties are generated consistent with the
        predefined probability distribution functions for the simulation domain.

        The particles are then injected into the simulation domain by assigning
        the position xs + r*vs*dt, where r is a random number to account for the
        fact that the particle could have entered the domain with a uniform
        probability in the range [xs-vs*dt, xs].

        The process of assigning positions and velocities continues untill N'
        particles have entered the simulation domain, i.e., xs + r*vs*dt is
        inside the simulation domain.
        """
        for j in range(2*self.d):
            N = int(self.num_particles[j])
            if np.random.rand() < self.num_particles[j] - N:
                N += 1
            xs = np.array([]).reshape(0, self.dim)
            vs = np.array([]).reshape(0, self.dim)
            while len(xs) < N:
                n = N-len(xs)
                new_xs = self.sample_positions(j, n)
                vel = self.sample_vs(j,n)
                w_random = np.random.rand(len(vel))

                for i in range(self.dim):
                    new_xs[:,i]  += self.dt*w_random*vel[:,i]

                for i in range(len(new_xs)):
                    x = new_xs[i]
                    v = vel[i]
                    if all(x[k] >= 0.0 and x[k] <= self.Ld[k]\
                           for k in range(self.dim)):
                        x = np.array(x).reshape(-1,self.dim)
                        v = np.array(v).reshape(-1,self.dim)
                        xs = np.concatenate([xs,x])
                        vs = np.concatenate([vs,v])

            self.pop.add_particles_of_specie(self.specie, xs, vs)

    def sample_positions(self, j, n):
        """
        Samples random positions on the exterior surface boundaries.

        Arguments:
                 j (int): The j-th boundary surface
                 n (int): Number of particle positions to be sampled

        Returns:
                xs: Array containing particle positions for the j-th surface
                   boundary.
        """
        pos = np.random.rand(n,self.dim-1)
        xs = np.empty((len(pos), self.dim))
        xs[:, self.index[j]] = self.L[j]

        for i in range(len(self.slices[j])):
            xs[:, self.slices[j][i]] = pos[:,i]
        return xs

    def initialize_injection(self):
        """
        Initializes the particle injection process.

        It first generates the non-periodic boundary surfaces and the
        corresponding particle position vector slices and indices. It then
        calculates the number of particles to be injected through each
        non-periodic boundary surface. The correct velocity distribution domain
        in the normal and transverse directions for each boundary surface is
        also generated.
        """

        self.Ld_non_periodic = [self.Ld[i] for i in range(self.dim)\
                                if not self.periodic[i]]
        self.d = len(self.Ld_non_periodic)
        self.L = [0]*self.d + self.Ld_non_periodic

        self.index = [i for i in range(self.dim) if not self.periodic[i]]*2
        self.boundary_surfaces()
        self.slices()

        self.surface_area = [np.prod(self.surfaces[i]) for i in range(self.d)]

    def combinations(self, a, k):
        """
        Returns k length subsequences of elements from the input array a.
        Elements are treated as unique based on their position, not on
        their value. So if the input elements are unique, there will be no
        repeated values in each combination.

        Args:
             a: An iterable array
             k: The length of subsequences

        Example:
                a = [2,4,6]
                k = 2
                combs = combinations(a, k)
                combs = [[2,4], [2,6], [4,6]]
        """
        for i in xrange(len(a)):
            if k == 1:
                yield [a[i]]
            else:
                for next in combinations(a[i+1:len(a)], k-1):
                    yield [a[i]] + list(next)

    def boundary_surfaces(self):
        """
        Returns the non-periodic boundary surfaces of the simulation domain.

        Example:
                If periodic = [False, True, False] and Ld = [2, 4, 6], it
                returns those boundary surfaces that are not periodic, i.e.,
                surfaces = [[4,6], [2,4]], where [4,6] corresponds to the
                boundary surface where x = 0 or x = 2, and [2,4] corresponds to
                the boundary surface where z = 0 or z = 6.
        """
        surfaces = list(self.combinations(self.Ld, self.dim-1))
        self.surfaces = [surfaces[-(i+1)] for i in range(self.dim) \
                                         if not self.periodic[i]]

    def slices(self):
        """
        Retruns slices for the random positions of particles for each boundary
        sufrace. xs[slices] gives those components of the particle position that
        corresponds to points on the boundary surface.

        Example:
                If periodic = [False, True, False] and Ld = [2, 4, 6], the
                non-periodic boundary surfaces are [[4,6], [2,4]], where [4,6]
                corresponds to the boundary surface where x = 0 or x = 2, and
                [2,4] corresponds to the boundary surface where z = 0 or z = 6.
                The slices are [[1,2], [0,1]]. Here 0,1,2 correspond to x,y,z
                components of the particle position.

                A point on the boundary surface [4, 6], has a constant x-value,
                (x = 0 or x = 2) but it has a y-value which is a number between
                0 to 4, and similarly it has a z-value which is a number between
                0 to 6. The corresponding slices are [1,2]. Which give the y and
                z components of a randomly chosen particle position, xs, on the
                surface:

                xs[slices[0]]: y-componet of the random point
                xs[slices[1]]: z-componet of the random point
        """
        slices = list(self.combinations(np.arange(0, self.dim), self.dim-1))
        self.slices = [slices[-(i+1)] for i in range(len(slices)) \
                                          if not self.periodic[i]]
        self.slices *= 2
