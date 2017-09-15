from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from scipy.special import erfcinv, erfinv, erf, erfc

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

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
                transform : transformation of the pdf to its original form
                Ld        : the domain size of the simulation box
                interval  : 1D interval where the pdf is defined
    
    Notes:
          If the simulation domain, Ld, is specified, the generated random
          numbers represent random particle positions according to the given
          pdf. If Ld is None, then a 1D interval should be given to generate 
          random numbers that may represent random velocities according to the
          given pdf.   
    """
    def __init__(self, pdf, pdf_max=1, transform=None, Ld=None, interval=[0,1]):
        """
        Arguments:
                pdf       : probability density function
                pdf_max   : the maximum value of the pdf
                transform : transformation of the pdf to its original form
                Ld        : the domain size of the simulation box
                interval  : 1D interval where the pdf is defined
        """
        self.pdf_max = pdf_max
        
        if Ld is not None:
            Ld = np.array(Ld)
            self.dim = len(Ld)
            self.generate_u = lambda n: np.random.rand(n,self.dim)*Ld
            self.empty_vs = np.array([]).reshape(0,self.dim)
            self.get_array = lambda v: np.array(v).reshape(-1,self.dim)
        else:
            lb = interval[0]
            ub = interval[1]
            self.generate_u = lambda n: np.random.uniform(lb,ub,n)
            self.empty_vs = np.array([])
            self.get_array = lambda v: np.array(v)

        if transform is not None:
            self.generate_v = lambda n, r, s: [transform(r[i]) \
                                                    for i in range(n) \
                                                        if s[i] < pdf(r[i])]
        else:
            self.generate_v = lambda n, r, s: [r[i] for i in range(n) \
                                                        if s[i] < pdf(r[i])]

    def sample(self, N):
        """
        Samples N random numbers, vs, from the pdf.

        Arguments:
                  N: Number of random numbers to be generated

        Returns:
                 vs: Array of N random numbers generated from the pdf
        """
        vs = self.empty_vs

        while len(vs) < N:
            n  = N - len(vs)
            u  = self.generate_u(n)
            r  = self.pdf_max*np.random.rand(n)
            v  = self.generate_v(n, u, r)
            v  = self.get_array(v)
            vs = np.concatenate([vs,v])
        return vs

class ORS(object):
    """
    Optimized rejection sampling for strictly convex potential functions.

    Given a strictly convex potential function, V(t), of a probability density
    function, p(t), i.e.,

                        V(t) = -log(p(t)),

    it generates random numbers whose distribution function is p(t).

    Attributes:
                Vt     : potential function, V(t)
                dVdt   : derivative of potential function, V(t)
                root   : The roots of the derivative of potential function
                transform : a function to transform the pdf to its original form
                interval : domian of the potential function, V(t)
                sp     : list of support points for construction of piecewise
                         linear functions
                
                nsp    : total number of support points to be used to construct
                         piecewise-linear functions
                y      : ordinates corresponding to support points
                dy     : derivatives of the potential function at support points

    """
    def __init__(self, Vt, dVdt, root=None, transform=None, interval=[0,1], 
                 sp=None, nsp=50):
        """
        If roots of the derivative of the potential function is specified, the 
        support points are constructed in such a way that no support point is
        equal to one of the roots. This is crucial for generating piece-wise 
        exponential lower hulls, as the derivative of the potential function 
        appears in the denominator of the expression for these lower hulls.   
        """
        self.Vt = Vt
        self.dVdt = dVdt
        self.lb = interval[0]
        self.ub = interval[1]
        self.sp = sp
        
        if sp is None:
            self.sp = np.linspace(self.lb, self.ub, num=nsp+1, endpoint=False)[1:]

        if root is not None:
            while len(np.where(np.abs(self.sp-root)<1e-13)[0])>0:
                nsp += 1
                self.sp = np.linspace(self.lb, self.ub, num=nsp, 
                                      endpoint=False)[1:]

        if transform is None:
            self.generate_u = lambda n, t, r, Y_i, y_t: [t[i] for i in range(n)\
                                              if r[i] < np.exp(Y_i[i] - y_t[i])]
        else:
            self.generate_u = lambda n, t, r, Y_i, y_t: [transform(t[i]) \
                                            for i in range(n) \
                                              if r[i] < np.exp(Y_i[i] - y_t[i])]

        self.y = self.Vt(self.sp)
        self.dy = self.dVdt(self.sp)
   
        self.lower_hull()

        self.integrand = np.diff(np.exp(-self.Yt))/(-self.dy)
        while len(np.where(self.integrand<0)[0])>0:
            nsp += 1
            self.sp = np.linspace(self.lb, self.ub, num=nsp, 
                                      endpoint=False)[1:]
            
            self.y = self.Vt(self.sp)
            self.dy = self.dVdt(self.sp)
            self.lower_hull()
            self.integrand = np.diff(np.exp(-self.Yt))/(-self.dy)

        self.exponentiated_lower_hull()

    def tangent_intersections(self):
        """
        For the list of support points and the corresponding ordinates and the
        derivative of the potential function, it finds the point of intersection
        between the piecewise-linear functions.
        """
        self.z = np.zeros(self.sp.__len__() + 1)
        self.z[0] = self.lb
        self.z[1:-1] = (np.diff(self.y) - np.diff(self.sp*self.dy))/\
                       -np.diff(self.dy)
        self.z[-1] = self.ub

    def lower_hull(self):
        """
        Constructs the lower hull, Y(t), of the potential by piecewise-linear
        functions at the intersection between the piecewise-linear functions.
        """        
        self.tangent_intersections()

        N = self.y.__len__()
        range_N = list(range(N))
        self.Yt = self.dy[[0]+range_N]*(self.z-self.sp[[0]+range_N]) + \
                  self.y[[0]+range_N]

    def exponentiated_lower_hull(self):
        """
        Constructs a piecewise-exponential lower hull of the potential.
        Moreover, it calculates the cumulative distribution function for the
        piecewise-exponential density function.

        Here, 'integrand' is the integrand in the expression for the cdf
        'exp_cdf' is the cumulative distribution function for the
        piecewise-exponential density function. 'c_i' is the normalization
        factor of the exponentiated lower hull.
        """

        self.exp_cdf = np.hstack([0,np.cumsum(self.integrand)])
        self.c_i = self.exp_cdf[-1]

    def sample_exp(self, N):
        """
        The inverse function of the cumulative distribution function for the
        piecewise-exponential function.

        First samples a uniform random number, r, and finds the index, i, for
        the largest intersection point, z_i, between the piecewise-linear
        functions, such that

                        cdf(z_i) < r,

        where cdf is the cumulative distribution function for
        piecewise-exponential function. Then, it finds the t-value by using the
        analytical expression of the inverse distribution function.

        Returns:
                t     : generated t-value from the inverse cdf of the
                        piecewise-exponential function
                index : the index of the largest z_i
        """        
        r = np.random.rand(N)
        index = [np.nonzero(self.exp_cdf/self.c_i < i)[0][-1] for i in r]
        t = self.sp[index] + (self.y[index] +\
             np.log(-self.dy[index]*(self.c_i*r - self.exp_cdf[index]) +\
             np.exp(-self.Yt[index]))) / (-self.dy[index])

        return t, index

    def sample(self, N):
        """
        Samples N random numbers, vs, from the density distribution function

                        p(v) = exp(-V(v))

        Arguments:
                   N: number of random numbers to be sampled

        Returns:
                  vs: Array of N sampled random numbers

        """
        vs = np.array([])

        while len(vs) < N:
            n  = N - len(vs)
            t, index = self.sample_exp(n)
            y_t = self.Vt(t)
            Y_i = self.y[index] + (t-self.sp[index])*self.dy[index]

            r  = np.random.rand(n)
            v = self.generate_u(n, t, r, Y_i, y_t)
            v  = np.array(v)
            vs = np.concatenate([vs,v])
        return vs

class Maxwellian(object):
    """
    Generates both drifting and non-drifting Maxwellian velocities for loading
    and injection of particles.

    Attributes:
            v_thermal (array) : Thermal velocity of particles
            v_drift (array)   : Drift velocity of particles
            periodic (list)   : A list containing which directions are
                               periodic or not.
            sampling_method (string): Currently implemented methods are
                                      standard rejection sampling, 'srs', and
                                      optimized rejection sampling, 'ors'.
    """
    def __init__(self, v_thermal, v_drift, periodic, sampling_method='ors', 
                       vd_ratio=16):
        """
        Arguments:
            v_thermal (array) : Thermal velocity of particles
            v_drift (array)   : Drift velocity of particles
            periodic (list)   : A list containing which directions are
                               periodic or not.
            sampling_method (string): currently implemented methods are
                                    standard rejection sampling, 'srs', and
                                    optimized rejection sampling, 'ors'.
        """
        self.periodic = np.array(periodic)
        self.dim = len(self.periodic)
        if v_thermal == 0.0:
            self.vth = np.finfo(float).eps
        else:
            self.vth = v_thermal
        if isinstance(v_drift, (float, int)):
            self.vd = np.array([v_drift]*self.dim)    
        else:
            self.vd = v_drift

        self.sampling_method = sampling_method
        self.vd_ratio = vd_ratio

        self.initialize_loading()
        self.initialize_injection()
        
    def initialize_loading(self):
        """
        Initializes the loading process.
        """
        self.loader = [None]*self.dim
        for i in range(self.dim):
            if self.vd[i] == 0:
                cdf = self.cdf_inv_nondrifting()
            else:
                cdf = self.cdf_inv_drifting(i)
            self.loader[i] = lambda N, cdf=cdf: self.generate(cdf, N)

    def initialize_injection(self):
        """
        Initializes the injection process.
        """
        self.dim_nonperiodic = len(np.where(self.periodic==False)[0])
        self.dim2 = self.dim**2
        self.len_cdf = self.dim*self.dim_nonperiodic
        self.generator = [None]*2*self.len_cdf
        self.normal_vec = np.eye(self.dim, dtype=bool).flatten()

        if all(v==0 for v in self.vd):
            self.inject_nondrifting()
        else:
            self.vd_nonzero = self.vd[self.vd.nonzero()[0]]
            self.dim_drift = len(self.vd_nonzero)
            if self.sampling_method == 'srs':
                self.initialize_srs()
            elif self.sampling_method == 'ors':
                self.initialize_ors()
            self.inject_drifting()

    def initialize_ors(self):
        """
        Initializes the optimized rejection sampling for a drifting Maxwellian
        density function.
        """
        self.rs = [None]*2*self.dim_drift

        for i, vd_i in enumerate(self.vd_nonzero):
            d = vd_i/self.vth
            j = 0
            for s in [1, -1]:
                root = self.pdf_max_drifting(d, s)
                V = lambda t, d=d, s=s: -np.log(t) + 3.*np.log(1.-t) +\
                                               0.5*(d - s*t/(1.-t))**2
                dV = lambda t, d=d, s=s: (-1./t) - (3./(1.-t)) +\
                                      (s*(t*(s+d)-d)/(1.-t)**3)
                transform = lambda t, vth=self.vth, s=s: s*vth*t/(1.-t)
                self.rs[i+j*self.dim_drift] = \
                                      ORS(V, dV, root=root, transform=transform)
                j += 1

    def initialize_srs(self):
        """
        Initializes the standard rejection sampling for a drifting Maxwellian
        density function.
        """
        self.rs = [None]*2*self.dim_drift

        for i, vd_i in enumerate(self.vd_nonzero):
            d = vd_i/self.vth
            j = 0
            for s in [1, -1]:
                root = self.pdf_max_drifting(d, s)
                pdf = self.pdf_flux_drifting_transformed(d, s)
                pdf_max = pdf(root)
                transform = lambda t, vth=self.vth, s=s: s*vth*t/(1.-t)
                self.rs[i+j*self.dim_drift] = \
                                          SRS(pdf, pdf_max, transform=transform)
                j += 1

    def pdf_max_drifting(self, d, s):
        """
        Finds the root of the cubic equation corresponding to the maximum value
        of the transformed drifting-Maxwellian distribution function.

        Returns:
                The root of the transformed drifting-Maxwellian distribution 
                function
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
        return root - cb/(3.0*ca)

    def pdf_flux_drifting_transformed(self, d, s):
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

    def pdf_maxwellian(self, index):
        """
        Probability distribution function for Maxwellian velocities.

        Arguments:
                index: The velocity component, (0,1,2) -> (x, y, z)

        """
        pdf = lambda t, i=index: 1.0/(np.sqrt(2*np.pi)*self.vth)*\
                        np.exp(-0.5*((t-self.vd[i])**2)/(self.vth**2))
        return pdf

    def pdf_flux_nondrifting(self, s):
        """
        Velocity probability distribution function for the particle flux 
        into the simulation domain for particles having a non-drifting 
        Maxwellian distribution outside the domain.

        Arguments:
                s: 1 for inward flux, and -1 for backward flux
        """
        pdf = lambda t, s=s: s*(t/self.vth**2)*np.exp(-0.5*(t/self.vth)**2)
        return pdf

    def pdf_flux_drifting(self, index, s):
        """
        Velocity probability distribution function for the particle flux 
        into the simulation domain for particles having a drifting 
        Maxwellian distribution outside the domain.

        Arguments:
                index: The velocity component, (0,1,2) -> (x, y, z)
                s: 1 for inward flux, and -1 for backward flux
        """
        i = index      
        def pdf(t):
            return (s*t*np.exp(-0.5*((t-self.vd[i])**2)/(self.vth**2)))/\
                   (self.vth**2*np.exp(-0.5*(self.vd[i]/self.vth)**2) + \
                   s*np.sqrt(0.5*np.pi)*self.vd[i]*self.vth * \
                   (1. + s*erf(self.vd[i]/(np.sqrt(2)*self.vth))))

        return pdf

    def cdf_inv_nondrifting(self):
        """
        The inverse of the cumulative distribution function for a nondrifting-
        Maxwellian distribution function.
        """
        cdf = lambda x, vth=self.vth: np.sqrt(2.)*vth*erfinv(2*x-1.)
        return cdf

    def cdf_inv_drifting(self, index):
        """
        The inverse of the cumulative distribution function for a drifting-
        Maxwellian distribution function.

        Arguments:
                index: The velocity component, (0,1,2) -> (x, y, z)
        """
        cdf = lambda x,vd=self.vd[index],vth=self.vth: vd -np.sqrt(2.)*vth*\
                                                                 erfcinv(2*x)
        return cdf

    def cdf_inv_flux_drifting(self, index):
        """
        An approximation for the inverse cumulative distribution function for 
        the flux of a drifting-Maxwellian distribution function. This 
        approximation is only valid for large drift to thermal velocity ratios.
        If vd/vth > 16, this approximation gives dissent results.  
        """

        m = (self.vd[index]**2+self.vth**2)/self.vd[index]
        cdf = lambda x, m=m, vth=self.vth: m + np.sqrt(2.)*vth*erfinv(2.*x-1)
        return cdf

    def cdf_inv_inward_flux_nondrifting(self):
        """
        The inverse of the cumulative distribution function for the inward flux
        of a nondrifting-Maxwellian distribution function.
        """
        return lambda x, vth=self.vth: vth*np.sqrt(-2.*np.log(1.-x))

    def cdf_inv_backward_flux_nondrifting(self):
        """
        The inverse of the cumulative distribution function for the backward
        flux of a drifting-Maxwellian distribution function.
        """
        return lambda x, vth=self.vth: -vth*np.sqrt(-2.*np.log(x))

    def inject_nondrifting(self):
        """
        Generates all the inverse cumulative distribution functions for
        nondrifting-Maxwellian distributed particles, needed to generate
        velocities for the injection of particles through the exterior surface
        boundaries.
        """
        cdf = [None]*2*self.dim2

        cdf_inv = [self.cdf_inv_nondrifting(), 
                   self.cdf_inv_inward_flux_nondrifting(),
                   self.cdf_inv_backward_flux_nondrifting()]

        maxwellian_flux = lambda N, cdf=cdf_inv[0]: self.generate(cdf, N)
        inward_flux     = lambda N, cdf=cdf_inv[1]: self.generate(cdf, N)
        backward_flux   = lambda N, cdf=cdf_inv[2]: self.generate(cdf, N)

        for i in range(self.dim):
            for j in range(self.dim):
                k = i*self.dim + j
                if self.normal_vec[k]:
                    cdf[k] = inward_flux
                    cdf[k+self.dim2] = backward_flux
                else:
                    cdf[k] = maxwellian_flux
                    cdf[k+self.dim2] = maxwellian_flux

        k = 0
        for i in range(self.dim):
            if not self.periodic[i]:
                for j in range(i*self.dim, (i+1)*self.dim):
                    self.generator[k] = cdf[j]
                    self.generator[k+self.len_cdf] = cdf[j+self.dim2]
                    k += 1

    def inject_drifting(self):
        """
        Generates all the inverse cumulative distribution functions or the
        sampling method for drifting-Maxwellian distributed particles, needed to
        generate velocities for the injection of particles through the exterior
        surface boundaries.
        """
        cdf = [None]*2*self.dim2

        m = 0
        for i in range(self.dim):
            for j in range(self.dim):
                k = i*self.dim + j
                if self.vd[j] != 0:
                    if self.normal_vec[k]:
                        if self.vd[j]/self.vth < self.vd_ratio:
                            cdf[k] = lambda N, index=m:self.rs[index].sample(N)
                        else:
                            cdf[k] = lambda N, index=(k%self.dim):\
                            self.generate(self.cdf_inv_flux_drifting(index), N)
                        cdf[k+self.dim2] = lambda N, index=(m+self.dim_drift):\
                                                        self.rs[index].sample(N)
                        m += 1
                    else:
                        cdf[k] = lambda N, index=(k%self.dim):\
                        self.generate(self.cdf_inv_drifting(index), N)
                        cdf[k+self.dim2] = lambda N, index=(k%self.dim):\
                        self.generate(self.cdf_inv_drifting(index), N)
                else:
                    if self.normal_vec[k]:
                        cdf[k] = lambda N:\
                        self.generate(self.cdf_inv_inward_flux_nondrifting(), N)
                        cdf[k+self.dim2] = lambda N:\
                        self.generate(self.cdf_inv_backward_flux_nondrifting(), 
                                      N)
                    else:
                        cdf[k] = lambda N:\
                        self.generate(self.cdf_inv_nondrifting(), N)
                        cdf[k+self.dim2] = lambda N:\
                        self.generate(self.cdf_inv_nondrifting(), N)

        k = 0
        for i in range(self.dim):
            if not self.periodic[i]:
                for j in range(i*self.dim, (i+1)*self.dim):
                    self.generator[k] = cdf[j]
                    self.generator[k+self.len_cdf] = cdf[j+self.dim2]
                    k += 1

    def generate(self, cdf_inv, N):
        """
        Samples and returns N velocities from the distribution function 
        specified by inverse cumulative distribution function, cdf_inv.

        Arguments:
                cdf_inv : Inverse cdf
                N       : Number of random velocities
        """
        w_r = np.random.rand(N)
        vs = [cdf_inv(w_r[i]) for i in range(N)]
        return np.array(vs)

    def sample(self, N, k=0):
        """
        Samples and returns N velocities for injection of particles into the 
        simulation domain through the k-th exterior boundary surface. 

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
            vs[:,j] = self.generator[j+k*self.dim](N)
        return vs

    def load(self, N):
        """
        Samples and returns N velocities for particle velocity loading.

        Arguments:
                N (int): Number of velocities to be sampled.

        Returns:
                vs (array): Array containing N sampled velocities.
        """
        vs = np.empty((N, self.dim))
        for j in range(self.dim):
            vs[:,j] = self.loader[j](N)
        return vs

    def flux_number(self, plasma_density, surface_area, dt):
        """
        For each boundary surface that are not periodic, calculates the number
        of total particles, based on a drifting or nondrifting-Maxwellian
        distribution function, to be injected through the surface.

        Arguments:
            plasma_density    : Plasma density
            surface_area      : A list containing all the surface areas of the
                                exterior boundaries
            dt                :  Simulation time step

        Returns:
                N : A list containing the total number of particles to be 
                    injected at each non-periodic surface boundary.
        """
        def inward_particle_flux_number(i):
            """
            Calculates the total number of particles to be injected at the 
            i-th exterior boundary for the inward flux.
            """
            N = plasma_density*surface_area[i%self.dim_nonperiodic]*dt*\
                (self.vth/(np.sqrt(2*np.pi))*\
                np.exp(-0.5*(self.vd[i]/self.vth)**2)+\
                0.5*self.vd[i]*(1. + erf(self.vd[i]/(np.sqrt(2)*self.vth))))
            return N

        def backward_particle_flux_number(i):
            """
            Calculates the total number of particles to be injected at the i-th 
            exterior boundary for the backward flux.
            """
            N = np.abs(plasma_density*surface_area[i%self.dim_nonperiodic]*dt*\
                       (0.5*self.vd[i]*erfc(self.vd[i]/(np.sqrt(2)*self.vth)) -\
             self.vth/(np.sqrt(2*np.pi))*np.exp(-0.5*(self.vd[i]/self.vth)**2)))
            return N
        
        N = np.empty(2*self.dim_nonperiodic)
        j = 0
        for i in range(self.dim):
            if not self.periodic[i]:
                N[j] = inward_particle_flux_number(i)
                N[j+self.dim_nonperiodic] = backward_particle_flux_number(i)
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
    time step and r is a random number. The choice of the random number r is to
    account for the fact that the particle could have entered the domain with a
    uniform probability in the range [xs-vs*dt, xs]. After assigning the
    position xs + r*vs*dt, only the particles that have entered the simulation
    domain are kept and injected into the domain. The process of assigning
    positions and velocities continues until N particles have entered the
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
    def __init__(self, pop, specie, dt):
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
        self.vel = pop.vel[specie]
        self.v_drift = pop.species[specie].v_drift
        self.v_thermal = pop.species[specie].v_thermal
        self.Ld = pop.Ld
        self.periodic = pop.periodic
        self.plasma_density = pop.plasma_density[specie]
        self.dim = len(self.Ld)
        self.dt = dt

        self.initialize_injection()

        self.num_particles = self.vel.flux_number(self.plasma_density, 
                                                  self.surface_area, dt)

        print("N: ", self.num_particles)

    def inject(self):
        """
        For each non-periodic boundary surface, injects N particles with
        velocities calculated from a predefined probability distribution
        function.

        N is a real number, therefore to have a statistically correct number of
        injected particles, at each time step a random number r is generated,
        and if r < N - int(N), then the number of injected particles is
        incremented by one, i.e., number of injected particles N' = int(N) + 1.
        If r > N - int(N), then the number of injected particles is N' = int(N).

        For each non-periodic boundary surface and for each of the N' particles
        random positions and velocities are generated consistent with the
        predefined probability distribution functions for the simulation domain.

        The particles are then injected into the simulation domain by assigning
        the position xs + r*vs*dt, where r is a random number to account for the
        fact that the particle could have entered the domain with a uniform
        probability in the range [xs-vs*dt, xs].

        The process of assigning positions and velocities continues until N'
        particles have entered the simulation domain, i.e., xs + r*vs*dt is
        inside the simulation domain.
        """
        for j in range(2*self.dim_nonperiodic):
            N = int(self.num_particles[j])
            if np.random.rand() < self.num_particles[j] - N:
                N += 1
            xs = np.array([]).reshape(0, self.dim)
            vs = np.array([]).reshape(0, self.dim)
            while len(xs) < N:
                n = N-len(xs)
                new_xs = self.sample_positions(j, n)
                new_vs = self.vel.sample(n, j)
                w_random = np.random.rand(len(new_vs))

                for i in range(self.dim):
                    new_xs[:,i]  += self.dt*w_random*new_vs[:,i]

                for i in range(len(new_xs)):
                    x = new_xs[i]
                    v = new_vs[i]
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
        self.dim_nonperiodic = len(self.Ld_non_periodic)
        self.L = [0]*self.dim_nonperiodic + self.Ld_non_periodic

        self.index = [i for i in range(self.dim) if not self.periodic[i]]*2

        area = np.prod(self.Ld)
        self.surface_area = [area/self.Ld[i] for i in range(self.dim) \
                                            if not self.periodic[i]]
        self.slices()

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
        for i in range(len(a)):
            if k == 1:
                yield [a[i]]
            else:
                for next in combinations(a[i+1:len(a)], k-1):
                    yield [a[i]] + list(next)

    def slices(self):
        """
        Retruns slices for the random positions of particles for each boundary
        surface. xs[slices] gives those components of the particle position that
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

class Move:
    def __init__(self, pop, dt):
        self.pop = pop
        self.dt = dt
        self.periodic = self.pop.periodic
        self.Ld = self.pop.Ld
        self.dim = self.pop.g_dim

        self.periodic_indices = [i for i in range(self.dim) 
                                       if self.periodic[i]]  
        self.nonperiodic_indices = [i for i in range(self.dim) 
                                          if not self.periodic[i]]

        if len(self.periodic_indices) == self.dim:
            self.move = self.move_periodic
        elif len(self.nonperiodic_indices) == self.dim:
            self.move = self.move_nonperiodic
        else:
            self.move = self.move_mixed_bnd

    def move_periodic(self):

        for cell in self.pop:
            for particle in cell:
                particle.x += self.dt*particle.v
                particle.x %= self.Ld
    
    def move_nonperiodic(self):

        for cell in self.pop:
            for particle in cell:
                particle.x += self.dt*particle.v

    def move_mixed_bnd(self):

        for cell in self.pop:
            for particle in cell:
                particle.x += self.dt*particle.v
                particle.x[self.periodic_indices] %= self.Ld[self.periodic_indices]

