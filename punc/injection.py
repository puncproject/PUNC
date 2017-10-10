import dolfin as df
import numpy as np
from scipy.special import erfcinv, erfinv, erf, erfc
import sys
import time

__UINT32_MAX__ = np.iinfo('uint32').max


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

    def __init__(self, Vt, dVdt, root=None, transform=None, interval=[0, 1],
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
            self.sp = np.linspace(
                self.lb, self.ub, num=nsp + 1, endpoint=False)[1:]

        if root is not None:
            while len(np.where(np.abs(self.sp - root) < 1e-13)[0]) > 0:
                nsp += 1
                self.sp = np.linspace(self.lb, self.ub, num=nsp,
                                      endpoint=False)[1:]

        if transform is None:
            self.generate_u = lambda n, t, r, Y_i, y_t: [t[i] for i in range(n)
                                                         if r[i] < np.exp(Y_i[i] - y_t[i])]
        else:
            self.generate_u = lambda n, t, r, Y_i, y_t: [transform(t[i])
                                                         for i in range(n)
                                                         if r[i] < np.exp(Y_i[i] - y_t[i])]

        self.y = self.Vt(self.sp)
        self.dy = self.dVdt(self.sp)

        self.lower_hull()

        self.integrand = np.diff(np.exp(-self.Yt)) / (-self.dy)
        while len(np.where(self.integrand < 0)[0]) > 0:
            nsp += 1
            self.sp = np.linspace(self.lb, self.ub, num=nsp,
                                  endpoint=False)[1:]

            self.y = self.Vt(self.sp)
            self.dy = self.dVdt(self.sp)
            self.lower_hull()
            self.integrand = np.diff(np.exp(-self.Yt)) / (-self.dy)

        self.exponentiated_lower_hull()

    def tangent_intersections(self):
        """
        For the list of support points and the corresponding ordinates and the
        derivative of the potential function, it finds the point of intersection
        between the piecewise-linear functions.
        """
        self.z = np.zeros(self.sp.__len__() + 1)
        self.z[0] = self.lb
        self.z[1:-1] = (np.diff(self.y) - np.diff(self.sp * self.dy)) /\
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
        self.Yt = self.dy[[0] + range_N] * (self.z - self.sp[[0] + range_N]) + \
            self.y[[0] + range_N]

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

        self.exp_cdf = np.hstack([0, np.cumsum(self.integrand)])
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
        r = np.random.random(N)
        index = [np.nonzero(self.exp_cdf / self.c_i < i)[0][-1] for i in r]
        t = self.sp[index] + (self.y[index] +
                              np.log(-self.dy[index] * (self.c_i * r - self.exp_cdf[index]) +
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
            n = N - len(vs)
            t, index = self.sample_exp(n)
            y_t = self.Vt(t)
            Y_i = self.y[index] + (t - self.sp[index]) * self.dy[index]

            r = np.random.random(n)
            v = self.generate_u(n, t, r, Y_i, y_t)
            v = np.array(v)
            vs = np.concatenate([vs, v])
        return vs

def create_mesh_pdf(pdf, mesh):

    mesh.init(0, mesh.topology().dim())
    tree = mesh.bounding_box_tree()
    def mesh_pdf(x):
        cell_id = tree.compute_first_entity_collision(df.Point(*x))
        inside_mesh = int(cell_id != __UINT32_MAX__ and cell_id != -1)
        return inside_mesh * pdf(x)

    return mesh_pdf

def random_domain_points(pdf, pdf_max, N, mesh):

    dim = mesh.geometry().dim()
    Ld_min = np.min(mesh.coordinates(), 0)
    Ld_max = np.max(mesh.coordinates(), 0)

    pdf = create_mesh_pdf(pdf, mesh)

    xs = np.array([]).reshape(0, dim)
    while len(xs) < N:
        n = N - len(xs)
        r1 = np.random.uniform(Ld_min, Ld_max, (n, dim))
        r2 = pdf_max * np.random.random(n)
        new_xs = [r1[i, :]
                    for i in range(n) if r2[i] < pdf(r1[i, :])]
        new_xs = np.array(new_xs).reshape(-1, dim)
        xs = np.concatenate([xs, new_xs])
    return xs

def random_facet_points(N, facet_vertices):
    dim = len(facet_vertices)
    xs = np.empty((N, dim))
    for j in range(N):
        xs[j, :] = facet_vertices[0, :]
        for k in range(1, dim):
            r = np.random.random()
            if k == dim - (k - 1):
                r = 1.0 - np.sqrt(r)
            xs[j, :] += r * (facet_vertices[k, :] - xs[j, :])
    return xs

def maxwellian(v_thermal, v_drift, N):
    dim = N[1]
    if v_thermal == 0.0:
        v_thermal = np.finfo(float).eps

    if isinstance(v_drift, (float, int)):
        v_drift = np.array([v_drift] * dim)

    cdf_inv = lambda x, vd=v_drift, vth=v_thermal: vd - \
        np.sqrt(2.) * vth * erfcinv(2 * x)
    w_r = np.random.random((N[0], dim))
    return cdf_inv(w_r)

class Facet(object):
    __slots__ = ['area', 'vertices', 'basis']
    def __init__(self, area, vertices, basis):
        self.area = area
        self.vertices = vertices
        self.basis = basis

class ExteriorBoundaries(list):
    def __init__(self, boundaries, id):
        self.boundaries = boundaries
        self.id = id
        mesh = boundaries.mesh()
        self.g_dim = mesh.geometry().dim()
        self.t_dim = mesh.topology().dim()
        self.num_facets = len(np.where(boundaries.array() == id)[0])


        area = self.get_area(mesh)
        vertices = self.get_vertices()
        basis = self.get_basis(mesh, vertices)

        for i in range(self.num_facets):
            self.append(Facet(area[i],
                              vertices[i*self.g_dim:self.g_dim*(i+1), :],
                              basis[i * self.g_dim:self.g_dim * (i + 1), :]))

    def get_area(self, mesh):
        facet_iter = df.SubsetIterator(self.boundaries, self.id)
        area = np.empty(self.num_facets)
        mesh.init(self.t_dim-1, self.t_dim)
        for i, facet in enumerate(facet_iter):
            cell = df.Cell(mesh, facet.entities(self.t_dim)[0])
            facet_id = list(cell.entities(self.t_dim - 1)).index(facet.index())
            area[i] = cell.facet_area(facet_id)

        return area

    def get_vertices(self):
        facet_iter = df.SubsetIterator(self.boundaries, self.id)
        vertices = np.empty((self.num_facets*self.g_dim, self.g_dim))
        for i, facet in enumerate(facet_iter):
            for j, v in enumerate(df.vertices(facet)):
                vertices[i*self.g_dim+j,:] = np.array([v.x(k)
                                                    for k in range(self.g_dim)])

        return vertices

    def get_basis(self, mesh, vertices):
        facet_iter = df.SubsetIterator(self.boundaries, self.id)
        basis = np.empty((self.num_facets * self.g_dim, self.g_dim))
        for i, facet in enumerate(facet_iter):
            fs = df.Facet(mesh, facet.index())

            basis[i * self.g_dim, :] = vertices[i*self.g_dim, :] -\
                                       vertices[i*self.g_dim+1, :]
            basis[i*self.g_dim, :] /= np.linalg.norm(basis[i*self.g_dim, :])
            basis[self.g_dim*(i+1)-1, :] = -1 * \
                np.array([fs.normal()[i] for i in range(self.g_dim)])
            if (self.g_dim == 3):
                basis[i*self.g_dim + 1, :] = \
                    np.cross(basis[self.g_dim*(i+1)-1, :],
                             basis[i*self.g_dim, :])
        return basis


class Flux(object):
    def __init__(self, v_thermal, v_drift, exterior_bnd, vd_ratio=16):
        self.dim = exterior_bnd.g_dim
        if v_thermal == 0.0:
            self.vth = np.finfo(float).eps
        else:
            self.vth = v_thermal
        if isinstance(v_drift, (float, int)):
            self.vd = np.array([v_drift] * self.dim)
        else:
            self.vd = v_drift
        self.vd_ratio = vd_ratio

        self.vn = self.get_vn(exterior_bnd)
        self.num_particles = self.flux_number(exterior_bnd)

        if all(v == 0 for v in self.vd):
            self.is_drifting = 0
            self.flux_nondrifting()
        else:
            self.is_drifting = 1
            self.flux_drifting(exterior_bnd)

    def get_vn(self, exterior_bnd):
        vn = np.empty((len(exterior_bnd), self.dim))
        for i, facet in enumerate(exterior_bnd):
            for j in range(self.dim):
                vn[i, j] = np.dot(facet.basis[j, :], self.vd)
        return vn

    def flux_number(self, exterior_bnd):
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
        def particle_number(facet, i):
            """
            Calculates the total number of particles to be injected at the
            i-th exterior boundary for the inward flux.
            """
            return facet.area*(self.vth / (np.sqrt(2 * np.pi)) *
                 np.exp(-0.5 * (self.vn[i, self.dim - 1] / self.vth)**2) +
                 0.5 * self.vn[i,self.dim - 1] *
                 (1. + erf(self.vn[i,self.dim - 1] / (np.sqrt(2) * self.vth))))

        N = np.array([particle_number(facet, i) for i, facet in enumerate(exterior_bnd)])
        return N

    def pdf_max_drifting(self, d):
        """
        Finds the root of the cubic equation corresponding to the maximum value
        of the transformed drifting-Maxwellian distribution function.

        Returns:
                The root of the transformed drifting-Maxwellian distribution
                function
        """
        ca = 2.0
        cb = -4.0 - d
        cc = d
        cd = 1.0

        p = (3.0 * ca * cc - cb * cb) / (3.0 * ca * ca)
        q = (2.0 * cb * cb * cb - 9.0 * ca * cb * cc +
             27 * ca * ca * cd) / (27 * ca * ca * ca)

        sqrt_term = np.sqrt(-p / 3.0)
        root = 2.0 * sqrt_term *\
            np.cos(np.arccos(3.0 * q / (2.0 * p * sqrt_term)) /
                   3.0 - 2.0 * np.pi / 3.0)
        return root - cb / (3.0 * ca)

    def cdf_inv_transverse(self, x, vn_i=0):
        """
        The inverse of the cumulative distribution function for a drifting-
        Maxwellian distribution function.

        Arguments:
                index: The velocity component, (0,1,2) -> (x, y, z)
        """
        return vn_i - np.sqrt(2.) * self.vth * erfcinv(2 * x)

    def cdf_inv_flux_nondrifting(self, x):
        """
        The inverse of the cumulative distribution function for the inward flux
        of a nondrifting-Maxwellian distribution function.
        """
        return self.vth * np.sqrt(-2. * np.log(1. - x))

    def flux_nondrifting(self):

        cdf = [None] * self.dim
        cdf[-1] = self.cdf_inv_flux_nondrifting
        for i in range(self.dim - 1):
            cdf[i] = self.cdf_inv_transverse

        self.generator = [lambda N, cdf=cdf[i]: cdf(
            np.random.random(N)) for i in range(self.dim)]

    def flux_drifting(self, exterior_bnd):
        """
        Initializes the optimized rejection sampling for a drifting Maxwellian
        density function.
        """
        num_facets = len(exterior_bnd)
        self.generator = [None] * num_facets * self.dim
        for i, facet in enumerate(exterior_bnd):
            for j in range(self.dim - 1):
                cdf = lambda x, v=facet.vn[j], vth=self.vth: v - \
                    np.sqrt(2.) * vth * erfcinv(2 * x)
                self.generator[i * self.dim +
                               j] = lambda N, cdf=cdf: cdf(np.random.random(N))

            d = facet.vn[self.dim - 1] / self.vth
            if d != 0.0:
                root = self.pdf_max_drifting(d)
                V = lambda t, d=d: -np.log(t) + 3. * np.log(1. - t) +\
                    0.5 * (d - t / (1. - t))**2
                dV = lambda t, d=d: (-1. / t) - (3. / (1. - t)) +\
                    ((t * (1. + d) - d) / (1. - t)**3)
                transform = lambda t, vth=self.vth: vth * t / (1. - t)

                ors = ORS(V, dV, root=root, transform=transform)
                self.generator[i * self.dim + self.dim -
                               1] = lambda N, ors=ors: ors.sample(N)
            else:
                def cdf(x): return self.vth * np.sqrt(-2. * np.log(1. - x))
                self.generator[i * self.dim + self.dim -
                               1] = lambda N, cdf=cdf: cdf(np.random.random(N))

def inject(pop, exterior_bnd, dt):
    dim = pop.g_dim
    for specie in range(len(pop.species)):
        xs = np.array([]).reshape(0, dim)
        vs = np.array([]).reshape(0, dim)

        flux = pop.flux[specie]
        n_p = pop.plasma_density[specie]
        for i, facet in enumerate(exterior_bnd):
            N = int(n_p*dt*flux.num_particles[i])

            if np.random.random() < n_p * dt * flux.num_particles[i] - N:
                N += 1
            count = 0
            while count < N:
                n = N - count
                new_xs = random_facet_points(n, facet.vertices)

                new_vs = np.empty((n, dim))
                for j in range(dim):
                    new_vs[:, j] = flux.generator[flux.is_drifting*i*dim + j](n)

                new_vs = np.dot(new_vs, facet.basis)

                w_random = np.random.random(len(new_vs))
                for k in range(dim):
                    new_xs[:, k] += dt * w_random * new_vs[:, k]

                for j in range(n):
                    x = new_xs[j, :]
                    v = new_vs[j, :]
                    cell_id = pop.locate(x)
                    if cell_id != __UINT32_MAX__ and cell_id != -1:
                        xs = np.concatenate([xs, x[None, :]])
                        vs = np.concatenate([vs, v[None, :]])
                        count += 1

        pop.add_particles_of_specie(specie, xs, vs)
