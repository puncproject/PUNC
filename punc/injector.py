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

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from scipy.special import erfcinv, erf
from functools import reduce
from scipy.special import gamma

__UINT32_MAX__ = np.iinfo('uint32').max

def shifted_maxwellian(vth, vd):
    """
    Shifted-Maxwellian velocity distribution.
    Parameteres: 
                vth - Thermal speed (scalar)
                vd  - Drift velocity (vector)
    """
    D = len(vd)
    def vdf(v):
        return (1.0 / ((np.sqrt(2 * np.pi * vth**2))**D)) *\
            np.exp(-0.5 * np.array(reduce(np.add, [(v[i] - vd[i])**2 \
                                    for i in range(D)])) / (vth**2))
    return vdf 

def kappa(vth, vd, k):
    """
    Kappa velocity distribution.
    Parameteres: 
        vth - Thermal speed (scalar)
        vd  - Drift velocity (vector)
        k   - Spectral index for Kappa vdf
    """
    D = len(vd)
    def vdf(v):
        return 1.0 / ((np.pi*(2*k-3.)*vth**2)**(D/2.0)) *\
        ((gamma(k + 0.5*(D-1.0))) / (gamma(k - 0.5))) *\
        (1. + np.array(reduce(np.add, [(v[i] - vd[i])**2\
        for i in range(D)])) / ((2*k-3.)*vth**2))**(-(k + 0.5*(D-1.)))
    return vdf

class ORS(object):
    """
    Optimized rejection sampling
    """
    def __init__(self, pdf, cutoffs, nsp=50):
        self.pdf = pdf
        self.dim = len(cutoffs)

        if isinstance(cutoffs[0], (np.int, np.float)):
            cutoffs = [cutoffs]

        nsp = [nsp] * self.dim
        df = np.diff(cutoffs)
        for i in range(1, self.dim):
            nsp[i] = nsp[i - 1] * df[i][0] / df[i - 1][0]

        midpoints = [(cutoffs[i][1] + cutoffs[i][0])/2.0 \
                                                      for i in range(self.dim)]

        ind = [None] * self.dim
        found = [False] * self.dim
        while all(found) is False:
            points = np.array([np.linspace(*cutoffs[i], nsp[i], retstep=True)\
                                                    for i in range(self.dim)])
            for i in range(self.dim):
                ind[i] = np.where(points[i][0] == midpoints[i])[0]
                if len(ind[i]) == 0:
                    found[i] = False
                    nsp[i] = nsp[i] + 1
                else:
                    found[i] = True

        self.dv = [points[i][1] for i in range(self.dim)]
        self.volume = np.prod(self.dv)

        sp = [points[i][0] for i in range(self.dim)]
        self.sp = np.array(np.meshgrid(*sp, indexing='ij'))

        self.build_pdf()

    def build_pdf(self):
        f_sp = self.pdf(self.sp)
        f_sp[np.where(f_sp < 0)] = 0

        u_slice = [slice(0, None), slice(0, None), slice(1, None)]
        l_slice = [slice(0, None), slice(0, None), slice(0, -1)]
        u_sl = [None] * self.dim
        l_sl = [None] * self.dim
        for i in range(self.dim):
            u_sl[i] = u_slice[-(i + 1):]
            l_sl[i] = l_slice[-(i + 1):]

        pdf_max = np.maximum(f_sp[u_sl[0]], f_sp[l_sl[0]])
        for i in range(1, self.dim):
            pdf_max = np.maximum(pdf_max[u_sl[i]], pdf_max[l_sl[i]])

        integral = self.volume * pdf_max
        w = integral / np.sum(integral)

        self.pdf_max = pdf_max.flatten()
        self.w_s = w.shape
        w = w.flatten()
        self.weights = np.cumsum(w)

    def sample_pdf(self, n):
        r = np.random.rand(n)
        inds = np.searchsorted(self.weights, r, side='right')
        index = np.unravel_index(inds, self.w_s)

        vs = np.array([self.sp[i][index] + self.dv[i] * np.random.rand(n)
                       for i in range(self.dim)]).T

        pdf_vs = self.pdf_max[inds] * np.random.rand(n)
        return vs, pdf_vs

    def sample(self, N):
        vs = np.array([]).reshape(-1, self.dim)
        while len(vs) < N:
            n = N - len(vs)
            vs_new, p_vs_new = self.sample_pdf(n)
            pdf_vs_new = self.pdf(vs_new.T)
            vs_new = vs_new[np.where(p_vs_new < pdf_vs_new)]
            vs = np.concatenate([vs, vs_new])
        return vs

def locate(mesh, x):
    '''
    Returns the cell id containing the point x in the mesh.
    Returns -1 if the point is not in the mesh.
    mesh.init(0, mesh.topology().dim())
    must be invoked sometime on the mesh before this function can be used.
    '''
    tree = mesh.bounding_box_tree()
    cell_id = tree.compute_first_entity_collision(df.Point(x))

    # cell_id is either -1 or max value of uint32 if the point x is outside
    # the mesh.
    if cell_id == np.iinfo('uint32').max: cell_id = -1
    return cell_id

def create_mesh_pdf(pdf, mesh):

    mesh.init(0, mesh.topology().dim())
    def mesh_pdf(x):
        inside_mesh = locate(mesh,x) >= 0
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
    __slots__ = ['area', 'vertices', 'basis', 'normal']
    def __init__(self, area, vertices, basis, normal):
        self.area = area
        self.vertices = vertices
        self.basis = basis
        self.normal = normal

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
        normal = self.get_normal(mesh)

        for i in range(self.num_facets):
            self.append(Facet(area[i],
                              vertices[i*self.g_dim:self.g_dim*(i+1), :],
                              basis[i*self.g_dim:self.g_dim*(i+1), :],
                              normal[i]))

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
                vertices[i*self.g_dim+j,:] = v.point().array()[:self.g_dim]

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
                np.array([fs.normal()[j] for j in range(self.g_dim)])
            if (self.g_dim == 3):
                basis[i*self.g_dim + 1, :] = \
                    np.cross(basis[self.g_dim*(i+1)-1, :],
                             basis[i*self.g_dim, :])
        return basis

    def get_normal(self, mesh):
        facet_iter = df.SubsetIterator(self.boundaries, self.id)
        normal = np.empty((self.num_facets, self.g_dim))
        for i, facet in enumerate(facet_iter):
            fs = df.Facet(mesh, facet.index())
            normal[i, :] = -1 * \
                np.array([fs.normal()[j] for j in range(self.g_dim)])
        return normal

def flux(vth, vd, k, vdf_type, vdf, cutoffs, nsp, ext_bnd):
    
    num_particles, generator = [None] * len(ext_bnd), [None] * len(ext_bnd)
    D = len(cutoffs)

    if vdf_type=='maxwellian':
        for i, facet in enumerate(ext_bnd):
            n = facet.normal
            vdn = np.dot(n, vd)
            num_particles[i] = facet.area * (vth / (np.sqrt(2 * np.pi)) *\
                                np.exp(-0.5 * (vdn/vth)**2) +\
                                0.5*vdn*(1. + erf(vdn / (np.sqrt(2) *vth))))
            
            vdf_flux = lambda x, n=n, vdf=vdf, D=D: reduce(np.add, [x[i] * n[i]\
                                                    for i in range(D)]) * vdf(x)

            generator[i] = ORS(vdf_flux, cutoffs, nsp=nsp)
    elif vdf_type=='kappa':
        for i, facet in enumerate(ext_bnd):
            n = facet.normal
            num_particles[i] = facet.area*(vth/(np.sqrt(2.*np.pi)))*\
                               np.sqrt(k - 1.5)*gamma(k - 1.)/gamma(k - 0.5)

            vdf_flux = lambda x, n=n, vdf=vdf, D=D: reduce(np.add, [x[i] * n[i]\
                                                    for i in range(D)]) * vdf(x)

            generator[i] = ORS(vdf_flux, cutoffs, nsp=nsp)
    else:
        from scipy.integrate import nquad
        for i, facet in enumerate(ext_bnd):
            n = facet.normal
            if D == 1:
                pdf_flux = lambda x, n=n, vdf=vdf: x*n*vdf(x)\
                                                   if x * n >= 0.0 else 0
            elif D == 2:
                vdf_flux = lambda x, y, n=n, vdf=vdf:\
                                           (x * n[0] + y * n[1]) * vdf([x, y]) \
                                          if (x * n[0] + y * n[1]) >= 0 else 0.0
            elif D == 3:
                vdf_flux = lambda x, y, z, n=n, vdf=vdf:\
                              (x * n[0] + y * n[1] + z * n[2]) * vdf([x, y, z])\
                              if (x * n[0] + y * n[1] + z * n[2]) >= 0 else 0.0

            num_particles[i] = facet.area * nquad(vdf_flux, cutoffs)[0]

            vdf_flux = lambda x, n=n, vdf=vdf, D=D: reduce(np.add, [x[i]*n[i] \
                                                    for i in range(D)]) * vdf(x)

            generator[i] = ORS(vdf_flux, cutoffs, nsp=nsp)
    return (num_particles, generator)

def inject_particles(pop, species, exterior_bnd, dt):
    D = pop.g_dim
    for s in species:
        xs = np.array([]).reshape(0, D)
        vs = np.array([]).reshape(0, D)

        num_particles = s.flux[0]
        flux = s.flux[1]
        n_p = s.n
        for i, facet in enumerate(exterior_bnd):
            N = int(n_p*dt*num_particles[i])

            if np.random.random() < n_p * dt * num_particles[i] - N:
                N += 1
            count = 0
            while count < N:
                n = N - count
                new_xs = random_facet_points(n, facet.vertices)
                new_vs = flux[i].sample(n)

                w_random = np.random.random(len(new_vs))
                for k in range(D):
                    new_xs[:, k] += dt * w_random * new_vs[:, k]

                for j in range(n):
                    x = new_xs[j, :]
                    v = new_vs[j, :]
                    cell_id = pop.locate(x)
                    if cell_id >= 0:
                        xs = np.concatenate([xs, x[None, :]])
                        vs = np.concatenate([vs, v[None, :]])
                    count += 1

        pop.add_particles(xs, vs, s.q, s.m)

def load_particles(pop, species):
    # To load just one species at a time, e.g. to give them
    # different pdf's, use a range operator on species
    for s in species:
        xs = random_domain_points(s.pdf, s.pdf_max, s.num, pop.mesh)
        ors = ORS(s.vdf, s.cutoffs, nsp=s.nsp)
        vs = ors.sample(s.num)
        pop.add_particles(xs, vs, s.q, s.m)
