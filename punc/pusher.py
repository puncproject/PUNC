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

import dolfin as df
import numpy as np

code="""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/FiniteElement.h>

Eigen::VectorXd restrict(const dolfin::GenericFunction& self,
                         const dolfin::FiniteElement& element,
                         const dolfin::Cell& cell){

    ufc::cell ufc_cell;
    cell.get_cell_data(ufc_cell);

    std::vector<double> coordinate_dofs;
    cell.get_coordinate_dofs(coordinate_dofs);

    std::size_t s_dim = element.space_dimension();
    Eigen::VectorXd w(s_dim);
    self.restrict(w.data(), element, cell, coordinate_dofs.data(), ufc_cell);

    return w; // no copy
}
PYBIND11_MODULE(SIGNATURE, m){
    m.def("restrict", &restrict);
}
"""
compiled = df.compile_cpp_code(code, cppargs='-O3')

def restrict(function, element, cell):
    return compiled.restrict(function.cpp_object(), element, cell)

def accel(pop, E, dt):

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    v_dim = element.value_dimension(0) # Number of values per node (=geom. dim.)

    KE = 0.0
    for cell in df.cells(mesh):

        coefficients = restrict(E, element, cell)

        for particle in pop[cell.index()]:

            basis_matrix = element.evaluate_basis_all(
                               particle.x,
                               cell.get_vertex_coordinates(),
                               cell.orientation()
                           ).reshape((-1,v_dim))

            Ei = np.dot(coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v

            inc = dt*(q/m)*Ei

            KE += 0.5*m*np.dot(vel,vel+inc) # This has error O(dt^2)

            particle.v += inc


    return KE

def boris(pop, E, B, dt):

    assert mesh.geometry().dim() == 3

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    v_dim = element.value_dimension(0) # Number of values per node (=geom. dim.)
    KE = 0.0
    for cell in df.cells(mesh):

        coefficients = restrict(E, element, cell)

        for particle in pop[cell.index()]:

            basis_matrix = element.evaluate_basis_all(
                               particle.x,
                               cell.get_vertex_coordinates(),
                               cell.orientation()
                           ).reshape((-1,v_dim))

            Ei = np.dot(coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v

            t = np.tan((dt*q/(2.*m))*B)
            s = 2.*t/(1.+t[0]**2+t[1]**2+t[2]**2)
            v_minus = vel + 0.5*dt*(q/m)*Ei

            KE += 0.5*m*np.dot(v_minus,v_minus)

            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            particle.v = v_plus[:] + 0.5*dt*(q/m)*Ei

    return KE


def boris_nonuniform(pop, E, B, dt):

    assert mesh.geometry().dim() == 3

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    v_dim = element.value_dimension(0)

    KE = 0.0
    for cell in df.cells(mesh):

        coefficients = restrict(E, element, cell)
        mag_coefficients = restrict(B, element, cell)

        for particle in pop[cell.index()]:

            basis_matrix = element.evaluate_basis_all(
                               particle.x,
                               cell.get_vertex_coordinates(),
                               cell.orientation()
                           ).reshape((-1,v_dim))

            Ei = np.dot(coefficients, basis_matrix)[:]
            Bi = np.dot(mag_coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v

            t = np.tan((dt * q / (2. * m)) * Bi)
            s = 2. * t / (1. + t[0]**2 + t[1]**2 + t[2]**2)
            v_minus = vel + 0.5 * dt * (q / m) * Ei

            KE += 0.5 * m * np.dot(v_minus, v_minus)

            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            particle.v = v_plus[:] + 0.5 * dt * (q / m) * Ei

    return KE

def move_periodic(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
            particle.x %= Ld

def move(pop, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
