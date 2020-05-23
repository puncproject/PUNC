from dolfin import *
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
compiled = compile_cpp_code(code) #, cppargs='-O3')

def restrict(function, element, cell):
    return compiled.restrict(function.cpp_object(), element, cell)

mesh = UnitSquareMesh(8,8)
W = FunctionSpace(mesh, 'CG', 1)
E = interpolate(Expression('x[0]', degree=1), W)

element = W.dolfin_element()
cell = list(cells(mesh))[4]

coefficients = restrict(E, element, cell)
print(coefficients)
