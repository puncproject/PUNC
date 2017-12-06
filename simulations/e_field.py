from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from mpi4py import MPI as piMPI
import dolfin as df
import numpy as np
import os
import sys
from punc.poisson import *
from punc.objects import *

df.set_log_level(df.PROGRESS)

tol = 1e-3
r_inner = 0.1
r_outer = 1.0

phi_inner = 1.0
phi_outer = 0.0

phi_expr = df.Expression(
    "(1.0/9.0)*(-1.0 + 1.0/(pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5)))", degree=5)

e_field_expr = df.Expression(("(1.0/9.0)*x[0]/(pow(x[0]*x[0] + x[1]*x[1] + x[2]*x[2], 1.5))",
                              "(1.0/9.0)*x[1]/(pow(x[0]*x[0] + x[1]*x[1] + x[2]*x[2], 1.5))",
                              "(1.0/9.0)*x[2]/(pow(x[0]*x[0] + x[1]*x[1] + x[2]*x[2], 1.5))"), degree=5)

fnames = ["../mesh/3D/sphere_in_sphere_res1.h5", "../mesh/3D/sphere_in_sphere_res2.h5",
          "../mesh/3D/sphere_in_sphere_res3.h5", "../mesh/3D/sphere_in_sphere_res4.h5",
          "../mesh/3D/sphere_in_sphere_res5.h5"]


# if df.MPI.rank(comm) == 0:
to_file = open('results/efield_data.txt', 'w')
err = np.empty((len(fnames), 5))
h = np.empty(len(fnames))
npts = [256, 512, 1024, 2048, 4096]
err_on_domain = True
for i, fname in enumerate(fnames[:1]):

    mesh, boundaries, comm = load_h5_mesh(fname)
    ext_bnd_id, int_bnd_ids = get_mesh_ids(boundaries, comm)

    V = df.FunctionSpace(mesh, 'CG', 1)

    bc = df.DirichletBC(V, df.Constant(phi_outer), boundaries, ext_bnd_id)
    objects = [Object(V, j, boundaries) for j in int_bnd_ids]
    objects[0].set_potential(phi_inner)

    # ----Potential - Poisson solver -------------------------------------------
    poisson = PoissonSolver(V, bc)
    rho = df.Function(V)
    phi = poisson.solve(rho, objects)

    # ----E-field projected onto CG 1 ------------------------------------------
    esolver = ESolver(V)
    E_cg1 = esolver.solve(phi)

    #------- E-field projected onto DG0 ----------------------------------------
    E_dg0 = efield_DG0(mesh, phi)

    #------- Arithmetic mean ---------------------------------------------------
    am = EfieldMean(mesh, arithmetic_mean=True)
    E_am = am.mean(phi)

    #------- Clement interpolation -------- ------------------------------------
    ci = EfieldMean(mesh, arithmetic_mean=False)
    E_ci = ci.mean(phi)

    #------- L2 errornorm ------------------------------------------------------
    if err_on_domain:
        h[i] = df.MPI.min(comm, mesh.hmin())
        for m, (numeric, exact) in enumerate(zip([phi, E_cg1, E_dg0, E_ci, E_am], [phi_expr, e_field_expr, e_field_expr, e_field_expr, e_field_expr])):
            err[i, m] = df.errornorm(exact, numeric, degree_rise=0)

    else:
        Apt = np.array([0.2, 0.0, 0.0])
        Bpt = np.array([0.9, 0.0, 0.0])
        npt = 2 * npts[i]
        mesh_points = [Apt + t * (Bpt - Apt)
                       for t in np.linspace(0, 1, npt + 1)]
        tdim, gdim = 1, len(Apt)
        line_mesh = df.Mesh()
        editor = df.MeshEditor()
        editor.open(line_mesh, tdim, gdim)
        editor.init_vertices(npt + 1)
        editor.init_cells(npt)

        for vi, v in enumerate(mesh_points):
            editor.add_vertex(vi, v)

        for cj in range(npt):
            editor.add_cell(cj, np.array([cj, cj + 1], dtype='uintp'))

        editor.close()

        h[i] = line_mesh.hmin()
        for m, (numeric, exact) in enumerate(zip([phi, E_cg1, E_dg0, E_ci, E_am], [phi_expr, e_field_expr, e_field_expr, e_field_expr, e_field_expr])):
            family = numeric.function_space().ufl_element().family()
            degree = numeric.function_space().ufl_element().degree()

            if numeric.value_rank() == 0:
                V_line = df.FunctionSpace(line_mesh, family, degree)
                numeric_line = df.interpolate(numeric, V_line)
            else:
                V_line = df.VectorFunctionSpace(line_mesh, family, degree)
                numeric_line = df.interpolate(numeric, V_line)

            err[i, m] = df.errornorm(exact, numeric_line, degree_rise=0)

    order_phi = np.log(err[i, 0] / err[i - 1, 0]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_cg = np.log(err[i, 1] / err[i - 1, 1]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_dg = np.log(err[i, 2] / err[i - 1, 2]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_ci = np.log(err[i, 3] / err[i - 1, 3]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_am = np.log(err[i, 4] / err[i - 1, 4]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0

    if df.MPI.rank(comm) == 0:
        to_file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                      (h[i], err[i, 0], err[i, 1], err[i, 2], err[i, 3], err[i, 4], order_phi, order_cg, order_dg, order_ci, order_am))
        to_file.flush()

    if df.MPI.rank(comm) == 0:
        print("--------------------------O----------------------------------     ")
        print("Calculations for mesh nr. ", i,
              "/", len(fnames), " is completed.")
        print("--------------------------O-----------------------------------    ")

if df.MPI.rank(comm) == 0:
    to_file.close()
