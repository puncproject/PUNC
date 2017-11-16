from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
import os
import sys
import importlib


r_inner = 0.2
r_outer = 1.0

phi_inner = 1.0
phi_outer = 0.0

phi_expr = df.Expression("( (phi_1-phi_2)*r_1*r_2/( (r_2-r_1) * pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5) )) +phi_1-r_2*(phi_1-phi_2)/(r_2-r_1)", phi_1=phi_inner, phi_2=phi_outer, r_1=r_inner, r_2=r_outer, degree=3)

e_field_expr = df.Expression( ( ("(phi_1-phi_2)*r_1*r_2*x[0] /\
                                    ((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5))"), 
                                      ("(phi_1-phi_2)*r_1*r_2*x[1] /\
                                    ((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5))"),
                                      ("(phi_1-phi_2)*r_1*r_2*x[2] /\
                                    ((r_2-r_1) * pow(x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.5))"),
                                    ), phi_1=phi_inner, phi_2=phi_outer, r_1=r_inner, r_2=r_outer, degree=3)


# Filename of mesh (excluding .xml)
# fnames = ["../mesh/3D/sphere_in_sphere_res1","../mesh/3D/sphere_in_sphere_res2",
#           "../mesh/3D/sphere_in_sphere_res3", "../mesh/3D/sphere_in_sphere_res4"]

fnames = ["../mesh/3D/sphere_in_sphere_res1", "../mesh/3D/sphere_in_sphere_res2"]

err_phi, err_E_cg1, err_E_dg0, h = np.empty(
    len(fnames)), np.empty(len(fnames)), np.empty(len(fnames)), np.empty(len(fnames))

for i, fname in enumerate(fnames):
    mesh, bnd = load_mesh(fname)
    ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)

    V = df.FunctionSpace(mesh, 'CG', 1)

    ext_bnd = ExteriorBoundaries(bnd, ext_bnd_id)
    bc = df.DirichletBC(V, df.Constant(phi_outer), bnd, ext_bnd_id)
    objects = [Object(V, j, bnd) for j in int_bnd_ids]

    objects[0].set_potential(phi_inner)

    # Get the solver
    poisson = PoissonSolver(V, bc)

    esolver_cg1 = ESolver(V, 'CG', 1)
    esolver_dg0 = ESolver(V, 'DG', 0)

    rho = df.Function(V)

    phi = poisson.solve(rho, objects)
    E_cg1 = esolver_cg1.solve(phi)
    E_dg0 = esolver_dg0.solve(phi)

    W_cg1 = E_cg1.ufl_function_space()
    W_dg0 = E_dg0.ufl_function_space()

    phi_analytical = df.interpolate(phi_expr, V)
    E_analytical_cg1 = df.interpolate(e_field_expr, W_cg1)
    E_analytical_dg0 = df.interpolate(e_field_expr, W_dg0)

    err_phi[i] = df.errornorm(phi, phi_analytical, degree_rise=0)
    err_E_cg1[i] = df.errornorm(E_cg1, E_analytical_cg1, degree_rise=0)
    err_E_dg0[i] = df.errornorm(E_dg0, E_analytical_dg0, degree_rise=0)
    h[i] = mesh.hmin()
    order_phi = np.log(err_phi[i] / err_phi[i - 1]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_cg1 = np.log(err_E_cg1[i] / err_E_cg1[i - 1]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0
    order_dg0 = np.log(err_E_dg0[i] / err_E_dg0[i - 1]) / \
        np.log(h[i] / h[i - 1]) if i > 0 else 0

    print("Running mesh %3d: h=%2.2E, E_phi=%2.2E, order=%2.2E, E_cg1=%2.2E, \
          order=%2.2E, E_dg0=%2.2E, order=%2.2E" %
          (i, h[i], err_phi[i], order_phi, err_E_cg1[i], order_cg1, err_E_dg0[i], order_dg0))
    
    df.File("phi_%i"%i + ".pvd") << phi
    df.File("phi_analytical_%i" % i + ".pvd") << phi_analytical
    df.File("E_cg1_%i" % i + ".pvd") << E_cg1
    df.File("E_analytical_cg1_%i" % i + ".pvd") << E_analytical_cg1
    df.File("E_dg0_%i" % i + ".pvd") << E_dg0
    df.File("E_analytical_dg0_%i" % i + ".pvd") << E_analytical_dg0


plt.loglog(h, err_phi, linestyle='-', label="potential")
plt.loglog(h, err_E_cg1, linestyle='-.', label="e-field, CG1")
plt.loglog(h, err_E_dg0, linestyle='--', label="e-field, DG0")
plt.grid()
plt.legend(loc='best')
plt.xlabel('Minimum cell diameter in mesh')
plt.ylabel('L2 error norm')
plt.savefig('e_field.eps', bbox_inches='tight', dpi=600)
plt.savefig('e_field.png', bbox_inches='tight', dpi=600)
plt.show()

