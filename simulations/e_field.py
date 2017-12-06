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
import clement as ci

df.parameters["allow_extrapolation"] = True

tol = 1e-3
r_inner = 0.1
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
fnames = ["../mesh/3D/sphere_in_sphere_res1","../mesh/3D/sphere_in_sphere_res3",
          "../mesh/3D/sphere_in_sphere_res4", "../mesh/3D/sphere_in_sphere_res5",
          "../mesh/3D/sphere_in_sphere_res2"]

# fnames = ["../mesh/3D/sphere_in_sphere_res1", "../mesh/3D/sphere_in_sphere_res2"]
# fnames = ["../mesh/2D/langmuir_probe_circle_in_circle"]

err_phi, err_E_cg1, err_E_dg0, h = np.empty(
    len(fnames)), np.empty(len(fnames)), np.empty(len(fnames)), np.empty(len(fnames))

thickness = [0.2,0.2,0.2,0.2,0.2]

interpolate = False
if interpolate:
    to_file = open('Results/FourthSim/efield_data.txt', 'w')

timer = TaskTimer(len(fnames), 'compact')

for i, fname in enumerate(fnames):
    timer.task("Loading mesh")
    mesh, bnd = load_mesh(fname)
    ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)


    V = df.FunctionSpace(mesh, 'CG', 1)

    ext_bnd = ExteriorBoundaries(bnd, ext_bnd_id)
    bc = df.DirichletBC(V, df.Constant(phi_outer), bnd, ext_bnd_id)
    objects = [Object(V, j, bnd) for j in int_bnd_ids]

    objects[0].set_potential(phi_inner)

    poisson = PoissonSolver(V, bc)    
    esolver_cg1 = ESolver(V, 'CG', 1)
    esolver_dg0 = ESolver(V, 'DG', 0)

    rho = df.Function(V)

    timer.task("Potential solver")
    phi = poisson.solve(rho, objects)
    timer.task("E-field solver - CG1")
    E_cg1 = esolver_cg1.solve(phi)
    timer.task("E-field solver - DG0")
    E_dg0 = esolver_dg0.solve(phi)

    if not interpolate:
        Wcg = df.VectorFunctionSpace(V.mesh(), 'CG', 1)
        Wdg = df.VectorFunctionSpace(V.mesh(), 'DG', 0)
        timer.task("Interpolation potential - phi")
        phie = df.interpolate(phi_expr, V)
        timer.task("Interpolation e-field - CG1")
        E_cg1e = df.interpolate(e_field_expr, Wcg)
        timer.task("Interpolation e-field - DG0")
        E_dg0e = df.interpolate(e_field_expr, Wdg)
        timer.task("Write to file")
        df.File("Results/FifthSim/phi_%i" % i + ".pvd") << phi
        df.File("Results/FifthSim/phi_analytical_%i" % i + ".pvd") << phie
        df.File("Results/FifthSim/E_cg1_%i" % i + ".pvd") << E_cg1
        df.File("Results/FifthSim/E_analytical_cg1_%i" % i + ".pvd") << E_cg1e
        df.File("Results/FifthSim/E_dg0_%i" % i + ".pvd") << E_dg0
        df.File("Results/FifthSim/E_analytical_dg0_%i" % i + ".pvd") << E_dg0e
        # W_cg1 = E_cg1.ufl_function_space()
        # W_dg0 = E_dg0.ufl_function_space()

        # phi_analytical = df.interpolate(phi_expr, V)
        # E_analytical_cg1 = df.interpolate(e_field_expr, W_cg1)
        # E_analytical_dg0 = df.interpolate(e_field_expr, W_dg0)
    if interpolate:
        cc = df.CellFunction('size_t', mesh, 0)
        xyplane = df.AutoSubDomain(lambda x: abs(x[2]) < thickness[i])
        xyplane.mark(cc, 1)

        submesh = df.SubMesh(mesh, cc, 1)

        Vsub = df.FunctionSpace(submesh, "CG", 1)
        Wsub_cg1 = df.VectorFunctionSpace(submesh, "CG", 1)
        Wsub_dg0 = df.VectorFunctionSpace(submesh, "DG", 0)

        timer.task("Interpolation - phi")
        phis = df.interpolate(phi, Vsub)
        phie = df.interpolate(phi_expr, Vsub)

        timer.task("Interpolation e-field - CG1")
        E_cg1s = df.interpolate(E_cg1, Wsub_cg1)
        E_cg1e = df.interpolate(e_field_expr, Wsub_cg1)
        timer.task("Interpolation e-field - DG0")
        E_dg0s = df.interpolate(E_dg0, Wsub_dg0)
        E_dg0e = df.interpolate(e_field_expr, Wsub_dg0)

        timer.task("Errornorm - potential")
        err_phi[i] = df.errornorm(phis, phie, degree_rise=3)
        timer.task("Errornorm - e-field - CG1")
        err_E_cg1[i] = df.errornorm(E_cg1s, E_cg1e, degree_rise=3)
        timer.task("Errornorm - e-field - DG0")
        err_E_dg0[i] = df.errornorm(E_dg0s, E_dg0e, degree_rise=3)
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

        timer.task("Write to file")
        to_file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                    (h[i], err_phi[i], err_E_cg1[i], err_E_dg0[i], order_phi, order_cg1, order_dg0))

        df.File("Results/FifthSim/phi_%i" % i + ".pvd") << phis
        df.File("Results/FifthSim/phi_analytical_%i" % i + ".pvd") << phie
        df.File("Results/FifthSim/E_cg1_%i" % i + ".pvd") << E_cg1s
        df.File("Results/FifthSim/E_analytical_cg1_%i" % i + ".pvd") << E_cg1e
        df.File("Results/FifthSim/E_dg0_%i" % i + ".pvd") << E_dg0s
        df.File("Results/FifthSim/E_analytical_dg0_%i" % i + ".pvd") << E_dg0e
    
    timer.end()

timer.summary()
if interpolate:
    to_file.close()
    #df.plot(us)
    #df.interactive()


    #     err_phi[i] = df.errornorm(phi, phi_analytical, degree_rise=0)
    #     err_E_cg1[i] = df.errornorm(E_cg1, E_analytical_cg1, degree_rise=0)
    #     err_E_dg0[i] = df.errornorm(E_dg0, E_analytical_dg0, degree_rise=0)
    #     h[i] = mesh.hmin()
    #     order_phi = np.log(err_phi[i] / err_phi[i - 1]) / \
    #         np.log(h[i] / h[i - 1]) if i > 0 else 0
    #     order_cg1 = np.log(err_E_cg1[i] / err_E_cg1[i - 1]) / \
    #         np.log(h[i] / h[i - 1]) if i > 0 else 0
    #     order_dg0 = np.log(err_E_dg0[i] / err_E_dg0[i - 1]) / \
    #         np.log(h[i] / h[i - 1]) if i > 0 else 0

    #     print("Running mesh %3d: h=%2.2E, E_phi=%2.2E, order=%2.2E, E_cg1=%2.2E, \
    #           order=%2.2E, E_dg0=%2.2E, order=%2.2E" %
    #           (i, h[i], err_phi[i], order_phi, err_E_cg1[i], order_cg1, err_E_dg0[i], order_dg0))


    plt.loglog(h, err_phi,  color='#1e7fbc', linewidth=2.0,
            linestyle='-', label="potential")
    plt.loglog(h, err_E_cg1,  color='#bc1d1d', linewidth=2.0,
            linestyle='-.', label="e-field, CG1")
    plt.loglog(h, err_E_dg0,  color='#12a823', linewidth=2.0,
            linestyle='--', label="e-field, DG0")
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Minimum cell diameter in mesh')
    plt.ylabel('L2 error norm')
    plt.savefig('Results/FourthSim/e_field.eps', bbox_inches='tight', dpi=1000)
    plt.savefig('Results/FourthSim/e_field.svg', bbox_inches='tight', dpi=1000)
    plt.savefig('Results/FourthSim/e_field.png', bbox_inches='tight', dpi=1000)
    plt.show()


