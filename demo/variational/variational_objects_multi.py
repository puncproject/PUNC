from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import *
import time

mesh = Mesh("../mesh/2D/circle_and_square_in_square_res2.xml")
boundaries = MeshFunction("size_t", mesh,
                          "../mesh/2D/circle_and_square_in_square_res2_facet_region.xml")
ext_bnd_id = 17
int_bnd_ids = [18, 19]

# Simulation settings
charges = [15., 20.]
# charges = [Constant(charge) for charge in charges]
V12 = Constant(3.)
method = 'bicgstab'
preconditioner = 'ilu'

cell = mesh.ufl_cell()
V = FiniteElement("Lagrange", cell, 1)
R = FiniteElement("Real", cell, 0)

W = FunctionSpace(mesh, MixedElement([V]+[R]*len(int_bnd_ids)))
c = list(TrialFunctions(W))
d = list(TestFunctions(W))
u = c.pop(0)
v = d.pop(0)

ext_bc = DirichletBC(W.sub(0), Constant(0), boundaries, ext_bnd_id)
int_bcs = [FloatingBC(W.sub(0), boundaries, id) for id in int_bnd_ids]

rho = df.Expression("100*x[1]", degree=3)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
areas = [assemble(Constant(1.)*dss(id)) for id in int_bnd_ids]

groups = [[0, 1]]
biases = [[0, 5]]

# groups = [[0], [1]]
# biases = [[0], [0]]

#
# CREATING VARIATIONAL FORM
#
a = inner(grad(u), grad(v)) * dx -\
    inner(v, dot(grad(u), n)) * ds
L = inner(rho, v) * dx

i = 0
for group, bias in zip(groups, biases):

    ds_group = dss(int_bnd_ids[group[0]])
    for p in range(1,len(group)):
        ds_group += dss(int_bnd_ids[group[p]])

    area_group = sum([areas[p] for p in group])
    charge_group = sum([charges[p] for p in group])

    a += inner(c[i], dot(grad(v), n)) * ds_group +\
         inner(d[i], dot(grad(u), n)) * ds_group
    L += inner(charge_group/area_group, d[i]) * ds_group
    i += 1

    ref = group[0]
    refid = int_bnd_ids[ref]
    for j in range(1, len(group)):
        this = group[j]
        thisid = int_bnd_ids[this]

        a += inner(c[i], (1./areas[this])*v)*dss(thisid) -\
             inner(c[i], (1./areas[ref] )*v)*dss(refid)  +\
             inner(d[i], (1./areas[this])*u)*dss(thisid) -\
             inner(d[i], (1./areas[ref] )*u)*dss(refid)
        L += inner((bias[j]/area_group), d[i]) * ds_group
        i += 1

assert i==len(int_bnd_ids)

Q1, Q2 = charges
S1, S2 = areas
ds1 = dss(int_bnd_ids[0])
ds2 = dss(int_bnd_ids[1])
dsi = ds1+ds2
c1, c2 = c
d1, d2 = d

# a = inner(grad(u), grad(v)) * dx -\
#     inner(v, dot(grad(u), n)) * ds +\
#     inner(c1, dot(grad(v), n)) * dsi +\
#     inner(d1, dot(grad(u), n)) * dsi +\
#     inner(c2, (1./S2)*v) * ds2 - inner(c2, (1./S1)*v) * ds1 +\
#     inner(d2, (1./S2)*u) * ds2 - inner(d2, (1./S1)*u) * ds1

# L = inner(rho, v) * dx +\
#     inner((Q1+Q2)/(S1+S2), d1) * dsi +\
#     inner(V12/(S1+S2), d2) * dsi

# a = inner(grad(u), grad(v)) * dx -\
#     inner(v, dot(grad(u), n)) * dsi +\
#     inner(c1, dot(grad(v), n)) * dsi +\
#     inner(d1, dot(grad(u), n)) * dsi +\
#     inner(c2, (1./S2)*v) * ds2 - inner(c2, (1./S1)*v) * ds1 +\
#     inner(d2, (1./S2)*u) * ds2 - inner(d2, (1./S1)*u) * ds1
#
# L = inner(rho, v) * dx +\
#     inner((Q1+Q2)/(S1+S2), d1) * dsi +\
#     inner(V12/(S1+S2), d2) * dsi

wh = df.Function(W)

A = df.assemble(a)
b = df.assemble(L)
ext_bc.apply(A)
ext_bc.apply(b)
for int_bc in int_bcs:
    int_bc.apply(A)
    int_bc.apply(b)

solver = KrylovSolver(method,preconditioner)
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-12 #e-12
solver.parameters['maximum_iterations'] = 100000
# solver.parameters['monitor_convergence'] = True
# solver.parameters['nonzero_initial_guess'] = True

print("Setting operator (computing preconditioning?)")
solver.set_operator(A)

for it in range(3):
    t0 = time.time()
    solver.solve(wh.vector(), b)
    t1 = time.time()
    print("Solving %dst time: %.5f"%(it+1,t1-t0))

ch = wh.split(deepcopy=True)
uh = ch[0]

charge_meas = [assemble(dot(grad(uh), n) * dss(id)) for id in int_bnd_ids]
print("Object charges:", charge_meas)
print("Total charge:", sum(charge_meas))
print("Potential difference:", uh(0.4,0)-uh(-0.4,0))

df.plot(uh)
df.File("phi.pvd") << uh
