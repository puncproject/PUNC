from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from punc import *

mesh = Mesh("mesh/two_obj.xml")
boundaries = MeshFunction("size_t", mesh, "mesh/two_obj_facet_region.xml")
ext_bnd_id = 17
int1_bnd_id = 18
int2_bnd_id = 19

# Charge and potential difference between objects (if connected)
Q1 = Constant(15.)
Q2 = Constant(20.)
V12 = Constant(3.)
connected = True

cell = mesh.ufl_cell()
V = FiniteElement("Lagrange", cell, 1)
R = FiniteElement("Real", cell, 0)

W = FunctionSpace(mesh, df.MixedElement([V, R, R]))
u, c1, c2 = TrialFunctions(W)
v, d1, d2 = TestFunctions(W)

# W = FunctionSpace(mesh, df.MixedElement([V, R]))
# u, c1 = TrialFunctions(W)
# v, d1 = TestFunctions(W)

ext_bc = DirichletBC(W.sub(0), df.Constant(0), boundaries, ext_bnd_id)
int1_bc = FloatingBC(W.sub(0), boundaries, int1_bnd_id)
int2_bc = FloatingBC(W.sub(0), boundaries, int2_bnd_id)

# rho = Constant(0.)
rho = df.Expression("100*x[1]", degree=3)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=boundaries)
ds1 = dss(int1_bnd_id)
ds2 = dss(int2_bnd_id)
dsi = ds1+ds2

S1 = assemble(Constant(1.)*ds1)
S2 = assemble(Constant(1.)*ds2)


if connected:

    # TWO OBJECTS BIASED WRT ONE ANOTHER
    a = inner(grad(u), grad(v)) * dx -\
        inner(v, dot(grad(u), n)) * dsi +\
        inner(c1, dot(grad(v), n)) * dsi +\
        inner(d1, dot(grad(u), n)) * dsi +\
        inner(c2, (1./S2)*v) * ds2 - inner(c2, (1./S1)*v) * ds1 +\
        inner(d2, (1./S2)*u) * ds2 - inner(d2, (1./S1)*u) * ds1

    L = inner(rho, v) * dx +\
        inner((Q1+Q2)/(S1+S2), d1) * dsi +\
        inner(V12/(S1+S2), d2) * dsi

else:

    # TWO INDEPENDENTLY FLOATING OBJECTS
    a = inner(grad(u), grad(v)) * dx -\
        inner(v, dot(grad(u), n)) * dsi +\
        inner(c1, dot(grad(v), n)) * ds1 +\
        inner(d1, dot(grad(u), n)) * ds1 +\
        inner(c2, dot(grad(v), n)) * ds2 +\
        inner(d2, dot(grad(u), n)) * ds2

    L = inner(rho, v) * dx +\
        inner(Q1/S1, d1) * ds1 +\
        inner(Q2/S2, d2) * ds2

wh = df.Function(W)

A = df.assemble(a)
b = df.assemble(L)
ext_bc.apply(A)
ext_bc.apply(b)
int1_bc.apply(A)
int1_bc.apply(b)
int2_bc.apply(A)
int2_bc.apply(b)

df.solve(A, wh.vector(), b)

uh, ph, qh = wh.split(deepcopy=True)

Q1m = assemble(dot(grad(uh), n) * ds1)
Q2m = assemble(dot(grad(uh), n) * ds2)
print("Object 1 charge: ", Q1m)
print("Object 2 charge: ", Q2m)
print("Total charge:", Q1m+Q2m)
print("Potential difference:", uh(0.4,0)-uh(-0.4,0))

df.plot(uh, interactive=True)
df.File("phi.pvd") << uh
