# Imports important python 3 behaviour to ensure correct operation and
# performance in python 2
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
from punc import *
import numpy as np
from matplotlib import pyplot as plt

#==============================================================================
# INITIALIZING FENICS
#------------------------------------------------------------------------------

n_dims = 2                           # Number of dimensions
Ld = 6.28*np.ones(n_dims)            # Length of domain
Nr = 32*np.ones(n_dims,dtype=int)    # Number of 'rectangles' in mesh
periodic = np.ones(n_dims, dtype=bool)

# Get the mesh:
# mesh, facet_func = load_mesh("../mesh/2D/nothing_in_square")
# mesh, facet_func = load_mesh("../mesh/2D/nonuniform_in_square")
mesh, facet_func = simple_mesh(Ld, Nr)

ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)

Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

exterior_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

V = df.FunctionSpace(mesh, 'CG', 1,
                     constrained_domain=PeriodicBoundary(Ld,periodic))

poisson = PoissonSolver(V, remove_null_space=True)
esolver = ESolver(V)

dv_inv = voronoi_volume(V, Ld, True)

A, mode = 0.5, 1
pdf = lambda x: 1+A*np.sin(mode*2*np.pi*x[0]/Ld[0])

pop = Population(mesh, facet_func, normalization='plasma params')
pop.init_new_specie('electron', exterior_bnd, pdf=pdf, pdf_max=1 + A)
pop.init_new_specie('proton', exterior_bnd)

dt = 0.251327
N = 30

KE = np.zeros(N-1)
PE = np.zeros(N-1)
KE0 = kinetic_energy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))
    rho = distribute(V,pop, dv_inv)
    phi = poisson.solve(rho)
    E = esolver.solve(phi)
    PE[n-1] = potential_energy(pop, phi)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)
    move_periodic(pop,Ld,dt)
    pop.update()

KE[0] = KE0

TE = KE + PE


plt.plot(KE,label="Kinetic Energy")
plt.plot(PE,label="Potential Energy")
plt.plot(KE+PE,label="Total Energy")
plt.legend(loc='lower right')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Normalized Energy")
plt.savefig('langmuir.eps',bbox_inches='tight',dpi=600)
plt.savefig('langmuir.png',bbox_inches='tight',dpi=600)

plt.show()

import matplotlib.tri as tri

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot(obj, title):
    plt.gca().set_aspect('equal')
    if isinstance(obj, df.Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().get_local()
            plt.tripcolor(mesh2triang(mesh), C, cmap='viridis')
            plt.colorbar()
            plt.title(title)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud', cmap='viridis')
            plt.colorbar()
            plt.title(title)
    elif isinstance(obj, df.Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')

if int(sys.version_info[:1][0]) < 3:
    # Use dolfin plot utilities if python version is 2.7
    df.plot(rho)
    df.plot(phi,window_width=800, window_height=600,)
    ux = df.Constant((1,0))
    Ex = df.project(df.inner(E,ux),V)
    df.plot(Ex)
    df.interactive()
else:
    # Otherwise use matplotlib
    plt.figure()
    plot(rho, 'rho')
    plt.xlim(0, Ld[0])
    plt.ylim(0, Ld[1])
    plt.savefig('rho.png', bbox_inches='tight', dpi=600)

    plt.figure()
    plot(phi, 'phi')
    plt.xlim(0, Ld[0])
    plt.ylim(0, Ld[1])
    plt.savefig('phi.png', bbox_inches='tight', dpi=600)

    ux = df.Constant((1, 0))
    Ex = df.project(df.inner(E, ux), V)

    plt.figure()
    plot(Ex, 'Ex')
    plt.xlim(0, Ld[0])
    plt.ylim(0, Ld[1])
    plt.savefig('Ex.png', bbox_inches='tight', dpi=600)
    plt.show()

    # Save solution in VTK format
    file = df.File("rho.pvd")
    file << rho
    file = df.File("phi.pvd")
    file << rho
    file = df.File("E.pvd")
    file << rho
