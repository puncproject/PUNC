import dolfin as df
from punc import *
import numpy as np
from matplotlib import pyplot as plt

df.set_log_active(False)

#==============================================================================
# INITIALIZING FENICS
#------------------------------------------------------------------------------

n_dims = 2                           # Number of dimensions
Ld = 6.28*np.ones(n_dims)            # Length of domain
Nr = 4*np.ones(n_dims,dtype=int)    # Number of 'rectangles' in mesh
periodic = np.ones(n_dims, dtype=bool)

# Get the mesh:
# mesh, bnd = load_mesh("../mesh/2D/nothing_in_square")
mesh, bnd = load_mesh("../mesh/2D/nonuniform_in_square")
# mesh, bnd = simple_mesh(Ld, Nr)

class OuterBoundary(df.SubDomain):
    def inside(self, x, on_bnd):
        return np.any( np.abs(x) > 6.28-df.DOLFIN_EPS ) and on_bnd

outer_bnd = OuterBoundary()

errors1 = []
errors2 = []
hmins = []
for i in range(3):

    print(i)

    bnd = df.MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, 1)

    Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

    ext_bnd = ExteriorBoundaries(bnd, 1)

    A, mode = 0.5, 1
    pdf = lambda x: 1+A*np.sin(mode*2*np.pi*x[0]/Ld[0])

    npc = 256
    vthe = np.finfo(float).eps
    vthi = np.finfo(float).eps
    vd = [0.0,0.0]
    ne = 1
    e = 1
    me = 1
    mp = 1
    X = 1

    species = SpeciesList(mesh, X)
    species.append_raw(-e, me, ne, vthe, vd, npc, ext_bnd, pdf=pdf, pdf_max=1 + A)
    species.append_raw(e, mp, ne, vthi, vd, npc, ext_bnd)

    pop = Population(mesh, bnd)
    load_particles(pop, species)


    V1 = df.FunctionSpace(mesh, 'CG', 1,
                         constrained_domain=PeriodicBoundary(Ld,periodic))
    dv_inv = voronoi_volume_approx(V1)
    rho1 = distribute(V1, pop, dv_inv)
    rhoe = df.project(df.Expression('-0.5*sin(2*PI*x[0]/6.28)',degree=3,PI=np.pi),V1)
    errors1.append(df.errornorm(rhoe, rho1))

    V2 = df.FunctionSpace(mesh, 'DG', 0,
                         constrained_domain=PeriodicBoundary(Ld,periodic))
    rho2 = distribute_dg0(V2, pop)
    errors2.append(df.errornorm(rhoe, rho2))

    hmins.append(mesh.hmin())

    mesh = df.refine(mesh)

# plot(rhoe); plt.show()
# plot(rho1); plt.show()
# plot(rho2); plt.show()

plt.loglog(hmins, errors1, label='CG1 (Voronoi)')
plt.loglog(hmins, errors2, label='DG0')
plt.grid()
plt.legend(loc='lower right')
plt.show()
