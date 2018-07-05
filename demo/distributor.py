import dolfin as df
from punc import *
import numpy as np
from matplotlib import pyplot as plt
import time

df.set_log_active(False)

#==============================================================================
# INITIALIZING FENICS
#------------------------------------------------------------------------------
n_dims = 2                           # Number of dimensions
Ld = 6.28*np.ones(n_dims)            # Length of domain
periodic = np.ones(n_dims, dtype=bool)

show_plots = False
mesh_type = 'simple' # 'gmsh_simple', 'gmsh_nonunifom'

if mesh_type == 'gmsh_simple':
    mesh, bnd = load_mesh("../mesh/2D/nothing_in_square")
elif mesh_type == 'gmsh_nonunifom':
    mesh, bnd = load_mesh("../mesh/2D/nonuniform_in_square")

class OuterBoundary(df.SubDomain):
    def inside(self, x, on_bnd):
        return np.any( np.abs(x) > 6.28-df.DOLFIN_EPS ) and on_bnd

times, npc_vec = [], []
for i in range(2,13):

    t0 = time.time()
    npc = 2**i
    errors1 = []
    errors2 = []
    hmins = []

    for j in range(2,7):

        if mesh_type == 'simple':
            Nr = 2**j*np.ones(n_dims,dtype=int)
            mesh, bnd = simple_mesh(Ld, Nr)

        print("npc: ", npc, ", run:",j-1,", Nr = ", 2**j)

        outer_bnd = OuterBoundary()
        bnd = df.MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
        bnd.set_all(0)
        outer_bnd.mark(bnd, 1)

        Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

        ext_bnd = ExteriorBoundaries(bnd, 1)

        A, mode = 0.5, 1
        pdf = lambda x: 1+A*np.sin(mode*2*np.pi*x[0]/Ld[0])

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
        # species.append_raw(e, mp, ne, vthi, vd, npc, ext_bnd)

        pop = Population(mesh, bnd)
        load_particles(pop, species)


        V1 = df.FunctionSpace(mesh, 'CG', 1,
                             constrained_domain=PeriodicBoundary(Ld,periodic))
        dv_inv = voronoi_volume_approx(V1)
        rho1 = distribute(V1, pop, dv_inv)
        # rhoe = df.project(df.Expression('-0.5*sin(2*PI*x[0]/6.28)',degree=3,PI=np.pi),V1)
        rhoe = df.Expression('-1.0-0.5*sin(2*PI*x[0]/6.28)',degree=3,PI=np.pi)
        errors1.append(df.errornorm(rhoe, rho1))

        V2 = df.FunctionSpace(mesh, 'DG', 0,
                             constrained_domain=PeriodicBoundary(Ld,periodic))
        rho2 = distribute_DG0(V2, pop)
        errors2.append(df.errornorm(rhoe, rho2))

        hmins.append(mesh.hmin())

        if mesh_type == 'gmsh_simple' or mesh_type=='gmsh_nonunifom':
            mesh = df.refine(mesh)

    t1 = time.time()
    times.append(t1 - t0)
    npc_vec.append(npc)

    hmins = np.array(hmins)

    plt.figure()
    plt.loglog(hmins, errors1, label='CG1 (Voronoi)')
    plt.loglog(hmins, errors2, label='DG0')
    plt.loglog(hmins, errors2[0]*(hmins/hmins[0])**1, '--')
    plt.loglog(hmins, errors1[0]*(hmins/hmins[0])**2, '--')
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlim([hmins[-1]-0.01, hmins[0]+0.1])
    # plt.savefig('convergence_npc_' + str(npc) + '.png', bbox_inches='tight')
    if show_plots:
        plt.show()


if n_dims == 2 and show_plots:
    plot(df.project(rhoe, V1)); plt.show()
    plot(rho1); plt.show()
    plot(rho2); plt.show()

for k in range(len(times)):
    minute, sec = divmod(times[k], 60)
    hour, minute = divmod(minute, 60)
    print("npc = %4d: time = %d:%02d:%02d" %(npc_vec[k],hour, minute, sec))

print("Total time: ", sum(times))
