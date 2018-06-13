import dolfin as df
from punc import *
import numpy as np
from matplotlib import pyplot as plt
from tasktimer import TaskTimer

df.set_log_active(False)

mode = 'dx' # 'dx' or 'dt' depending on what convergence to test

dts = [0.2, 0.1, 0.05, 0.025, 0.0125] if mode=='dt' else [1e-3]
Nrs = [2, 4, 8, 16, 32, 64] if mode=='dx' else [32]
N = 15
npc = 4

if mode == 'dt': Nrs *= len(dts)
if mode == 'dx': dts *= len(Nrs)

order = 1

errors = []
hmins = []
for dt, Nr_ in zip(dts, Nrs):
    print("dt={}, Nr={}".format(dt,Nr_))

    n_dims = 2                           # Number of dimensions
    Ld = 6.28*np.ones(n_dims)            # Length of domain
    Nr = Nr_*np.ones(n_dims,dtype=int)   # Number of 'rectangles' in mesh
    periodic = np.ones(n_dims, dtype=bool)

    # Get the mesh:
    # mesh, facet_func = load_mesh("../mesh/2D/nothing_in_square")
    # mesh, facet_func = load_mesh("../mesh/2D/nonuniform_in_square")
    mesh, facet_func = simple_mesh(Ld, Nr)
    # df.plot(mesh); plt.show()

    ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)

    Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

    ext_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

    V = df.FunctionSpace(mesh, 'CG', order,
                         constrained_domain=PeriodicBoundary(Ld,periodic))
    W = df.FunctionSpace(mesh, 'CG', 1,
                         constrained_domain=PeriodicBoundary(Ld,periodic))
    Y = df.FunctionSpace(mesh, 'DG', 0,
                         constrained_domain=PeriodicBoundary(Ld,periodic))

    poisson = PoissonSolver(V, remove_null_space=True)
    esolver = ESolver(V)

    # dv_inv = voronoi_volume_approx(W)
    dv_inv = voronoi_volume(W, Ld)

    A, mode_ = 0.5, 1
    pdf = lambda x: 1+A*np.sin(mode_*2*np.pi*x[0]/Ld[0]+np.pi/4)

    eps0 = constants.value('electric constant')
    me = constants.value('electron mass')
    mp = constants.value('proton mass')
    e = constants.value('elementary charge')
    ne = 1e2
    vthe = np.finfo(float).eps
    vthi = np.finfo(float).eps
    vd = [0.0,0.0]
    X = np.mean(Ld)

    species = SpeciesList(mesh, X)
    species.append(-e, me, ne, vthe, vd, npc, ext_bnd, pdf=pdf, pdf_max=1 + A)
    species.append(e, mp, ne, vthi, vd, npc, ext_bnd)

    pop = Population(mesh, facet_func)

    load_particles(pop, species)

    KE = np.zeros(N)
    PE = np.zeros(N)
    KE0 = kinetic_energy(pop)

    timer = TaskTimer()
    for n in timer.range(N):
        rho = distribute(W, pop, dv_inv)
        # rho = distribute_dg0(Y, pop)
        phi = poisson.solve(rho)
        E = esolver.solve(phi)
        # PE[n] = particle_potential_energy(pop, phi)
        PE[n] = mesh_potential_energy(rho, phi)
        # PE[n] = efield_potential_energy(E)
        KE[n] = accel(pop,E,(1-0.5*(n==0))*dt)
        move_periodic(pop, Ld, dt)
        pop.update()

    KE[0] = KE0
    TE = KE + PE

    # plt.plot(KE, label="Kinetic Energy")
    # plt.plot(PE, label="Potential Energy")
    # plt.plot(TE, label="Total Energy")
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.xlabel("Timestep")
    # plt.ylabel("Normalized Energy")
    # plt.show()

    error = np.max(np.abs(TE-TE[0]))/TE[0]
    print("Relative energy error: {}".format(error))
    errors.append(error)
    hmins.append(mesh.hmin())

dts = np.array(dts)
hmins = np.array(hmins)
errors = np.array(errors)

if mode=='dt':
    plt.loglog(dts, errors)
    plt.loglog(dts, errors[0]*(dts/dts[0])**1, '--')
    plt.loglog(dts, errors[0]*(dts/dts[0])**2, '--')
    plt.loglog(dts, errors[0]*(dts/dts[0])**3, '--')
else:
    plt.loglog(hmins, errors)
    plt.loglog(hmins, errors[0]*(hmins/hmins[0])**1, '--')
    plt.loglog(hmins, errors[0]*(hmins/hmins[0])**2, '--')
    plt.loglog(hmins, errors[0]*(hmins/hmins[0])**3, '--')

plt.grid()
plt.show()
