import dolfin as df
from punc import *
from tasktimer import TaskTimer
import numpy as np
from matplotlib import pyplot as plt

dts = [2**(-n) for n in range(10)]
N = 30

n_dims = 2                           # Number of dimensions
Ld = 6.28*np.ones(n_dims)            # Length of domain
Nr = 32*np.ones(n_dims,dtype=int)   # Number of 'rectangles' in mesh
periodic = np.ones(n_dims, dtype=bool)

mesh, facet_func = simple_mesh(Ld, Nr)
ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)
Ld = get_mesh_size(mesh)  # Get the size of the simulation domain
ext_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

V = df.VectorFunctionSpace(mesh, 'CG', 1,
                     constrained_domain=PeriodicBoundary(Ld,periodic))

A = Ld[0]/4
theta = np.pi/4
xm = Ld[0]/2*np.ones(n_dims)
dir = np.array([1,0])
x0 = xm + A*np.cos(theta)*dir
v0 = -A*np.sin(theta)*dir

E = df.project(df.Expression(('x[0]-xm','0'),degree=1,xm=xm[0]),V)

errors = []
timer = TaskTimer()
for dt in timer.iterate(dts):

    pop = Population(mesh, facet_func)
    pop.add_particles([x0],[v0],[-1],[1])
    x = []
    for n in range(N):

        # Find position of the only particle
        for c in pop:
            if c!=[]: break
        x.append(c[0].x[0])

        # Creative use of TaskTimer (not recommended)
        timer.task("Position: {}".format(c[0].x[0]))

        accel(pop,E,(1-0.5*(n==0))*dt)
        move_periodic(pop, Ld, dt)
        # move(pop, dt)
        pop.update()

    t = dt*np.array(range(N))
    xe = A*np.cos(t+theta)+xm[0]

    # plt.plot(t, x)
    # plt.plot(t, xe)
    # plt.show()

    error = np.max(np.abs(x-xe))
    errors.append(error)

dts = np.array(dts)
plt.loglog(dts, errors)
plt.loglog(dts, errors[0]*(dts/dts[0])**1, '--')
plt.loglog(dts, errors[0]*(dts/dts[0])**2, '--')
plt.loglog(dts, errors[0]*(dts/dts[0])**3, '--')
plt.grid()
plt.show()
