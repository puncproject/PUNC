from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
import numpy as np
from punc import *

from mesh import *

#-------------------------------------------------------------------------------
#          Get the mesh and the information about the object
#-------------------------------------------------------------------------------
dim = 2
n_components = 0
Ld = [np.pi, 2*np.pi]
N = [16, 32]
msh = SimpleMesh(Ld, N)
mesh = msh.mesh()
L = msh.L
#-------------------------------------------------------------------------------
#           Create the objects
#-------------------------------------------------------------------------------
objects = []
#-------------------------------------------------------------------------------
#        Create facet and cell functions to to mark the boundaries
#-------------------------------------------------------------------------------
facet_f = FacetFunction('size_t', mesh)
facet_f.set_all(n_components+len(L))
#-------------------------------------------------------------------------------
#       Mark the exterior boundaries of the simulation domain
#-------------------------------------------------------------------------------
facet_f = mark_exterior_boundaries(facet_f, n_components, L)
#-------------------------------------------------------------------------------
#                       Simulation parameters
#-------------------------------------------------------------------------------
n_pr_cell = 8             # Number of particels per cell
n_pr_super_particle = 8   # Number of particles per super particle
tot_time = 10             # Total simulation time
dt = 0.251327             # Time step

tot_volume = assemble(1*dx(mesh)) # Volume of simulation domain

n_cells = mesh.num_cells()    # Number of cells
N_e = n_pr_cell*n_cells       # Number of electrons
N_i = n_pr_cell*n_cells       # Number of ions
num_species = 2               # Number of species
#-------------------------------------------------------------------------------
#                       Physical parameters
#-------------------------------------------------------------------------------
n_plasma = N_e/tot_volume   # Plasma density

epsilon_0 = 1.              # Permittivity of vacuum
mu_0 = 1.                   # Permeability of vacuum
T_e = 1.                    # Temperature - electrons
T_i = 1.                    # Temperature - ions
kB = 1.                     # Boltzmann's constant
e = 1.                      # Elementary charge
Z = 1                       # Atomic number
m_e = 1.                    # particle mass - electron
m_i = 1836.15267389         # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e         # Electric charge - electron
q_i = Z*e        # Electric charge - ions

vd_x = .10; vd_y = 0.0; dv_z = 0.0;
vd = np.array([vd_x, vd_y])  # Drift velocity
#-------------------------------------------------------------------------------
#            Create boundary conditions and function space
#-------------------------------------------------------------------------------
V = FunctionSpace(mesh, "CG", 1)
#-------------------------------------------------------------------------------
#         Get the solver
#-------------------------------------------------------------------------------
bc = DirichletBC(V, Constant(0), NonPeriodicBoundary(Ld))
poisson = PoissonSolver(V, bc)
#-------------------------------------------------------------------------------
#   Initialize particle positions and velocities, and populate the domain
#-------------------------------------------------------------------------------
pop = Population(mesh, objects=objects, dirichlet=True)
distr = Distributor(V, Ld)

pdf = [lambda x: 1, lambda x: 1]
init = Initialize(pop, pdf, Ld, vd, [alpha_e,alpha_i], 4, dt = dt)
init.initial_conditions()

# Plotting
def scatter_plot(pop, fig):
    'Scatter plot of all particles on process 0'
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    ax = fig.gca()
    p_ions = []
    p_electrons = []
    for cell in pop:
        for particle in cell:
            if np.sign(particle.q) == 1.:
                p_ions.append(particle.x)
            if np.sign(particle.q) == -1.:
                p_electrons.append(particle.x)

    cmap = cmx.get_cmap('viridis')
    if len(p_ions) > 0 :
        xy_ions = np.array(p_ions)
        ax.scatter(xy_ions[:, 0], xy_ions[:, 1],
                   label='ions',
                   marker='o',
                   c='r',
                   s = 1,
                   edgecolor='none')
    if len(p_electrons) > 0:
        xy_electrons = np.array(p_electrons)
        ax.scatter(xy_electrons[:, 0], xy_electrons[:, 1],
                   label='electrons',
                   marker = 'o',
                   c='b',
                   s = 3,
                   edgecolor='none')
    ax.legend(bbox_to_anchor=(1.09, 0.99))
    ax.axis([0, Ld[0], 0, Ld[1]])

import matplotlib.pylab as plt
fig = plt.figure()
scatter_plot(pop, fig)
fig.suptitle('Initial')
fig.show()

plt.ion()
save = True

#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
N = tot_time
KE = np.zeros(N-1)
PE = np.zeros(N-1)
KE0 = kineticEnergy(pop)
for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))
    rho, q_rho = distr.distr(pop)

    phi = poisson.solve(rho, bcs)
    E = electric_field(phi)
    PE[n-1] = potentialEnergy(pop, phi)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)
    move(pop, Ld, dt)
    tot_p = pop.total_number_of_particles()
    print("Total number of particles in the domain: ", tot_p)

    init.inject()

    tot_p = pop.total_number_of_particles()
    print("Total number of particles in the domain: ", tot_p)
    scatter_plot(pop, fig)
    fig.suptitle('At step %d' % n)
    fig.canvas.draw()

    if (save and n%1==0): plt.savefig('img%s.png' % str(n).zfill(4))

    fig.clf()

KE[0] = KE0

plot(rho)
plot(phi)
#
ux = Constant((1,0))
Ex = project(inner(E,ux),V)
plot(Ex)
interactive()

# File("rho.pvd") << rho
# File("phi.pvd") << phi
# File("E.pvd") << E
