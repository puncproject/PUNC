from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
import numpy as np
from punc import *

from get_object import *
from initial_conditions import *
from mesh_types import *

#-------------------------------------------------------------------------------
#                   Object type
#-------------------------------------------------------------------------------
random_domain = 'box'
initial_type = 'spherical_object'
object_type = 'spherical_object'
#-------------------------------------------------------------------------------
#                  Get the mesh
#-------------------------------------------------------------------------------
dim = 2
n_components = 1
mesh, L = mesh_with_object(dim, n_components, object_type)
d = mesh.geometry().dim()
Ld = [L[d],L[d+1]]
#-------------------------------------------------------------------------------
#           Mark the facets of the boundary and the object
#-------------------------------------------------------------------------------
object_info = get_object(d, object_type, n_components)
facet_f = mark_boundaries(mesh, L, object_type, object_info, n_components)
#-------------------------------------------------------------------------------
#                       Simulation parameters
#-------------------------------------------------------------------------------
n_pr_cell = 8             # Number of particels per cell
n_pr_super_particle = 8   # Number of particles per super particle
tot_time = 20             # Total simulation time
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
w = (L[d]*L[d+1])/N_e  # Non-dimensionalization factor

vd_x = 0.0; vd_y = 0.0; dv_z = 0.0;

if d == 2:
    vd = [vd_x, vd_y]
if d == 3:
    vd = [vd_x, vd_y, vd_z]

sigma_e, sigma_i, mu_e, mu_i = [], [], [], []
for i in range(d):
    sigma_e.append(alpha_e)
    sigma_i.append(alpha_i)
    mu_e.append(vd[i])
    mu_i.append(vd[i])
#-------------------------------------------------------------------------------
#            Create boundary conditions and function space
#-------------------------------------------------------------------------------
PBC = PeriodicBoundary(Ld)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
#-------------------------------------------------------------------------------
#                  Get the object dofs
#-------------------------------------------------------------------------------
object_dofs = objects_dofs(V, facet_f, n_components)
#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i, mu_e, mu_i, sigma_e,
                   sigma_i, object_info, random_domain, initial_type)
#-------------------------------------------------------------------------------
#          The inverse of capacitance matrix of the object
#-------------------------------------------------------------------------------
inv_capacitance = capacitance_matrix(V, mesh, facet_f, n_components, epsilon_0)
#-------------------------------------------------------------------------------
#         Get the solver
#-------------------------------------------------------------------------------
poisson = PoissonSolverPeriodic(V)
#-------------------------------------------------------------------------------
#             Add particles to the mesh
#-------------------------------------------------------------------------------
pop = Population(mesh, object_type, object_info)
distr = Distributor(V, Ld)
#-------------------------------------------------------------------------------
#             Add electrons to population
#-------------------------------------------------------------------------------
xs = initial_positions[:N_e]
vs = initial_velocities[:N_e]
q = properties['q'][0]
m = properties['m'][0]
pop.addParticles(xs,vs,q,m)
#-------------------------------------------------------------------------------
#             Add ions to population
#-------------------------------------------------------------------------------
xs = initial_positions[N_e:]
vs = initial_velocities[N_e:]
q = properties['q'][-1]
m = properties['m'][-1]
pop.addParticles(xs,vs,q,m)
#-------------------------------------------------------------------------------
#             Initial object charge
#-------------------------------------------------------------------------------
q_diff = []
q_object = []
for i in range(n_components):
    q_diff.append(Constant(0.0))
    q_object.append(0.0)
#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
N = tot_time
KE = np.zeros(N-1)
PE = np.zeros(N-1)
KE0 = kineticEnergy(pop)
Ld = [L[d], L[d+1]]
for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))
    rho, q_rho = distr.distr(pop, n_components, object_dofs)
    print("Interpolated object charge: ", q_rho)

    object_bcs = []
    for k in range(n_components):
        phi_object = 0.0
        for j in range(n_components):
            phi_object += (q_object[j]-q_rho[j])*inv_capacitance[k,j]
        q_diff[k].assign(phi_object)
        object_bcs.append(DirichletBC(V, q_diff[k], facet_f, k))

    phi = poisson.solve(rho, object_bcs)
    E = electric_field(phi)
    PE[n-1] = potentialEnergy(pop, phi)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)
    q_object = movePeriodic(pop, Ld, dt, q_object)
    print("Collected object charge: ", q_rho)
    # tot_p = pop.total_number_of_particles()
    # print("Total number of particles in the domain: ", tot_p)

KE[0] = KE0

# plt.plot(KE,label="Kinetic Energy")
# plt.plot(PE,label="Potential Energy")
# plt.plot(KE+PE,label="Total Energy")
# plt.legend(loc='lower right')
# plt.grid()
# plt.xlabel("Timestep")
# plt.ylabel("Normalized Energy")
# plt.show()

plot(rho)
plot(phi)

ux = Constant((1,0))
Ex = project(inner(E,ux),V)
plot(Ex)
interactive()
