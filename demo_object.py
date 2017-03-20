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
#              Simulation and physical  parameters
#-------------------------------------------------------------------------------
tot_time = 20             # Total simulation time
dt = 0.251327             # Time step
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

vd_x = 0.0; vd_y = 0.0;
vd = np.array([vd_x, vd_y])  # Drift velocity
#-------------------------------------------------------------------------------
#             Get the mesh and the object
#-------------------------------------------------------------------------------
mesh, circles = get_mesh_circle()
Ld = get_mesh_size(mesh)
#-------------------------------------------------------------------------------
#          The inverse of capacitance matrix of the object
#-------------------------------------------------------------------------------
inv_cap_matrix = capacitance_matrix(mesh, Ld, circles, epsilon_0)
#-------------------------------------------------------------------------------
#            Create boundary conditions and function space
#-------------------------------------------------------------------------------
PBC = PeriodicBoundary(Ld)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)

objects = [None]*len(circles)
for i, c in enumerate(circles):
    objects[i] = Object(V, c)
#-------------------------------------------------------------------------------
#         Get the solver
#-------------------------------------------------------------------------------
poisson = PoissonSolver(V)
#-------------------------------------------------------------------------------
#   Initialize particle positions and velocities, and populate the domain
#-------------------------------------------------------------------------------
pop = Population(mesh)
distr = Distributor(V, Ld)

pdf = [lambda x: 1, lambda x: 1]
init = Initialize(pop, pdf, Ld, vd, [alpha_e,alpha_i], 8, objects=objects)
init.initial_conditions()
#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
N = tot_time
KE = np.zeros(N-1)
PE = np.zeros(N-1)
KE0 = kineticEnergy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))

    q = distr.distr(pop)

    compute_object_potentials(q, objects, inv_cap_matrix)

    rho = distr.charge_density(q)

    phi = poisson.solve(rho, objects)
    E = electric_field(phi) 
    PE[n-1] = potentialEnergy(pop, phi)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)
    movePeriodic(pop, Ld, dt)
    pop.relocate(objects)

KE[0] = KE0

plot(rho)
plot(phi)

ux = Constant((1,0))
Ex = project(inner(E,ux),V)
plot(Ex)
interactive()
