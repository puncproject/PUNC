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
n_components = 4
msh = ObjectMesh(dim, n_components, 'spherical_object')
mesh, object_info, L = msh.mesh()

d = mesh.geometry().dim()
Ld = np.asarray(L[d:])
#-------------------------------------------------------------------------------
#           Create the objects
#-------------------------------------------------------------------------------
tol = 1e-8
objects = []
for i in range(n_components):
    j = i*(dim+1)
    s0 = object_info[j:j+dim]
    r0 = object_info[j+dim]
    func = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2+tol
    objects.append(Object(func, i))
#-------------------------------------------------------------------------------
#           Mark the facets of the boundary and the object
#-------------------------------------------------------------------------------
mk = Marker(mesh, L, objects)
facet_f = mk.markers()
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

vd_x = 0.0; vd_y = 0.0; dv_z = 0.0;
vd = [vd_x, vd_y]  # Drift velocity
#-------------------------------------------------------------------------------
#            Create boundary conditions and function space
#-------------------------------------------------------------------------------
PBC = PeriodicBoundary(Ld)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
#-------------------------------------------------------------------------------
#                  Get the object dofs
#-------------------------------------------------------------------------------
object_dofs = []
for o in objects:
    object_dofs.append(o.dofs(V, facet_f))
#-------------------------------------------------------------------------------
#          The inverse of capacitance matrix of the object
#-------------------------------------------------------------------------------
inv_capacitance = capacitance_matrix(V, mesh, facet_f, n_components, epsilon_0)
#-------------------------------------------------------------------------------
#          The circuits and the relative potential bias
#-------------------------------------------------------------------------------
circuits_info = [[0, 2], [1, 3]]
bias_1 = 0.1
bias_2 = 0.2
inv_D = circuits(inv_capacitance, circuits_info)
#-------------------------------------------------------------------------------
#         Get the solver
#-------------------------------------------------------------------------------
poisson = PoissonSolver(V)
#-------------------------------------------------------------------------------
#   Initialize particle positions and velocities, and populate the domain
#-------------------------------------------------------------------------------
pop = Population(mesh, objects)
distr = Distributor(V, Ld)

pdf = [lambda x: 1, lambda x: 1]
init = Initialize(pop, pdf, Ld, 0, [alpha_e,alpha_i], 8, objects=objects)
init.initial_conditions()
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
for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))
    rho, q_rho = distr.distr(pop, object_dofs)
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
    movePeriodic(pop, Ld, dt)
    for (i, o) in enumerate(objects):
        q_object[i] = o.charge

    bias_voltage = np.array([[bias_1, 0.0],[bias_2, 0.0]])
    for k in range(len(circuits_info)):
        for l in circuits_info[k]:
            bias_voltage[k,-1] += (q_object[l] - q_rho[l])
        for m,l in enumerate(circuits_info[k]):
            q_object[l] = np.dot(inv_D[k], bias_voltage[k])[m] + q_rho[l]

    for (i, o) in enumerate(objects):
        o.charge = q_object[i]

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
