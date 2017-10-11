from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *

# Simulation parameters
tot_time = 1000                     # Total simulation time
dt       = 0.5                      # Time step
npc      = 8
# vd       = np.array([0.0, 0.0])  # Drift velocity

# Get the mesh
mesh   = df.Mesh('mesh/lafram_coarse.xml')
ext_boundaries = df.MeshFunction("size_t", mesh, "mesh/lafram_coarse_facet_region.xml")
# mesh   = df.Mesh('mesh/lafram.xml')
# ext_boundaries = df.MeshFunction("size_t", mesh, "mesh/lafram_facet_region.xml")
bnd_id = 53

ext_bnd = ExteriorBoundaries(ext_boundaries, bnd_id)

Ld     = get_mesh_size(mesh)

# Create boundary conditions and function space
periodic = [False, False, False]
bnd = NonPeriodicBoundary(Ld, periodic)
constr = PeriodicBoundary(Ld, periodic)

V = df.FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

bc = df.DirichletBC(V, df.Constant(0.0), bnd)

# Get the object
class Probe(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and np.all(np.linalg.norm(x-Ld/2)<1.1)

objects = [Object(V, Probe())]

facet_func = markers(mesh, objects)
ds = df.Measure('ds', domain = mesh, subdomain_data = facet_func)
normal = df.FacetNormal(mesh)

# Get the solver
poisson = PoissonSolver(V, bc)

# The inverse of capacitance matrix
inv_cap_matrix = capacitance_matrix(V, poisson, bnd, objects)
print("capacitance: ", 1.0/inv_cap_matrix[0,0])
epsilon_0 = 1.0
r = 1.0
R = 5.0
C_sphere = 4.0*np.pi*epsilon_0*r*R/(R-r)
print("Analytical Sphere capacitance: ", C_sphere)
# Probe radius in terms of Debye lengths
Rp = 1.

Vnorm = Rp**(-2)
#Inorm = np.sqrt(8*np.pi/1836.)/Rp
Inorm = np.sqrt(8*np.pi)/Rp

# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh, periodic)
pop.init_new_specie('electron', ext_bnd, normalization='particle scaling', v_thermal=1./Rp, num_per_cell=npc)
pop.init_new_specie('proton', ext_bnd,   normalization='particle scaling', v_thermal=1./(np.sqrt(1836.)*Rp), num_per_cell=npc)

dv_inv = voronoi_volume_approx(V, Ld)

# injection = []
num_total = 0
# n_plasma = [None]*2

# for i in range(len(pop.species)):
#     # n_plasma[i] = pop.species[i].num_total
#     # weight = pop.species.weight
#     num_total += pop.species[i].num_total
#     injection.append(Injector(pop, i, dt))

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kinetic_energy(pop)

current_collected = -1.987*Inorm
current_measured = np.zeros(N)
potential = np.zeros(N)
particles = np.zeros(N)

num_particles = np.zeros(N)
num_particles_outside = np.zeros(N)
num_injected_particles = np.zeros(N)
num_particles[0] = pop.total_number_of_particles()[0]


timer = TaskTimer(N-1,'compact')
num_e = np.zeros(N)
num_i = np.zeros(N)
num_e[0] = num_particles[0]/2
num_i[0] = num_particles[0]/2

for n in range(1,N):

    timer.task("Distribute charge")
    rho = distribute(V, pop)
    # compute_object_potentials(rho, objects, inv_cap_matrix)
    rho.vector()[:] *= dv_inv

    timer.task("Calculate potential")
    objects[0].set_potential(df.Constant(0.0))
    phi     = poisson.solve(rho, objects)
    E       = electric_field(phi)
    obj_flux = df.inner(E, -1*normal)*ds(0)
    image_charge = df.assemble(obj_flux)

    object_potential = (objects[0].charge-image_charge)*inv_cap_matrix[0,0]
    objects[0].set_potential(df.Constant(object_potential))

    timer.task("Solving Poisson")
    phi     = poisson.solve(rho, objects)
    E       = electric_field(phi)
    PE[n-1] = potential_energy(pop, phi)

    timer.task("Move particles")
    KE[n-1] = accel(pop, E, (1-0.5*(n==1))*dt)
    tot_num0 = pop.total_number_of_particles()[0]
    move(pop, Ld, dt)

    timer.task("Relocating particles")
    old_charge = objects[0].charge
    pop.relocate(objects, open_bnd=True)

    timer.task("Impose current")
    tot_num1 = pop.total_number_of_particles()[0]
    num_particles_outside[n] = tot_num0 - tot_num1
    objects[0].add_charge(-current_collected*dt)
    current_measured[n] = ((objects[0].charge-old_charge)/dt)/Inorm
    potential[n] = objects[0]._potential/Vnorm
    particles[n] = pop.total_number_of_particles()[0]

    timer.task("Inject particles")
    inject(pop, ext_bnd, dt)

    timer.task("Count particles")
    tot_num2 = pop.total_number_of_particles()[0]
    # Total number of injected particles:
    num_injected_particles[n] = tot_num2 - tot_num1
    # Total number of particles after injection:
    num_particles[n] = tot_num2

    for cell in pop:
        for particle in cell:
            if np.sign(particle.q) == 1:
                num_i[n] +=1
            else:
                num_e[n] +=1

    timer.end()

timer.summary()


KE[0] = KE0

plt.figure()
plt.plot(potential,label='potential')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('potential.png', format='png', dpi=1000)

plt.figure()
plt.plot(particles,label='number of particles')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('particles.png', format='png', dpi=1000)

plt.figure()
plt.plot(current_measured,label='current collected')
plt.grid()
plt.legend(loc="lower right")
plt.savefig('current.png', format='png', dpi=1000)

plt.figure()
plt.plot(num_particles, label="Total number denisty")
plt.legend(loc='lower right')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Total number denisty")
plt.savefig('total_num.png', format='png', dpi=1000)

plt.figure()
plt.plot(num_injected_particles[1:], label="Number of injected particles")
plt.plot(num_particles_outside[1:], label="Number of particles leaving the domain")
plt.legend(loc='lower right')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Number of particles")
plt.savefig('injected.png', format='png', dpi=1000)

plt.figure()
plt.plot(num_i, label="Number of ions")
plt.plot(num_e, label="Number of electrons")
plt.legend(loc='lower right')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Number of particles")
plt.savefig('e_i_numbers.png', format='png', dpi=1000)

# np.savetxt('data1.txt', (potential, num_e, num_i))
# np.savetxt('data2.txt', (num_particles_outside, num_injected_particles, particles, current_measured))

to_file = open('data.txt', 'w')
for i,j,k,l,m,n,o in zip(potential, num_e, num_i,num_particles_outside, num_injected_particles, particles, current_measured ):
    to_file.write("%f %f %f %f %f %f %f\n" %(i, j, k, l, m, n, o))
to_file.close()

df.File('phi_laframboise.pvd') << phi
df.File('rho_laframboise.pvd') << rho
df.File('E_laframboise.pvd') << E

plt.show()

# df.plot(rho)
# df.plot(phi)

# ux = df.Constant((1,0,0))
# Ex = df.project(df.inner(E, ux), V)
# df.plot(Ex)
# df.interactive()
