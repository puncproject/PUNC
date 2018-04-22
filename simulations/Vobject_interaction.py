import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
from dolfin import *
import os
import sys

# df.set_log_level(df.WARNING)

# NB: This is generally a discouraged technique, but allows great flexibility
# in the config file and is therefore used at least temporarily. That it is
# unsafe is not considered a problem since it's currently supposed to be used
# only by expert users.
exec('from %s import *'%(sys.argv[1]))

assert object_method in ['capacitance', 'variational']

V = df.FunctionSpace(mesh, 'CG', 1)

bc = df.DirichletBC(V, df.Constant(0.0), bnd, ext_bnd_id)


int_bnd_id = int_bnd_ids[0]

if object_method=='capacitance':
    objects = [Object(V, bnd, i) for i in int_bnd_ids]
else:
    objects = [Object(V, bnd, i) for i in int_bnd_ids]
    # objects[0].set_potential(imposed_potential)

# Get the solver
poisson = PoissonSolver(V, bc, eps_0=eps0)
esolver = ESolver(V)

if object_method=='capacitance':
    # The inverse of capacitance matrix
    inv_cap_matrix = capacitance_matrix(V, poisson, objects, bnd, ext_bnd_id)
    inv_cap_matrix /= cap_factor

pop = Population(mesh, bnd)

print("Loading particles")
if os.path.isfile('stop'):
    os.remove('stop')
    pop.load_file('population.dat')
    f = open('state.dat','r')
    nstart = int(f.readline()) + 1
    objects[0].charge = float(f.readline())
    f.close()
    hist_file = open('history.dat', 'a')

else:
    load_particles(pop, species)
    nstart = 0
    hist_file = open('history.dat', 'w')

dv_inv = voronoi_volume_approx(V)

KE  = np.zeros(N-1)
PE  = np.zeros(N-1)

num_e = pop.num_of_negatives()
num_i = pop.num_of_positives()
potential = 0
current_measured = 0

timer = TaskTimer(N, 'compact')

Q = -10000.
# objects[0].charge=Q
Q = df.Constant(Q)

cell = mesh.ufl_cell()
WV = FiniteElement("Lagrange", cell, 1)
WR = FiniteElement("Real", cell, 0)

W = FunctionSpace(mesh, MixedElement([WV, WR]))
u, c = TrialFunctions(W)
v, d = TestFunctions(W)

ext_bc = DirichletBC(W.sub(0), Constant(0), bnd, ext_bnd_id)
int_bc = FloatingBC(W.sub(0), bnd, int_bnd_id)
d2v = df.dof_to_vertex_map(V)
pot_index = d2v[bnd.where_equal(int_bnd_id)[0]]

normal = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

S = assemble(Constant(1.)*dsi)

a = inner(grad(u), grad(v)) * dx -\
    inner(v, dot(grad(u), normal)) * dsi +\
    inner(c, dot(grad(v), normal)) * dsi +\
    inner(d, dot(grad(u), normal)) * dsi

print("Applying boundaries")
A = assemble(a)
ext_bc.apply(A)
int_bc.apply(A)

wh = Function(W)

for n in range(nstart, N):

    # We are now at timestep n
    # Velocities and currents are at timestep n-0.5 (or 0 if n==0)

    timer.task("Distribute charge")
    rho = distribute(V, pop, dv_inv)
    # rho = df.Expression('10*x[0]',degree=2)

    if object_method == 'capacitance':

        timer.task("Solving potential (cap. matrix)")

        reset_objects(objects)
        phi = poisson.solve(rho, objects)

        E = esolver.solve(phi)
        compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
        phi = poisson.solve(rho, objects)

    if object_method == 'variational':

        timer.task("Solving potential (variational)")

        L = inner(rho, v) * dx +\
            inner(Q/S, d) * dsi

        b = assemble(L)
        ext_bc.apply(b)
        int_bc.apply(b)

        solve(A, wh.vector(), b)
        phi, ph = wh.split(deepcopy=True)

    # There's something wrong with pot_index
    # potential = phi.vector().get_local()[pot_index]*Vnorm
    potential = phi(0,0,1)*Vnorm

    # Qm = assemble(dot(grad(uh), normal) * dsi)
    # print("Object charge: ", Qm)

    # rel_error = df.errornorm(uh,phi)/df.norm(phi)
    # print("Relative error, phi:",rel_error)

    timer.task("Solving E-field")
    E = esolver.solve(phi)

    # rel_error = df.errornorm(E,E2)/df.norm(E2)
    # print("Relative error, E:",rel_error)


    timer.task("Computing potential energy")
    PE = mesh_potential_energy(rho, phi)

    timer.task("Move particles")
    old_charge = objects[0].charge # Charge at n
    KE = accel(pop, E, (1-0.5*(n==1))*dt) # Advancing velocities to n+0.5
    if n==0: KE = kinetic_energy(pop)
    move(pop, dt) # Advancing position to n+1

    timer.task("Updating particles")
    pop.update(objects)
    Q.assign(objects[0].charge)

    timer.task("Write history")
    hist_file.write("%d\t%d\t%d\t%f\t%f\t%f\t%f\n"%(n, num_e, num_i, KE, PE, potential, current_measured))
    hist_file.flush()

    timer.task("Impose current")
    current_measured = ((objects[0].charge-old_charge)/dt)*Inorm # at n+0.5
    objects[0].charge -= current_collected*dt

    timer.task("Inject particles")
    inject_particles(pop, species, ext_bnd, dt)

    timer.task("Count particles")
    num_i = pop.num_of_positives()
    num_e = pop.num_of_negatives()

    if os.path.isfile('stop') or n==N-1:
        pop.save_file('population.dat')
        f = open('state.dat','w')
        f.write("%d\n%f"%(n,objects[0].charge))
        f.close()
        break

    timer.end()

timer.summary()
hist_file.close()

df.File('phi.pvd') << phi
df.File('rho.pvd') << rho
df.File('E.pvd') << E
