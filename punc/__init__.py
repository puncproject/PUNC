"""
__all__ = [ "population",
            "pusher",
            "distributor",
            "poisson",
            "diagnostics"]
"""

from punc.poisson import *
from punc.pusher import *
from punc.distributor import *
from punc.population import *
from punc.diagnostics import *
from punc.object import *
from punc.initialize import *
from punc.capacitance import *


"""

Ld = ...
mesh = ...

V = FunctionSpace(mesh, 'CG', 1, constr=constr)
Vv = VectorFunctionSpace(mesh, 'CG', 1, constr=constr)

E = Function(Vv)
rho = Function(V)
phi = Function(V)


fsolver = PeriodicPoissonSolver(V)
pop = Population()
acc = Accelerator()
mov = Mover()
dist = Distributer()
objects = Objects()
# charge = [c1, c2, c3] # int
# inside = [f1, f2, f3] # func -> Bool

# Mark objects

# Adding electrons
vd, vth = ..., ...
q, m, N = stdSpecie(mesh, Ld, -1, 1, 8)
xs = randomPoints(lambda x: ..., Ld, N)
vs = maxwellian(vd, vth)
pop.addParticles(xs,vs,q,m)

# Adding ions
vd, vth = ..., ...
q, m, N = stdSpecie(mesh, Ld, -1, 1, 8)
xs = randomPoints(lambda x: ..., Ld, N)
vs = maxwellian(vd, vth)
pop.addParticles(xs,vs,q,m)

# x, v given in n=0

KE0 = kinEnergy(pop)

def func(x):
    # return i if inside object i, 0 otherwise

Nt = ...
for n in range(1,Nt):
    dist.dist(pop,RHO)
    bcs = boundaryConds(...)
    PHI = fsolver.solve(RHO)        # PHI = fsolver.dirichlet_solver(RHO,bcs)
    E = gradient(PHI)
    PE[n-1] = potEnergy(RHO,PHI)
    KE[n-1] = acc.acc(pop,E,(1-0.5*(n==1))*dt)  # v now at step n-0.5
    mov.move(pop)                               # x now at step n
    pop.relocate(objects)
    objCurrent(...)
    inject(...)
    delete(...)

KE[0] = KE0

"""
