from dolfin import *
from mshr import *
import numpy as np

dims = 2
Ld = np.ones(dims)
R = 0.1

domain = Rectangle(Point(0,0),Point(Ld)) - Circle(Point(Ld/2),R,20)
mesh = generate_mesh(domain,20)

V = FunctionSpace(mesh, 'CG', 1)
phi = TrialFunction(V)
phi_ = TestFunction(V)

a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
L = Constant(0.0)*phi_*dx

class Boundary(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and np.sum((x-Ld/2)**2)<R**2+DOLFIN_EPS

class Outer(SubDomain):
    def inside(self, x, on_bnd):
        return on_bnd and (
            any([near(a,0) for a in x]) or
            any([near(a,b) for a,b in zip(x,Ld)]) )

bci = DirichletBC(V, Constant(1), Boundary())
bce = DirichletBC(V, Constant(0), Outer())

bci.set_value(Constant(-1))

bcs = [bci, bce]

phi = Function(V)
solve(a==L,phi,bcs)
plot(phi)
interactive()
