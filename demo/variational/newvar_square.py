from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

resolutions = [10, 20, 40, 80, 160]#, 320, 640, 1280]
order = 2
ll = 1.0
l = 0.2

EPS = DOLFIN_EPS
class ConstantBoundary(SubDomain):

    def inside(self, x, on_bnd):
        # x = x[:2] # trim away third element if necessary
        on_vertex = np.linalg.norm(x-[l,l])<EPS
        return on_vertex

    def map(self, x, y):
        on_square = np.abs(x[0])<l+EPS and np.abs(x[1])<l+EPS
        if on_square:
            y[0] = l
            y[1] = l
        else:
            y[0] = x[0]
            y[1] = x[1]

class InnerBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.abs(x[0])<l+EPS and np.abs(x[1])<l+EPS

class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.abs(x[0])>ll-EPS or np.abs(x[1])>ll-EPS

ext_bnd_id = 1
int_bnd_id = 2

domain = Rectangle(Point(-ll,-ll),Point(ll,ll)) \
       - Rectangle(Point(-l,-l),Point(l,l))

#
# REFERENCE SOLUTION (STANDARD POISSON)
#
mesh = generate_mesh(domain, resolutions[-1])

outer_bnd  = OuterBoundary()
inner_bnd  = InnerBoundary()

bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
bnd.set_all(0)
outer_bnd.mark(bnd, ext_bnd_id)
inner_bnd.mark(bnd, int_bnd_id)

rho = Expression("100*x[0]", degree=order+1)
n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
dsi = dss(int_bnd_id)

W = FunctionSpace(mesh, 'CG', order+1)
u = TrialFunction(W)
v = TestFunction(W)

ext_bc = DirichletBC(W, Constant(0), bnd, ext_bnd_id)
int_bc = DirichletBC(W, Constant(1), bnd, int_bnd_id)

a = inner(grad(u), grad(v))*dx
L = inner(rho, v)*dx

wh1 = Function(W)

A, b = assemble_system(a, L, [ext_bc, int_bc])

print("Solving pure Poisson problem, resolution: {}".format(resolutions[-1]))
solver = PETScKrylovSolver('gmres','hypre_amg')
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = True
solver.set_operator(A)
solver.solve(wh1.vector(), b)

Q1 = assemble(dot(grad(wh1), n) * dsi)
error = []
hmin = []

for resolution in resolutions:

    mesh = generate_mesh(domain, resolution)

    outer_bnd  = OuterBoundary()
    inner_bnd  = InnerBoundary()
    constraint = ConstantBoundary()

    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, ext_bnd_id)
    inner_bnd.mark(bnd, int_bnd_id)

    rho = Expression("100*x[0]", degree=order+1)
    n = FacetNormal(mesh)
    dss = Measure("ds", domain=mesh, subdomain_data=bnd)
    dsi = dss(int_bnd_id)

    W = FunctionSpace(mesh, 'CG', order, constrained_domain=constraint)
    # W = FunctionSpace(mesh, 'CG', order)
    u = TrialFunction(W)
    v = TestFunction(W)

    ext_bc = DirichletBC(W, Constant(0), bnd, ext_bnd_id)
    int_bc = DirichletBC(W, Constant(1), bnd, int_bnd_id)

    S = assemble(Constant(1.)*dsi)

    a = inner(grad(u), grad(v))*dx
    # L = inner(rho, v)*dx
    L = inner(rho, v)*dx + v*(Constant(Q1)/S)*dsi

    wh2 = Function(W)

    # A, b = assemble_system(a, L, [ext_bc, int_bc])
    A, b = assemble_system(a, L, ext_bc)
    print("A.is_symmetric(1e-10) =", A.is_symmetric(1e-10))

    print("Solving Poisson problem with new method. resolution: {}".format(resolution))
    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = True
    solver.set_operator(A)
    solver.solve(wh2.vector(), b)

    Q2 = assemble(dot(grad(wh2), n) * dsi)

    print("Computing error")
    line = np.linspace(l, ll, 10000, endpoint=False)
    wh1_line = np.array([wh1(x,0) for x in line])
    wh2_line = np.array([wh2(x,0) for x in line])
    dl = line[1] - line[0]
    e_abs = np.sqrt(dl*np.sum((wh1_line-wh2_line)**2))
    e_rel1 = e_abs/np.sqrt(dl*np.sum(wh2_line**2))
    e_rel2 = np.sqrt(dl*np.sum(((wh1_line-wh2_line)/wh2_line)**2))

    hmin.append(mesh.hmin())
    # error.append(errornorm(wh1,wh2)) # slow. Give similar results up to 320.
    error.append(e_rel1)

hmin = np.array(hmin)
error = np.array(error)
p = plt.loglog(hmin, error, 'o-')
color = p[0].get_color()
plt.loglog(hmin, error[0]*(hmin/hmin[0])**3, ':', color=color)
plt.loglog(hmin, error[0]*(hmin/hmin[0])**2, ':', color=color)
plt.loglog(hmin, error[0]*(hmin/hmin[0]), ':', color=color)
plt.grid()
plt.show()
