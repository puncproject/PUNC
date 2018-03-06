from dolfin import *

mesh = Mesh("../mesh/3D/sphere_in_sphere_res1.xml")
bnd  = MeshFunction("size_t", mesh, "../mesh/3D/sphere_in_sphere_res1_facet_region.xml")
ext_bnd_id = 58
int_bnd_id = 59

V = FunctionSpace(mesh, 'CG', 1)
bce = DirichletBC(V, Constant(0.0), bnd, ext_bnd_id)
bci = DirichletBC(V, Constant(1.0), bnd, int_bnd_id)

u = TrialFunction(V)
v = TestFunction(V)
phi = Function(V)

a = inner(grad(u), grad(v))*dx
L = Constant(0.0)*v*dx

solve(a==L, phi, [bci, bce])
E = project(-grad(phi))

dss = Measure('ds', domain=mesh, subdomain_data=bnd)
n = FacetNormal(mesh)

print(assemble(inner(E, -1*n) * dss(int_bnd_id)))
print(assemble(dot(E, -1*n)   * dss(int_bnd_id)))
# print(assemble(inner(E, -1.*n) * dss(int_bnd_id)))
# print(assemble(dot(E, -1.*n)   * dss(int_bnd_id)))
# print(assemble(inner(E, Constant(-1.)*n) * dss(int_bnd_id)))
# print(assemble(dot(E, Constant(-1.)*n)   * dss(int_bnd_id)))
# print(assemble(inner(E, Constant(-1)*n) * dss(int_bnd_id)))
# print(assemble(dot(E, Constant(-1)*n)   * dss(int_bnd_id)))
