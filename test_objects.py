from __future__ import print_function
import numpy as np
import dolfin as df
from mesh import *
from punc import *

def test_facet():
    import matplotlib.pyplot as plt

    dim = 2
    n_components = 4
    msh = ObjectMesh(dim, n_components, 'spherical_object')
    mesh, object_info, L = msh.mesh()
    d = mesh.topology().dim()

    PBC = PeriodicBoundary(L[d:])
    V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    v2d = df.vertex_to_dof_map(V)
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+len(L))
    cell_f = df.CellFunction('size_t', mesh)
    cell_f.set_all(n_components)
    tol = 1e-8
    objects = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        func = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2+tol
        objects.append(Object(func, i, mesh, facet_f, cell_f, v2d))

    for o in objects:
        o.mark_facets()
    facet_f = mark_exterior_boundaries(facet_f, n_components, L)
    df.plot(facet_f, interactive=True)

def test_capacitance():

    epsilon_0 = 1.0
    dim = 2
    n_components = 4
    circuits_info = [[0, 2], [1, 3]]

    msh = ObjectMesh(dim, n_components, 'spherical_object')
    mesh, object_info, L = msh.mesh()

    d = mesh.geometry().dim()
    Ld = np.asarray(L[d:])
    #-------------------------------------------------------------------------------
    #           Create the objects
    #-------------------------------------------------------------------------------
    PBC = PeriodicBoundary(Ld)
    V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    v2d = df.vertex_to_dof_map(V)
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+len(L))
    cell_f = df.CellFunction('size_t', mesh)
    cell_f.set_all(n_components)
    tol = 1e-8
    objects = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        func = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2+tol
        objects.append(Object(func, i, mesh, facet_f, cell_f, v2d))

    for o in objects:
        o.mark_facets()
    facet_f = mark_exterior_boundaries(facet_f, n_components, L)
    inv_capacitance = capacitance_matrix(V,
                                         mesh,
                                         facet_f,
                                         n_components,
                                         epsilon_0)

    inv_D = circuits(inv_capacitance, circuits_info)
    print("capacitance matrix: ", inv_capacitance)
    print("Bias voltage matrix: ", inv_D)

def test_object():
    dim = 2
    n_components = 1
    object_type = 'spherical_object'
    msh = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh.mesh()

    PBC = PeriodicBoundary(L[dim:])
    V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    v2d = df.vertex_to_dof_map(V)
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+len(L))
    cell_f = df.CellFunction('size_t', mesh)
    cell_f.set_all(n_components)
    tol = 1e-8

    circles = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
        circles.append(Object(fun,i, mesh, facet_f, cell_f, v2d))

    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    circles[0].inside(x, q)
    print("Circle: ", circles[0].inside, "  ", circles[0].charge)

    n_components = 4
    object_type = 'spherical_object'
    msh1 = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh1.mesh()

    objects = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
        objects.append(Object(fun,i, mesh, facet_f, cell_f, v2d))

    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    for o in objects:
        print("Circles: ", o.inside(x, q), "  ", o.charge)

    dim = 3
    Ld = [2*np.pi,2*np.pi,2*np.pi]
    n_components = 1
    object_type = 'cylindrical_object'
    msh = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh.mesh()

    s = [object_info[0], object_info[1]]
    r = object_info[2]
    h = object_info[3]
    Lz = Ld[2]
    z0 = (Lz-h)/2.      # Bottom point of cylinder
    z1 = (Lz+h)/2.      # Top point of cylinder

    fun = lambda x, z0=z0, z1=z1, s=s, r=r: ((x[2]>z0) and (x[2]<z1) and (np.dot(x[:2]-s, x[:2]-s) <= r**2))
    cylinder = Object(fun,0, mesh, facet_f, cell_f, v2d)

    x = np.array([np.pi, 0.5+np.pi,np.pi])
    q = 100
    print("Cylinder: ", cylinder.inside(x,q), "  ", cylinder.charge)

test_capacitance()
test_facet()
test_object()
