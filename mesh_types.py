from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

from dolfin import *
import numpy as np

def UnitHyperCube(divisions):
    mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d-1](*divisions)
    return mesh

def HyperCube(coordinates, divisions):
    mesh_classes = [RectangleMesh, BoxMesh]
    d = len(divisions)
    mesh = mesh_classes[d-2](Point(*coordinates[:d]), Point(*coordinates[d:]),
                            *divisions)
    return mesh

def single_circle():
    return Mesh("mesh/circle.xml")

def two_circles():
    return Mesh("mesh/capacitance2.xml")

def four_circles():
    return Mesh("mesh/circuit.xml")

def single_sphere():
    return Mesh('mesh/sphere.xml')

def single_cylinder():
    return Mesh('mesh/cylinder_object.xml')

def mesh_with_object(d, n_components, object_type):

    if d == 2:
        if n_components == 1:
            mesh = single_circle()
        if n_components == 2:
            mesh = two_circles()
        if n_components == 4:
            mesh = four_circles()
    elif d == 3:
        if object_type == 'spherical_object':
            mesh = single_sphere()
        elif object_type == 'cylindrical_object':
            mesh = single_cylinder()

    d = mesh.topology().dim()
    L = np.empty(2*d)
    for i in range(d):
        l_min = mesh.coordinates()[:,i].min()
        l_max = mesh.coordinates()[:,i].max()
        L[i] = l_min
        L[d+i] = l_max

    return mesh, L


def simple_mesh(d=None, l1=None, l2=None, w1=None, w2=None, h1=None, h2=None):
    if d == None:
        # Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
        d = 2              # Space dimension
        l1 = 0.            # Start position x-axis
        l2 = 2.*np.pi      # End position x-axis
        w1 = 0.            # Start position y-axis
        w2 = 2.*np.pi      # End position y-axis
        h1 = 0.            # Start position z-axis
        h2 = 2.*np.pi      # End position z-axis
    nx = 32; ny = 32; nz = 32
    if d == 2:
        L = [l1, w1, l2, w2]
        divisions = [nx, ny]
        mesh = HyperCube(L, divisions)
    if d == 3:
        L = [l1, w1, h1, l2, w2, h2]
        divisions = [nx, ny, nz]
        mesh = HyperCube(L, divisions)
    return mesh, L

def test_Unit_mesh():
    from pylab import show, triplot
    divs = [[10,10], [10,10,10]]
    for i in range(len(divs)):
        divisions = divs[i]
        mesh = UnitHyperCube(divisions)
        coords = mesh.coordinates()
        triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
        show()
        #plot(mesh, interactive = True)

def test_mesh():
    from pylab import show, triplot
    divs = [[10,10], [10,10,10]]
    L = [[-1., -1, 2., 1.], [-1., -1, 0, 2., 1., 1.]]
    for i,j in zip(L, divs):
        L = i
        divisions = j
        mesh = HyperCube(L, divisions)
        coords = mesh.coordinates()
        triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
        show()
        #plot(mesh, interactive = True)

if __name__=='__main__':

    test_Unit_mesh()
    test_mesh()
