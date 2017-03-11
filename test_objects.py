from __future__ import print_function
import numpy as np
import dolfin as df
from mesh_types import *
from punc import *

def tests():
    dim = 2
    n_components = 1
    object_type = 'spherical_object'
    msh = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh.mesh()
    # object_info = get_object(dim, object_type, n_components)

    circles = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
        circles.append(Object(fun))

    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    circles[0].is_inside(x, q)
    print("Circle: ", circles[0].inside, "  ", circles[0].charge)

    n_components = 4
    object_type = 'spherical_object'
    msh1 = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh1.mesh()
    # object_info = get_object(dim, object_type, n_components)

    objects = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
        objects.append(Object(fun))

    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    for o in objects:
        o.is_inside(x, q)
        print("Circles: ", o.inside, "  ", o.charge)

    dim = 3
    Ld = [2*np.pi,2*np.pi,2*np.pi]
    n_components = 1
    object_type = 'cylindrical_object'
    msh = ObjectMesh(dim, n_components, object_type)
    mesh, object_info, L = msh.mesh()
    # object_info = get_object(dim, object_type, n_components)

    s = [object_info[0], object_info[1]]
    r = object_info[2]
    h = object_info[3]
    Lz = Ld[2]
    z0 = (Lz-h)/2.      # Bottom point of cylinder
    z1 = (Lz+h)/2.      # Top point of cylinder

    fun = lambda x, z0=z0, z1=z1, s=s, r=r: ((x[2]>z0) and (x[2]<z1) and (np.dot(x[:2]-s, x[:2]-s) <= r**2))
    cylinder = Object(fun)

    x = np.array([np.pi, 0.5+np.pi,np.pi])
    q = 100
    cylinder.is_inside(x, q)
    print("Cylinder: ", cylinder.inside, "  ", cylinder.charge)

tests()
