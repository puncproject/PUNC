from __future__ import print_function
import numpy as np

class Object:
    """This class is used to find whether a particle is inside an object or not.
    If the particle is inside the object it returns 'True' and adds the charge
    of the particle to the accumulated charge of the object.
    """
    def __init__(self, geometry):
        self.geometry = geometry
        self.inside = False
        self.charge = 0.0

    def is_inside(self, p, q):
        if self.geometry.is_inside(p):
            self.inside = True
            self.charge += q
        else:
            self.inside = False

class Sphere:
    """This class is used to find whether a particle is inside a spherical
    object or not. For 2D simulations the object is simply a circle.
    """
    def __init__(self, object_info):
        self.s = [i for i in object_info[:-1]]
        self.r = object_info[2]

    def is_inside(self, p):
        if np.dot(p-self.s, p-self.s) <= self.r**2:
            return True
        else:
            return False

class Cylinder:
    """This class is used to find whether a particle is inside a cylindrical
    object or not.

    Note: By default a cylinder is a 3D object.
    """
    def __init__(self, object_info, Ld):
        assert len(Ld) == 3, "By default a cylinder is a 3D object."
        self.s = [object_info[0], object_info[1]]
        self.r = object_info[2]
        self.h = object_info[3]
        self.Lz = Ld[2]
        self.z0 = (self.Lz-self.h)/2.      # Bottom point of cylinder
        self.z1 = (self.Lz+self.h)/2.      # Top point of cylinder

    def is_inside(self, p):
        if (p[2] > self.z0 and
            p[2] < self.z1 and
            np.dot(p[:2]-self.s, p[:2]-self.s) <= self.r**2):
            return True
        else:
            return False

# objects = [Object(lambda x: ...), Object(lambda x: ...)]


if __name__=='__main__':

    import sys
    from os.path import dirname, abspath
    d = dirname(dirname(abspath(__file__)))
    sys.path.insert(0, d)
    from get_object import *

    dim = 2
    n_components = 1
    object_type = 'spherical_object'
    object_info = get_object(dim, object_type, n_components)
    C1 = Sphere(object_info)
    objects = Object(C1)
    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    objects.is_inside(x, q)
    print("Circle: ", objects.inside, "  ", objects.charge)

    n_components = 4
    object_type = 'multi_circles'
    object_info = get_object(dim, object_type, n_components)

    objects = []
    for i in range(n_components):
        objects.append(Object(Sphere(object_info[3*i:3*(i+1)])))

    x = np.array([np.pi, 0.5+np.pi])
    q = 10
    for o in objects:
        o.is_inside(x, q)
        print("Circles: ", o.inside, "  ", o.charge)

    dim = 3
    Ld = [2*np.pi,2*np.pi,2*np.pi]
    n_components = 1
    object_type = 'cylindrical_object'
    object_info = get_object(dim, object_type, n_components)

    objects = []
    for i in range(n_components):
        objects.append(Object(Cylinder(object_info[4*i:4*(i+1)],Ld)))

    x = np.array([np.pi, 0.5+np.pi,np.pi])
    q = 10
    for o in objects:
        o.is_inside(x, q)
        print("Cylinder: ", o.inside, "  ", o.charge)
