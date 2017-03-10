from __future__ import print_function
import numpy as np
import sympy.geometry as geo
import os

def test_single_circle():
    x0 = np.pi
    y0 = np.pi
    r = 0.5
    s = geo.Point(x0, y0)
    return geo.Circle(s, r)

def test_four_circles():
    r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
    x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
    y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
    z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
    s0 = geo.Point(x0, y0)
    s1 = geo.Point(x1, y1)
    s2 = geo.Point(x2, y2)
    s3 = geo.Point(x3, y3)
    c = [geo.Circle(s0, r0),geo.Circle(s1, r1),geo.Circle(s2, r2),geo.Circle(s3, r3)]
    return c

def single_circle():
    x0 = np.pi
    y0 = np.pi
    r0 = 0.5
    return [x0, y0, r0]

def two_circles():
    r0 = 0.5; r1 = 0.5;
    x0 = np.pi; x1 = np.pi;
    y0 = np.pi; y1 = np.pi + 3*r1;
    z0 = np.pi; z1 = np.pi;
    return [x0, y0, r0, x1, y1, r1]

def four_circles():
    r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
    x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
    y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
    z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
    return [x0, y0, r0, x1, y1, r1, x2, y2, r2, x3, y3, r3]

def single_sphere():
    x0 = np.pi
    y0 = np.pi
    z0 = np.pi
    r0 = 0.5
    return [x0, y0, z0, r0]

def two_spheres():
    print("Not implemented yet")

def single_cylinder():
    x0 = np.pi
    y0 = np.pi
    r0 = 0.5
    h0 = 1.0
    return [x0, y0, r0, h0]

def get_object(d, object_type, n_components):
    if object_type == 'spherical_object':
        if d == 2:
            object_info = single_circle()
        if d == 3:
            object_info = single_sphere()
    if object_type == 'cylindrical_object':
        object_info = single_cylinder()
    if object_type == 'multi_circles':
        if n_components == 2:
            object_info = two_circles()
        if n_components == 4:
            object_info = four_circles()
    if object_type == 'multi_spheres':
        print("Not implemented yet :( ")
        sys.exit()
    return object_info
