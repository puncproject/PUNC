from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

import dolfin as df
import numpy as np
from punc import *

def unit_mesh(N):

	d = len(N)
	mesh_types = [df.UnitIntervalMesh,
	     		  df.UnitSquareMesh,
				  df.UnitCubeMesh]

	return mesh_types[d-1](*N)

def simple_mesh(Ld, N):

	d = len(N)
	mesh_types = [df.RectangleMesh, df.BoxMesh]

	return mesh_types[d-2](df.Point(0,0,0), df.Point(*Ld), *N)

def get_mesh_size(mesh):
	"""
	Returns a vector containing the size of the mesh presuming the mesh is
	rectangular and starts in the origin.
	"""
	return np.max(mesh.coordinates(),0)

def get_mesh_circle():

	mesh = df.Mesh("mesh/circle.xml")

	s0, r0 = np.array([np.pi, np.pi]), 0.5

	tol = 1e-8
	class Circle(df.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and np.dot(x-s0, x-s0) <= r0**2+tol

	objects = [Circle()]

	return mesh, objects

def get_circles(circles, ind):

	r = 0.5
	s = [np.array([np.pi, np.pi]), np.array([np.pi, np.pi + 3*r]),
	     np.array([np.pi, np.pi - 3*r]), np.array([np.pi + 3*r, np.pi])]

	tol = 1e-8

	class Circle(df.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and func(x)

	func = lambda x, s = s[ind], r = r: np.dot(x-s, x-s) <= r**2+tol
	return Circle()

def get_mesh_circuit():

	mesh = df.Mesh("mesh/circuit.xml")

	circuits_info = [[0, 2], [1, 3]]
	bias_1 = [0.1]
	bias_2 = [0.2]
	bias_potential = [bias_1, bias_2]

	n_components = 4
	circles = [None]*n_components
	for i in range(n_components):
		circles[i] = get_circles(circles[i], i)

	return mesh, circles, circuits_info, bias_potential

# Not complete
def get_mesh_sphere():

	mesh = df.Mesh("mesh/sphere.xml")
	object_info = [np.pi, np.pi, np.pi, 0.5]

	return mesh

# Not complete
def get_mesh_cylinder():

	mesh = df.Mesh('mesh/cylinder_object.xml')
	object_info = [np.pi, np.pi, 0.5, 1.0]
	return mesh

if __name__=='__main__':

	mesh, circles = get_mesh_circuit()

	dim = mesh.geometry().dim()

	Ld = get_mesh_size(mesh)

	V = df.FunctionSpace(mesh,'CG',1)

	objects = [None]*len(circles)
	for i, c in enumerate(circles):
	    objects[i] = Object(V, c)

	# from IPython import embed; embed()

	phi = [5,10,15, 20]
	for i, obj in enumerate(objects):
		obj.set_potential(phi[i])
		print("phi: ", obj.potential)
		print("inside: ", obj.inside([np.pi, np.pi], True))
		dof = obj.get_boundary_values().keys()#dofs
		val = obj.get_boundary_values().values()
		print("dofs: ", dof)
		print("vals: ", val)

		f = df.Function(V)
		f.vector()[dof] = 5
		df.plot(f, interactive=True)
		obj.set_q_rho(f)
		print("q_rho: ", obj.q_rho)


	def test_simple():
		from pylab import axis, show, triplot
		Ld = [1., 2.]
		N = [20, 10]
		mesh = simple_mesh(Ld, N)
		coords = mesh.coordinates()
		triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
		axis('equal')
		show()

	def test_unit():
		from pylab import axis, show, triplot
		N = [20, 10]
		mesh = unit_mesh(N)
		coords = mesh.coordinates()
		triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
		axis('equal')
		show()

	def test_objects():
		from pylab import axis, show, triplot
		d = 3
		n_components = 1
		object_type = 'spherical_object' # 'cylindrical_object'
		msh = ObjectMesh(d, n_components, object_type)
		mesh, object_info, L = msh.mesh()
		if d == 2:
			coords = mesh.coordinates()
			triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
			axis('equal')
			show()
		else:
			df.plot(mesh, interactive=True)

	# test_simple()
	# test_unit()
