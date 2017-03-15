from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

import dolfin as df
import numpy as np

class UnitMesh:

	def __init__(self, N):
		self.N = N
		self.d = len(N)
		self.mesh_types = [df.UnitIntervalMesh,
						   df.UnitSquareMesh,
						   df.UnitCubeMesh]

	def mesh(self):
		msh = self.mesh_types[self.d-1](*self.N)
		return msh

class SimpleMesh:

	def __init__(self, Ld, N):
		self.N = N
		self.d = len(N)
		L0 = [0.0]*self.d
		self.L = L0 + Ld
		self.mesh_types = [df.RectangleMesh, df.BoxMesh]

	def mesh(self):
		msh = self.mesh_types[self.d-2](df.Point(*self.L[:self.d]),
		                                df.Point(*self.L[self.d:]),
	                                    *self.N)
		return msh

def get_mesh_circle():

	class Circle(Object):
		def inside(self, x, on_bnd):
			return ...

	objects[0] = Circle()

	return mesh, objects

def get_mesh_circuit():
	return mesh, objects

class ObjectMesh:

	def __init__(self, d, n_components, object_type):
		self.d = d
		self.n_components = n_components
		self.object_type = object_type
		L = np.empty(2*self.d)
		self.L = L

	def mesh(self):
		if (self.d == 2 and self.object_type == 'spherical_object'):
			if self.n_components == 1:
				msh = df.Mesh("mesh/circle.xml")
				object_info = [np.pi, np.pi, 0.5]
			elif self.n_components == 2:
				msh = df.Mesh("mesh/capacitance2.xml")
				r0 = 0.5; r1 = 0.5;
				x0 = np.pi; x1 = np.pi;
				y0 = np.pi; y1 = np.pi + 3*r1;
				z0 = np.pi; z1 = np.pi;
				object_info = [x0, y0, r0, x1, y1, r1]
			elif self.n_components == 4:
				msh = df.Mesh("mesh/circuit.xml")
				r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
				x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
				y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
				z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
				object_info = [x0, y0, r0, x1, y1, r1, x2, y2, r2, x3, y3, r3]
		if (self.d == 3 and self.object_type == 'spherical_object'):
			if self.n_components == 1:
				msh = df.Mesh("mesh/sphere.xml")
				object_info = [np.pi, np.pi, np.pi, 0.5]
		if self.object_type == 'cylindrical_object':
			if self.n_components == 1:
				msh = df.Mesh('mesh/cylinder_object.xml')
				object_info = [np.pi, np.pi, 0.5, 1.0]

		for i in range(self.d):
		    self.L[i] = msh.coordinates()[:,i].min()
		    self.L[self.d+i] = msh.coordinates()[:,i].max()

		return msh, object_info, self.L

if __name__=='__main__':

	def test_simple():
		from pylab import axis, show, triplot
		Ld = [1., 2.]
		N = [20, 10]
		msh = SimpleMesh(Ld, N)
		mesh = msh.mesh()
		coords = mesh.coordinates()
		triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
		axis('equal')
		show()

	def test_unit():
		from pylab import axis, show, triplot
		N = [20, 10]
		msh = UnitMesh(N)
		mesh = msh.mesh()
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

	test_objects()
