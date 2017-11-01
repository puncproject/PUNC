from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pylab as plt

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

domain = Rectangle(Point(-5,-5),Point(5,5))-Circle(Point(0,0), 1)
mesh = generate_mesh(domain, 8)

#==============================================================================
# INTEGRATE ON SURFACE
#------------------------------------------------------------------------------

def interiorSurface(x, onBnd):
	return all(np.abs(x)<1.5) and onBnd

InteriorSurface = AutoSubDomain(interiorSurface)

mfi = FacetFunction("size_t",mesh)
mfi.set_all(0)
InteriorSurface.mark(mfi,1)

#plot(mfi,interactive=True)

#dsi = Measure('ds',domain=mesh)[mfi]
#dsi = ds(subdomain_data=surface)
#dsi = Measure('ds')[mfi]
dsi = ds(subdomain_data=mfi)

V = FunctionSpace(mesh,'CG',1)

I = Constant(1.0)

t = assemble(project(I,V)*dsi(1))

print(t)

#==============================================================================
# GENERATE SUBMESH
#------------------------------------------------------------------------------

class Omega(SubDomain):
	def inside(self, x, onBnd):
		return x[0] <= 0.5
omegaL=Omega()

subd = CellFunction("size_t",mesh)
subd.set_all(0)
omegaL.mark(subd,1)

submesh = SubMesh(mesh, mfi, 1)

#plot(submesh,interactive=True)

bmesh = BoundaryMesh(mesh, "exterior")

#x = bmesh.coordinates()[:,0]
#y = bmesh.coordinates()[:,1]
#plt.plot(x,y,'.')
#plt.show()

cc = CellFunction('size_t',bmesh,0)
cc.set_all(0)

InteriorSurface = AutoSubDomain(lambda x: np.linalg.norm(x)<2)
InteriorSurface.mark(cc,1)

submesh = SubMesh(bmesh,cc,1)

x = submesh.coordinates()[:,0]
y = submesh.coordinates()[:,1]
plt.plot(x,y,'*')
plt.show()
