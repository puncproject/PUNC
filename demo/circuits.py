from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *
from mesh import *
import sys

# Simulation parameters
tot_time = 20                   # Total simulation time
dt       = 0.251327              # Time step

v_thermal = 1.0
v_drift  = np.array([1.0, 0.0])

# Get the mesh
circles = CircuitDomain()      # Create the CircleDomain object
mesh    = circles.get_mesh()   # Get the mesh
Ld      = get_mesh_size(mesh)  # Get the size of the simulation domain

# Create boundary conditions and function space
periodic = [False, False]
bnd = NonPeriodicBoundary(Ld, periodic)
constr = PeriodicBoundary(Ld, periodic)

V        = df.FunctionSpace(mesh, "CG", 1, constrained_domain=constr)

bc = df.DirichletBC(V, df.Constant(0.0), bnd)

# Get the solver
poisson = PoissonSolver(V, bc)

# Create objects
objects = circles.get_objects(V)

# The inverse of capacitance matrix
inv_cap_matrix = capacitance_matrix(V, poisson, bnd, objects)

# Create the circuits
circuits = circles.get_circuits(objects, inv_cap_matrix)

# Initialize particle positions and velocities, and populate the domain
pop    = Population(mesh, periodic)
pop.init_new_specie('electron', v_drift=v_drift, v_thermal=v_thermal, num_per_cell=1)
pop.init_new_specie('proton', v_drift=v_drift, v_thermal=v_thermal, num_per_cell=1)

#------------------Plots--------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from voronoi import *
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.artist as artists

from matplotlib import cm

from matplotlib.ticker import MaxNLocator
from scipy.signal import argrelextrema
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.artist as artists

l1 = 0.
l2 = 2.*np.pi
w1 = 0.
w2 = 2.*np.pi
h1 = 0.
h2 = 2.*np.pi
x0 = np.pi
y0 = np.pi
r0 = 0.5

object_info = [x0, y0, r0]
l = [l1, w1, l2, w2]

xi, xe = [], []
for cell in pop:
    for p in cell:
        if np.sign(p.q)==1:
            xi.append(p.x)
        elif np.sign(p.q)==-1:
            xe.append(p.x)
print("num ions: ", len(xi))
print("num electrons: ", len(xe))
#
g_ratio = 1.61803398875
width = 15
height = width/g_ratio
fs = 12
fz = 12
#
# tick_spacing = 20
#
font = {'family':'serif','size':fs, 'serif': ['computer modern roman']}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':fz})
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
plt.rc('text', usetex=True)
# plt.rc('font', family='cmr')
# plt.rcParams['font.size'] = 18*1.5
plt.rcParams['axes.labelsize'] = fs+2
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['legend.fontsize'] = fz
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.color_cycle'] =  '#248dd8',  '#e22222','#3ba323', '#15efe4','#b009bf', '#1e0eb2', '#33c452', '#ef7b15', '#841654', '#2a296d', '#bcbf09'
# , '#16960d', '#2eb237'
#
tick_spacingx = 4
tick_spacingy = 0.1

#---------------------------------------------------------------------
#                  Voronoi
#---------------------------------------------------------------------

points = np.zeros((21,2))

xs = [0, 1/3., 2/3., 1]

points[:4,0] = xs
points[4:7, 1] = xs[1:]
points[7:10, 0] = xs[1:]
points[7:10, 1] = [1,1,1]
points[10:12, 1] = xs[1:3]
points[10:12, 0] = [1, 1]
points[12,:] = [0.5, 0.5]
r = 0.3
pi = np.pi
sin = np.pi
theta = [0, pi/4, pi/2., 3*pi/4, pi, 5*pi/4., 6*pi/4., 7*pi/4.]
xs = np.zeros((8,2))
xs[:,0] = 0.5+r*np.cos(theta)
xs[:,1] = 0.5+r*np.sin(theta)
points[13:, :] = xs

# compute Voronoi tesselation
vor = Voronoi(points)
tri = Delaunay(points)



fig = plt.figure(figsize=(width,height))
gs = gridspec.GridSpec(2, 4)
ax1 = plt.subplot(gs[:2,:2])
ax2 = plt.subplot(gs[:2,2:])

delaunay_plot_2d(tri, ax=ax2)
voronoi_plot_2d(vor, ax = ax2, line_colors='#ff0000', line_width = 3, line_alpha=1.0,show_points=True, show_vertices=False)
ax2.set_aspect(1)
ax2.text(-0.01, 1.1, 'b)', transform=ax2.transAxes, va='top')
# theta goes from 0 to 2pi
theta = np.linspace(0, 2*np.pi, 100)
# the radius of the circle
r = np.sqrt(0.25)
# compute x1 and x2
x1 = np.pi + r*np.cos(theta)
x2 = np.pi + r*np.sin(theta)
# ax = fig.gca()
ax1.plot(x1, x2, c='k', linewidth=3)
ax1.set_aspect(1)

x1 = np.pi + r*np.cos(theta)
x2 = 3*r + np.pi + r*np.sin(theta)
ax1.plot(x1, x2, c='k', linewidth=3)
ax1.set_aspect(1)

x1 = np.pi + r*np.cos(theta)
x2 = np.pi-3*r + r*np.sin(theta)
ax1.plot(x1, x2, c='k', linewidth=3)
ax1.set_aspect(1)

x1 = np.pi+3*r + r*np.cos(theta)
x2 = np.pi + r*np.sin(theta)
ax1.plot(x1, x2, c='k', linewidth=3)
ax1.set_aspect(1)

skip = 1
xy_electrons = np.array(xe)
xy_ions = np.array(xi)
ax1.scatter(xy_ions[::skip, 0], xy_ions[::skip, 1],
           label='ions',
           marker='o',
           c='r',
           edgecolor='none')
ax1.scatter(xy_electrons[::skip, 0], xy_electrons[::skip, 1],
           label='electrons',
           marker = 'o',
           c='b',
           edgecolor='none')
# ax.legend(loc='best')
# leg = plt.legend(loc = "center left", bbox_to_anchor = (1, 0.95), numpoints = 1)
ax1.axis([0, l2, 0, w2])
ax1.text(-0.01, 1.1, 'a)', transform=ax1.transAxes, va='top')
ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

fig.set_tight_layout(True)
# fig.savefig('particles.eps', bbox_extra_artists=(leg,),bbox_inches='tight',dpi=600)
fig.savefig('particles.eps',bbox_inches='tight',dpi=600)
fig.savefig('particles.png',bbox_inches='tight',dpi=600)

plt.show()



# sys.exit()
dv_inv = voronoi_volume_approx(V)

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kinetic_energy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))

    rho = distribute(V, pop)
    compute_object_potentials(rho, objects, inv_cap_matrix)
    rho.vector()[:] *= dv_inv

    phi     = poisson.solve(rho, objects)
    E       = electric_field(phi)
    PE[n-1] = potential_energy(pop, phi)
    KE[n-1] = accel(pop, E, (1-0.5*(n==1))*dt)

    move_periodic(pop, Ld, dt)
    pop.relocate(objects)

    redistribute_circuit_charge(circuits)

KE[0] = KE0

# df.File('phi_circuit.pvd') << phi

df.plot(rho)
df.plot(phi)
df.interactive()
sys.exit()
# ux = df.Constant((1,0))
# Ex = df.project(df.inner(E, ux), V)
df.plot(E)


import matplotlib.pyplot as plt
import matplotlib.tri as tri

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def mplot_cellfunction(cellfn):
    C = cellfn.get_local()
    tri = mesh2triang(cellfn.mesh())
    return plt.tripcolor(tri, facecolors=C)

def mplot_function(f):
    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        C = f.vector().get_local()
        return plt.tripcolor(mesh2triang(mesh), C)
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        C = f.compute_vertex_values(mesh)
        return plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    # Vector function, interpolated to vertices
    elif f.value_rank() == 1:
        w0 = f.compute_vertex_values(mesh)
        if (len(w0) != 2*mesh.num_vertices()):
            raise AttributeError('Vector field must be 2D')
        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = w0[:mesh.num_vertices()]
        V = w0[mesh.num_vertices():]
        return plt.quiver(X,Y,U,V)

# Plot a generic dolfin object (if supported)
def plot(obj):
    plt.gca().set_aspect('equal')
    if isinstance(obj, df.Function):
        return mplot_function(obj)
    elif isinstance(obj, df.CellFunctionSizet):
        return mplot_cellfunction(obj)
    elif isinstance(obj, df.CellFunctionDouble):
        return mplot_cellfunction(obj)
    elif isinstance(obj, df.CellFunctionInt):
        return mplot_cellfunction(obj)
    elif isinstance(obj, df.Mesh):
        if (obj.geometry().dim() != 2):
            raise AttributeError('Mesh must be 2D')
        return plt.triplot(mesh2triang(obj), color='#808080')

    raise AttributeError('Failed to plot %s'%type(obj))

fig = plt.figure(figsize=(width,height))
plt.subplot(1,2,1)
plot(phi)
plot(E)
plt.subplot(1,2,2)
plot(phi)
plt.show()
df.interactive()
sys.exit()

import numpy as np
import os
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import plt
import matplotlib.patches as patches

def plot_variable(u, name, direc, cmap='viridis', scale='lin', numLvls=12,
                  umin=None, umax=None, tp=False, tpAlpha=0.5, show=True,
                  hide_ax_tick_labels=False, label_axes=True,
                  title='',
                  use_colorbar=False, hide_axis=True, colorbar_loc='top'):
    """
    """
    mesh = u.function_space().mesh()
    v    = u.compute_vertex_values(mesh)
    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]
    t    = mesh.cells()

    d    = os.path.dirname(direc)
    if not os.path.exists(d):
        os.makedirs(d)

    if umin != None:
        vmin = umin
    else:
        vmin = v.min()
    if umax != None:
        vmax = umax
    else:
        vmax = v.max()

    # countour levels :
    if scale == 'log':
        v[v < vmin] = vmin + 1e-12
        v[v > vmax] = vmax - 1e-12
        from matplotlib.ticker import LogFormatter
        levels      = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
        formatter   = LogFormatter(10, labelOnlyBase=False)
        norm        = colors.LogNorm()

    elif scale == 'lin':
        v[v < vmin] = vmin + 1e-12
        v[v > vmax] = vmax - 1e-12
        from matplotlib.ticker import ScalarFormatter
        levels    = np.linspace(vmin, vmax, numLvls)
        formatter = ScalarFormatter()
        norm      = None

    elif scale == 'bool':
        from matplotlib.ticker import ScalarFormatter
        levels    = [0, 1, 2]
        formatter = ScalarFormatter()
        norm      = None

    fig = plt.figure(figsize=(width,height))
    # ax  = fig.add_subplot(111)
    ax = fig.gca(projection='3d')

    # ax.add_patch(
    #     patches.Circle(
    #         (np.pi, np.pi),
    #         1.,
    #         fill=False      # remove background
    #     )
    # )
    c = ax.plot_trisurf(x, y, v, triangles = t, cmap=cm.viridis, linewidth=0.2)
    #
    #
    # c = ax.tricontourf(x, y, t, v, levels=levels, norm=norm,
    #                  cmap=plt.get_cmap(cmap))
    plt.axis('equal')

    if tp == True:
        p = ax.triplot(x, y, t, '-', lw=0.2, alpha=tpAlpha)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    if label_axes:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'Electric potential')
    if hide_ax_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if hide_axis:
        plt.axis('off')

    # include colorbar :
    if scale != 'bool' and use_colorbar:
        divider = make_axes_locatable(plt.gca())
        cax  = divider.append_axes(colorbar_loc, "5%", pad="3%")
        cbar = fig.colorbar(c)#, cax=cax, format=formatter)#,ticks=levels)
        tit = plt.title(title)

    if use_colorbar:
        plt.tight_layout(rect=[.03,.03,0.97,0.97])
    else:
        cbar = fig.colorbar(c)
        cbar.set_label(r'$\varPhi [V]$')
        plt.tight_layout()
    fig.set_tight_layout(True)
    plt.savefig(direc + name + '.eps',bbox_inches='tight', dpi=600)
    plt.savefig(direc + name + '.png',bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    plt.close(fig)

plot_variable(phi, 'phi2', '/home/diako/Documents/Work/punc')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scipy.interpolate import griddata



# mesh = phi.function_space().mesh()
# v    = phi.compute_vertex_values(mesh)
# x    = mesh.coordinates()[:,0]
# y    = mesh.coordinates()[:,1]
#
# # Make data.
# X, Y = np.meshgrid(x, y)
# Z = v
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

uvals = phi.vector().get_local()
xyvals = mesh.coordinates()
xvals = xyvals[:,0]
yvals=xyvals[:,1]

xx = np.linspace(0,np.pi*2)
yy = np.linspace(0,2*np.pi)

XX, YY = np.meshgrid(xx,yy)

uu = griddata(xvals, yvals, uvals,xx, yy, interp='linear')




fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, uu, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
