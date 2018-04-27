# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of PUNC.
#
# PUNC is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PUNC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC.  If not, see <http://www.gnu.org/licenses/>.

# Imports important python 3 behaviour to ensure correct operation and
# performance in python 2
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import subprocess as sp
import os
import tempfile
import warnings

# This is not yet parallelized
#comm = pyMPI.COMM_WORLD
#...comm.Get_rank()

def get_voronoi_points(V, Ld, periodic, tol=1e-13):
    """
    Returns the center points (in 3D) of the Voronoi diagram being the dual of
    the mesh of function space V. It also returns the corresponding FEniCS DOF
    index of each point:

        points, indices = get_voronoi_points(V, Ld, periodic)

    Ld is the size of the mesh and periodic is a list of three bools signifying
    whether each dimension is periodic or not.

    Points within tol of the boundary are shifted an amount tol inwards.
    """

    # All center points are vertices in the Delaunay mesh. Let them be 3D.
    mesh = V.mesh()
    N = mesh.num_vertices()
    dim = mesh.geometry().dim()
    points = np.zeros((N,3))
    points[:,:dim] = mesh.coordinates() # replace with df.vertices(mesh)?
    dof_indices = df.vertex_to_dof_map(V)

    # Remove vertices on upper periodic boundaries
    i = 0
    while i<len(points):
        if any([df.near(a,b,tol) and p for a,b,p in zip(points[i],list(Ld),periodic)]):
            points = np.delete(points,[i],axis=0)
            dof_indices = np.delete(dof_indices,[i],axis=0)
        else:
            i = i+1

    # Shift points on boundary inwards
    for d in range(3):
        points[:,d] = [x+tol if df.near(x,0    ,tol) else x for x in points[:,d]]
        points[:,d] = [x-tol if df.near(x,Ld[d],tol) else x for x in points[:,d]]

    return points, dof_indices

def exec_voropp(points, indices, Ld, periodic):
    """
    Executes Voro++ on a list of points in a domain of size Ld to get the
    volumes of the Voronoi cells about those points:

        volumes = exec_voropp(points, indices, Ld, periodic)

    indices gives the position each point is to have in the returned value, e.g.
    the volume of the Voronoi cell centered at points[i] is volumes[indices[i]].
    If indices is the FEniCS DOF indices volumes will be arranged like FEniCS
    vectors.

    periodic is a list of three bools signifying whether each direction is
    periodic or not. The points must be three-dimensional.
    """

    with tempfile.NamedTemporaryFile(   mode='w',
                                        prefix='punc_',
                                        delete=False) as f:
        fname = f.name

        # Write points enumerated by indices with full hex. accuracy.
        for i,v in zip(indices,points):
            f.write("%d %s %s %s\n"%(i,v[0].hex(),v[1].hex(),v[2].hex()))

        f.close() # Flush

        # Call Voro++
        cmd = os.path.join(os.path.dirname(os.path.realpath(__file__)),"voro++")
        if periodic[0]: cmd += " -px"
        if periodic[1]: cmd += " -py"
        if periodic[2]: cmd += " -pz"
        cmd += " 0 %s 0 %s 0 %s "%(Ld[0].hex(),Ld[1].hex(),Ld[2].hex())
        cmd += fname
        sp.call(cmd,shell=True)

        volumes = np.zeros(len(points))
        with open(fname+'.vol','r') as f:
            for i,line in enumerate(f):
                words = line.split()
                index = int(words[0])
                volume = float(words[4])
                volumes[index] = volume

        try:
            os.remove(fname+'.vol')
        except:
            print("Failed to delete temporary file: %s.vol"%fname)

    try:
        os.remove(fname)
    except:
        print("Failed to delete temporary file: %s"%fname)

    return volumes

def voronoi_volume(V, Ld, periodic=True, inv=True,
                   raw=True, tol=1e-13, vol_tol=1e-4):
    """
    Returns the volume of the Voronoi cells centered at the DOFs as a FEniCS
    function. V is the function space for the function to be returned (must be
    CG1), Ld is the size of the domain and periodic is a list of bools
    signifying whether each direction is periodic or not. Alternatively, a
    single bool can be used to apply on all directions. Works for 1D, 2D and 3D.

    Vertices in the mesh lying outside the domain or exactly on the upper
    boundary are deleted by the underlying Voro++ program. Therefore, points
    which are within tol of a boundary are shifted tol inwards. This number
    must be larger than the finite precision but small enough to not compromise
    accuracy. The default value is usually sufficient.

    The function checks that the sum of the volume of all Voronoi cells are
    within vol_tol of the total volume of the domain.
    """

    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1

    n_dims = len(Ld)

    # Expand to list
    if isinstance(periodic,bool):
        periodic = [periodic]*n_dims

    # Expand to 3D. Voro++ only works on 3D. 2D (1D) is acheived by letting
    # the points have z=0 (and y=0) and the domain have thickness 1 in those
    # directions.
    Ld       = np.concatenate([ Ld,       np.ones(3-n_dims) ])
    periodic = np.concatenate([ periodic, np.ones(3-n_dims,bool)])

    points, indices = get_voronoi_points(V, Ld, periodic, tol)

    assert len(points) == V.dim(), \
        "Got too many Voronoi center points. Check periodic argument."

    volumes = exec_voropp(points, indices, Ld, periodic)

    assert all(volumes!=0), \
        "Some Voronoi cells are missing (have zero volume). Try increasing tol."

    total_volume = df.assemble(1*df.dx(V.mesh()))
    assert np.abs(sum(volumes)-total_volume)<vol_tol, \
        "The volume of the Voronoi cells (%E) "%sum(volumes) +\
        "doesn't add up to total domain volume (%E). "%total_volume +\
        "A possible cause is that objects aren't implemented yet or " +\
        "that objects are concave. A possible remedy could be to use" +\
        "voronoi_volume_approx()."

    if inv:
        volumes = volumes**(-1)

    if raw:
        return volumes
    else:
        dv = df.Function(V)
        dv.vector()[:] = volumes
        return dv

def voronoi_volume_approx(V, inv=True, raw=True):
    """
    Returns the approximated volume for every Voronoi cell centered at
    the a DOF as a FEniCS function. V is the function space for the function
    to be returned (must be CG1). Works for
    1D, 2D and 3D, with and without periodic boundaries and objects.

    The approximated volume of a Voronoi cell centered at a vertex is the
    sum of the neighboring cells divided by the number of geometric dimensions
    plus one. This approximation is better the closer to equilateral the cells
    are, a feature which is desirable in a FEM mesh anyhow.

    Curiously, the result of this function happens to be exact not only for
    completely equilateral cells, but also for entirely periodic simple
    meshes as created using simple_mesh(). For non-periodic simple meshes it
    becomes inaccurate on the boundary nodes. The total volume of all cells is
    always correct.
    """
    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1

    n_dofs = V.dim()
    dof_indices = df.vertex_to_dof_map(V)
    volumes = np.zeros(n_dofs)

    # These loops inherently deal with periodic boundaries
    for i,v in enumerate(df.vertices(V.mesh())):
        for c in df.cells(v):
            volumes[dof_indices[i]] += c.volume()

    volumes /= (V.mesh().geometry().dim()+1)

    if inv:
        volumes = volumes**(-1)

    if raw:
        return volumes
    else:
        dv = df.Function(V)
        dv.vector()[:] = volumes
        return dv

def voronoi_length(V, Ld, periodic=True, inv=True, raw=True):
    """
    Returns the length of 1D Voronoi cells centered at the DOFs as a FEniCS
    function. V is the function space for the function to be returned (must be
    CG1), Ld is the size of the domain and periodic indicates it's length.

    See also voronoi_volume() which implements 1D, 2D and 3D using the Voro++
    library. This is an independent 1D implementation for comparison.
    """

    vertices = V.mesh().coordinates()[:,0]

    # Sort vertices in incresing order
    srt = np.argsort(vertices)
    vertices = vertices[srt]

    # Dual grid has nodes on average positions
    dual = 0.5*(vertices[1:]+vertices[:-1])

    # For finite non-periodic grid we must add edges
    if not periodic:
        dual = np.concatenate([[0],dual,Ld])

    # Compute volume of Voronoi cells
    volume = dual[1:]-dual[:-1]

    # Add volume of "wrapping" cell for periodic boundaries
    if periodic:
        first = Ld[0]-dual[-1]+dual[0]
        volume = np.concatenate([[first],volume,[first]])

    # volume is now the Voronoi volume for the vertices in mesh
    # sort volume back to same ordering as mesh.coordinates()
    srt_back = np.argsort(srt)
    volume = volume[srt_back]

    if inv:
        volume = volume**(-1)

    # Store as Function using correct dof-ordering.
    v2d = df.vertex_to_dof_map(V)
    dv = df.Function(V)
    dv.vector()[v2d] = volume

    if raw:
        return dv.vector().get_local()
    else:
        return dv

def distribute(V, pop, dv_inv):

    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1

    element = V.dolfin_element()
    s_dim = element.space_dimension() # Number of nodes per element
    basis_matrix = np.zeros((s_dim,1))


    rho = df.Function(V)

    for cell in df.cells(V.mesh()):
        cellindex = cell.index()
        dofindex = V.dofmap().cell_dofs(cellindex)

        accum = np.zeros(s_dim)
        for particle in pop[cellindex]:

            element.evaluate_basis_all( basis_matrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            accum += particle.q * basis_matrix.T[0]


        rho.vector()[dofindex] += accum

    # rho.vector()[:] *= dv_inv
    rho_arr = rho.vector().get_local()
    rho_arr *= dv_inv
    rho.vector().set_local(rho_arr)

    return rho

def distribute_elementwise(V, pop):

    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1

    element = V.dolfin_element()
    s_dim = element.space_dimension() # Number of nodes per element
    basis_matrix = np.zeros((s_dim,1))

    rho = df.Function(V)

    for cell in df.cells(V.mesh()):
        cellindex = cell.index()
        dofindex = V.dofmap().cell_dofs(cellindex)

        accum = np.zeros(s_dim)
        for particle in pop[cellindex]:

            element.evaluate_basis_all( basis_matrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            accum += particle.q * basis_matrix.T[0]

        accum /= cell.volume()
        rho.vector()[dofindex] += accum

    return rho
