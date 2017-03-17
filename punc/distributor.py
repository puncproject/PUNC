# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

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
    points[:,:dim] = mesh.coordinates()
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
                dof_index = int(words[0])
                volume = float(words[4])
                volumes[dof_index] = volume

        try:
            os.remove(fname+'.vol')
        except:
            print("Failed to delete temporary file: %s.vol"%fname)

    try:
        os.remove(fname)
    except:
        print("Failed to delete temporary file: %s"%fname)

    return volumes

def voronoi_volume(V, Ld, periodic=True, tol=1e-13, inv=True, raw=True):
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
    volumes = exec_voropp(points, indices, Ld, periodic)

    assert all(volumes!=0), \
        "Some Voronoi cells are missing (have zero volume). Try increasing tol."

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
        return dv.vector().array()
    else:
        return dv

def distribute(V, pop):

    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1

    element = V.dolfin_element()
    s_dim = element.space_dimension() # Number of nodes per element
    basisMatrix = np.zeros((s_dim,1))

    rho = df.Function(V)

    for cell in df.cells(V.mesh()):
        cellindex = cell.index()
        dofindex = V.dofmap().cell_dofs(cellindex)

        accum = np.zeros(s_dim)
        for particle in pop[cellindex]:

            element.evaluate_basis_all( basisMatrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            accum += particle.q * basisMatrix.T[0]

        rho.vector()[dofindex] += accum

    return rho
