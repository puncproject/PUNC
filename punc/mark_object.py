from __future__ import print_function
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

def object_cells(mesh, facet_f, n_components):
    D = mesh.topology().dim()
    mesh.init(D-1,D) # Build connectivity between facets and cells
    components_cells = []
    for i in range(n_components):
        itr_facet = SubsetIterator(facet_f, i+1)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        components_cells.append(object_adjacent_cells)

    return components_cells

def object_verticesV2(mesh, V, facet_f, n_components):
    D = mesh.topology().dim()
    mesh.init(D-1,D) # Build connectivity between facets and cells
    # cell_number = 0
    # tdim = mesh.topology().dim()
    dofmap = V.dofmap()
    # facet_dofs = [None]*len(dofmap.tabulate_facet_dofs(0))

    # cell = Cell(mesh,cell_number)
    itr_facet = SubsetIterator(facet_f, 1)
    facet_dofs = set()
    for f in itr_facet:
        local_facet_dofs = dofmap.tabulate_facet_dofs(f.index())
        print("local_facet_dofs: ", local_facet_dofs)
        cell_number = f.entities(D)[0]
        print("cell dofs: ", dofmap.cell_dofs(cell_number))
        facet_dofs.update(dofmap.cell_dofs(cell_number)[local_facet_dofs])
    facet_dofs = list(facet_dofs)

    return facet_dofs

def object_verticesV3(mesh, V, facet_f, n_components):
    dm = V.dofmap()
    facet_map = np.zeros((3,2), dtype=np.uintc)
    for ind in range(3):
       dm.tabulate_facet_dofs(ind)
       print("dof_map: ", dm.tabulate_facet_dofs(ind))
    itr_facet = SubsetIterator(facet_f, 1)
    all_facet_dofs = set()
    for facet in itr_facet:
       cell_ind = facet.entities(2)[0]
       local_cell = Cell(mesh, cell_ind)
       local_facet_ind = (local_cell.entities(1)\
                          ==facet.index()).nonzero()[0][0]
       all_facet_dofs.update(dm.cell_dofs(cell_ind)\
                             [facet_map[local_facet_ind]])

    print(all_facet_dofs)

def object_vertices(facet_f, n_components):
    components_vertices = []
    for i in range(n_components):
        itr_facet = SubsetIterator(facet_f, i+1)
        object_vertices = set()
        for f in itr_facet:
            for v in vertices(f):
                object_vertices.add(v.index())

        object_vertices = list(object_vertices)
        components_vertices.append(object_vertices)

    return components_vertices

def mark_boundary_adjecent_cells(mesh):
    d = mesh.topology().dim()
    if d == 2:
        boundary_adjacent_cells = [myCell for myCell in cells(mesh)
                                  if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                    (myFacet.midpoint().y()-np.pi)**2 < 0.25) \
                                    for myFacet in facets(myCell)])]
    elif d == 3:
        boundary_adjacent_cells = [myCell for myCell in cells(mesh)
                                  if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                    (myFacet.midpoint().y()-np.pi)**2 + \
                                    (myFacet.midpoint().z()-np.pi)**2 < 0.25) \
                                    for myFacet in facets(myCell)])]

    cell_domains = CellFunction('size_t', mesh)
    cell_domains.set_all(1)
    for myCell in boundary_adjacent_cells:
        cell_domains[myCell] = 0

    return cell_domains

def mark_outer_boundary(facet_f, domain_info, n_components):
    d = len(domain_info)/2

    for i in range(2*d):
        boundary = 'near((x[i]-l), 0, tol)'
        boundary = CompiledSubDomain(boundary,i=i%d,l=domain_info[i],tol=1E-8)
        boundary.mark(facet_f, (n_components+i+1))
    return facet_f

def mark_circular_objects(facet_f, object_info, n_components):
    if n_components > 1:
      tmp = 3
      for i in range(2, n_components+1):
          x1 = object_info[tmp]
          y1 = object_info[tmp+1]
          r1 = object_info[tmp+2]
          boundary_circle =\
                   'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1), r1*r1, tol)'
          boundary_circle =CompiledSubDomain(boundary_circle,
                                             x1 = x1, y1 = y1,
                                             r1 = r1, tol=1E-2)
          boundary_circle.mark(facet_f, i)
          tmp += 3
    return facet_f

def mark_spherical_objects(facet_f, object_info, n_components):
    if n_components > 1:
      tmp = 4
      for i in range(2, n_components+1):
          x1 = object_info[tmp]
          y1 = object_info[tmp+1]
          z1 = object_info[tmp+2]
          r1 = object_info[tmp+3]
          boundary_sphere =\
         'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1), r1*r1, tol)'
          boundary_sphere = CompiledSubDomain(boundary_sphere,
                                              x1 = x1, y1 = y1,
                                              z1 = z1, r1 = r1, tol=1E-2)
          boundary_sphere.mark(facet_f, i)
          tmp += 4
    return facet_f

def mark_cylindrical_objects(facet_f, object_info, n_components, h2):
    if n_components > 1:
      tmp = 4
      for i in range(2, n_components+1):
          x1 = object_info[tmp]
          y1 = object_info[tmp+1]
          r1 = object_info[tmp+2]
          h1 = object_info[tmp+3]
          z0 = (h2-h1)/2.      # Bottom point of cylinder
          z1 = (h2+h1)/2.      # Top point of cylinder
          boundary_cylinder = 'near(x[2],z0,tol)' and 'near(x[2],z1,tol)' and \
                   'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1), r1*r1, tol)'
          boundary_cylinder =\
               CompiledSubDomain(boundary_cylinder, x1 = x1, y1 = y1, r1 = r1,
                                                   z0 = z0, z1= z1, tol=1E-2)
          boundary_cylinder.mark(facet_f, i)
          tmp += 4

    return facet_f

def mark_boundaries(mesh, domain_info, object_type, object_info, n_components):
    facet_f = FacetFunction('size_t', mesh, 0)
    DomainBoundary().mark(facet_f, 1)
    if n_components > 1:
        if object_type == 'multi_circles':
            facet_f = mark_circular_objects(facet_f, object_info, n_components)
        elif object_type == 'multi_spheres':
            facet_f = mark_spherical_objects(facet_f, object_info, n_components)
        elif object_type == 'multi_cylinders':
            h2 = mesh.coordinates()[:,2].max()
            facet_f = mark_cylindrical_objects(facet_f, object_info,
                                               n_components, h2)
    facet_f = mark_outer_boundary(facet_f, domain_info, n_components)
    return facet_f

if __name__ == '__main__':

    n_components = 4
    object_type = 'multi_circles'
    mesh = Mesh("circuit.xml")
    d = mesh.topology().dim()
    L = np.empty(2*d)
    for i in range(d):
        l_min = mesh.coordinates()[:,i].min()
        l_max = mesh.coordinates()[:,i].max()
        L[i] = l_min
        L[d+i] = l_max
    print("L: ", L)
    r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
    x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
    y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
    z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
    if d == 2:
        if n_components == 2:
            object_info = [x0, y0, r0, x1, y1, r1]
        else:
            object_info = [x0, y0, r0, x1, y1, r1,
                           x2, y2, r2, x3, y3, r3]
    elif d == 3:
        object_info = [x0, y0, z0, r0, x1, y1, z1, r1]

    facet_f = mark_boundaries(mesh, L, object_type, object_info, n_components)
    plot(facet_f, interactive=True)
