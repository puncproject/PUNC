from __future__ import print_function
import dolfin as df
import numpy as np
import sys

def object_cells(mesh, facet_f, n_components):
    """This function returns the cells adjecent to the objects

    Args:
        mesh           : The mesh
        facet_f     : contains the facets of each surface component
        n_components: number of surface components

    returns:
        A list of all cells adjecent to each electrical component
    """
    D = mesh.topology().dim()
    mesh.init(D-1,D) # Build connectivity between facets and cells
    components_cells = []
    for i in range(n_components):
        itr_facet = df.SubsetIterator(facet_f, i)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        components_cells.append(object_adjacent_cells)
    return components_cells

def object_vertices(facet_f, n_components):
    """This function returns the vertices of all objects

    Args:
        facet_f     : contains the facets of each surface component
        n_components: number of surface components

    returns:
        A list of all object vertices for each component
    """
    components_vertices = []
    for i in range(n_components):
        itr_facet = df.SubsetIterator(facet_f, i)
        object_vertices = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_vertices.add(v.index())

        object_vertices = list(object_vertices)
        components_vertices.append(object_vertices)

    return components_vertices

def objects_dofs(V, facet_f, n_components):
    """This function returns the dofs of the objects

    Args:
        V           : FunctionSpace(mesh, "CG", 1)
        facet_f     : contains the facets of each surface component
        n_components: number of surface components

    returns:
        A list of all object dofs for each component
    """
    components_dofs = []
    v2d = df.vertex_to_dof_map(V)
    for i in range(n_components):
        itr_facet = df.SubsetIterator(facet_f, i)
        object_dofs = set()
        for f in itr_facet:
            for v in df.vertices(f):
                object_dofs.add(v2d[v.index()])
        object_dofs = list(object_dofs)
        components_dofs.append(object_dofs)
    return components_dofs

def mark_boundary_adjecent_cells(mesh):
    """This function marks the cells adjecent to the objects

    Args:
        mesh: The mesh

    returns:
        Marks the cells adjecent to objects

    Note:
    ----   Not Finished! Works only for a single circle or sphere
    """
    d = mesh.topology().dim()
    if d == 2:
        boundary_adjacent_cells = [myCell for myCell in df.cells(mesh)
                                  if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                    (myFacet.midpoint().y()-np.pi)**2 < 0.25) \
                                    for myFacet in df.facets(myCell)])]
    elif d == 3:
        boundary_adjacent_cells = [myCell for myCell in df.cells(mesh)
                                  if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                    (myFacet.midpoint().y()-np.pi)**2 + \
                                    (myFacet.midpoint().z()-np.pi)**2 < 0.25) \
                                    for myFacet in df.facets(myCell)])]

    cell_domains = df.CellFunction('size_t', mesh)
    cell_domains.set_all(1)
    for myCell in boundary_adjacent_cells:
        cell_domains[myCell] = 0

    return cell_domains

def mark_outer_boundary(facet_f, domain_info, n_components):
    """This function marks the outer boundaries of the simulation domain
    Args:
        facet_f     : contains the facets of each surface component
        domain_info : contains the length of the domain in each direction
        n_components: number of surface components

    returns:
        facet_f: marked facets of the outer boundaries of the domain
    """
    d = len(domain_info)/2

    for i in range(2*d):
        boundary = 'near((x[i]-l), 0, tol)'
        boundary = df.CompiledSubDomain(boundary,i=i%d,l=domain_info[i],tol=1E-8)
        boundary.mark(facet_f, (n_components+i))
    return facet_f

def mark_circular_objects(facet_f, object_info, n_components):
    """This function marks all the circular objects inside the simulation domain
    Args:
        facet_f     : contains the facets of each surface component
        object_info : contains the geometry of each circular object
        n_components: number of surface components

    returns:
        facet_f: marked facets of the circular objects
    """
    assert n_components > 1
    tmp = 3
    for i in range(1, n_components):
        x1 = object_info[tmp]
        y1 = object_info[tmp+1]
        r1 = object_info[tmp+2]
        boundary_circle =\
               'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1), r1*r1, tol)'
        boundary_circle = df.CompiledSubDomain(boundary_circle,
                                         x1 = x1, y1 = y1,
                                         r1 = r1, tol=1E-2)
        boundary_circle.mark(facet_f, i)
        tmp += 3
    return facet_f

def mark_spherical_objects(facet_f, object_info, n_components):
    """This function marks all the spherical objects inside the simulation
    domain.

    Args:
        facet_f     : contains the facets of each surface component
        object_info : contains the geometry of each spherical object
        n_components: number of surface components

    returns:
        facet_f: marked facets of the spherical objects
    """
    assert n_components > 1
    tmp = 4
    for i in range(1, n_components):
        x1 = object_info[tmp]
        y1 = object_info[tmp+1]
        z1 = object_info[tmp+2]
        r1 = object_info[tmp+3]
        boundary_sphere =\
        'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1), r1*r1, tol)'
        boundary_sphere = df.CompiledSubDomain(boundary_sphere,
                                          x1 = x1, y1 = y1,
                                          z1 = z1, r1 = r1, tol=1E-2)
        boundary_sphere.mark(facet_f, i)
        tmp += 4
    return facet_f

def mark_cylindrical_objects(facet_f, object_info, n_components, h2):
    """This function marks all the cylindrical objects inside the simulation
    domain.

    Args:
        facet_f     : contains the facets of each surface component
        object_info : contains the geometry of each cylindrical object
        n_components: number of surface components

    returns:
        facet_f: marked facets of the cylindrical objects
    """
    assert n_components > 1
    tmp = 4
    for i in range(1, n_components):
          x1 = object_info[tmp]
          y1 = object_info[tmp+1]
          r1 = object_info[tmp+2]
          h1 = object_info[tmp+3]
          z0 = (h2-h1)/2.      # Bottom point of cylinder
          z1 = (h2+h1)/2.      # Top point of cylinder
          boundary_cylinder = 'near(x[2],z0,tol)' and 'near(x[2],z1,tol)' and \
                   'near((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1), r1*r1, tol)'
          boundary_cylinder =\
               df.CompiledSubDomain(boundary_cylinder, x1 = x1, y1 = y1, r1 = r1,
                                                   z0 = z0, z1= z1, tol=1E-2)
          boundary_cylinder.mark(facet_f, i)
          tmp += 4

    return facet_f

def mark_boundaries(mesh, domain_info, object_type, object_info, n_components):
    """This function marks the exterior boundaries, as well as all the objects
    inside the simulation domain.

    Args:
        mesh        : mesh of the domain
        domain_info : length of the domain in each direction
        object_type : geometrical type of object
        object_info : contains the geometry of each object
        n_components: number of surface components

    returns:
        facet_f: marked facets of all exterior and object boundaries.
    """
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+len(domain_info))
    df.DomainBoundary().mark(facet_f, 0)
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

    import matplotlib.pyplot as plt
    n_components = 4
    object_type = 'multi_circles'
    mesh = df.Mesh("circuit.xml")
    d = mesh.topology().dim()
    L = np.empty(2*d)
    for i in range(d):
        l_min = mesh.coordinates()[:,i].min()
        l_max = mesh.coordinates()[:,i].max()
        L[i] = l_min
        L[d+i] = l_max
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
    df.plot(facet_f, interactive=True)
