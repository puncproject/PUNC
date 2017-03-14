from punc import *
from mesh import *

class TestRandomPoints:
    def test_length(self):
        N = 100
        points = randomPoints(lambda x: x[0],[1,1],N)
        assert len(points)==N


def test_initial_conditions():
    import numpy as np

    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    pdf = lambda x: 1
    Ld = [2*np.pi, 2*np.pi]
    N = 2000

    n_components = 4
    dim = 2
    msh = ObjectMesh(dim, n_components, 'spherical_object')
    mesh, object_info, L = msh.mesh()

    dim = mesh.geometry().dim()
    Ld = np.asarray(L[dim:])
    #-------------------------------------------------------------------------------
    #            Create boundary conditions and function space
    #-------------------------------------------------------------------------------
    PBC = PeriodicBoundary(Ld)
    V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    v2d = df.vertex_to_dof_map(V)
    #-------------------------------------------------------------------------------
    #        Create facet and cell functions to to mark the boundaries
    #-------------------------------------------------------------------------------
    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+len(L))

    cell_f = df.CellFunction('size_t', mesh)
    cell_f.set_all(n_components)

    # r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
    # x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
    # y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
    # z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
    # object_info = [x0, y0, r0, x1, y1, r1, x2, y2, r2, x3, y3, r3]
    #
    # # the objects:
    d = len(Ld)
    s, r = [], []
    for i in range(n_components):
        j = i*(d+1)
        s.append(object_info[j:j+d])
        r.append(object_info[j+d])
    #
    # objects = []
    # for i in range(n_components):
    #     s0 = np.asarray(s[i])
    #     r0 = r[i]
    #     fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
    #     objects.append(Object(fun,i))
    tol = 1e-8
    objects = []
    for i in range(n_components):
        j = i*(dim+1)
        s0 = object_info[j:j+dim]
        r0 = object_info[j+dim]
        func = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2+tol
        objects.append(Object(func, i, mesh, facet_f, cell_f, v2d))

    points = random_points(pdf, Ld, N, pdfMax=1, objects=objects)

    fig = plt.figure()
    theta = np.linspace(0, 2*np.pi, 100)
    # the radius of the circle
    for i in range(n_components):
        # compute x1 and x2
        x1 = s[i][0] + r[i]*np.cos(theta)
        x2 = s[i][1] + r[i]*np.sin(theta)
        ax = fig.gca()
        ax.plot(x1, x2, c='k', linewidth=3)
        ax.set_aspect(1)

    ax.scatter(points[:, 0], points[:, 1],
               label='ions',
               marker='o',
               c='r',
               edgecolor='none')
    ax.legend(loc='best')
    ax.axis([0, Ld[0], 0, Ld[1]])
    plt.show()

test_initial_conditions()
