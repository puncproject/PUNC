from punc import *

class TestRandomPoints:
    def test_length(self):
        N = 100
        points = randomPoints(lambda x: x[0],[1,1],N)
        assert len(points)==N


def test_object():
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    pdf = lambda x: 1
    Ld = [2*np.pi, 2*np.pi]
    N = 2000

    n_components = 4
    r0 = 0.5; r1 = 0.5; r2 = 0.5; r3 = 0.5;
    x0 = np.pi; x1 = np.pi; x2 = np.pi; x3 = np.pi + 3*r3;
    y0 = np.pi; y1 = np.pi + 3*r1; y2 = np.pi - 3*r1; y3 = np.pi;
    z0 = np.pi; z1 = np.pi; z2 = np.pi; z3 = np.pi;
    object_info = [x0, y0, r0, x1, y1, r1, x2, y2, r2, x3, y3, r3]

    # the objects:
    d = len(Ld)
    s, r = [], []
    for i in range(n_components):
        j = i*(d+1)
        s.append(object_info[j:j+d])
        r.append(object_info[j+d])

    objects = []
    for i in range(n_components):
        s0 = np.asarray(s[i])
        r0 = r[i]
        fun = lambda x, s0 = s0, r0 = r0: np.dot(x-s0, x-s0) <= r0**2
        objects.append(Object(fun,i))

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

test_object()
