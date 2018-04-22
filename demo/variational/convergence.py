import matplotlib.pyplot as plt
import numpy as np

data = dict()
with open("convergence.txt") as f:
    for l in f:
        res, order, Qm, e_abs, e_rel1, e_rel2, hmax, hmin = l.split()

        order = int(order)
        res = int(res)

        if not order in data:
            data[order] = dict()

        if not res in data[order]:
            data[order][res] = dict()

        data[order][res]['Qm']     = float(Qm)
        data[order][res]['e_abs']  = float(e_abs)
        data[order][res]['e_rel1'] = float(e_rel1)
        data[order][res]['e_rel2'] = float(e_rel2)
        data[order][res]['hmax']   = float(hmax)
        data[order][res]['hmin']   = float(hmin)

for order in data.keys():

    hmins = [values['hmin'] for res, values in data[order].items()]
    error = [values['e_rel1'] for res, values in data[order].items()]
    p = plt.loglog(hmins, error, 'o-', label='CG{}'.format(order))
    color = p[-1].get_color()
    y = error[0]*(np.array(hmins)/hmins[0])**order
    plt.loglog(hmins, y, ':', color=color)

plt.legend(loc='lower right')
plt.grid()
plt.title('Convergence of potential along x-axis')
plt.xlabel('hmin')
plt.ylabel('$||u-u_h||/||u||$')
plt.show()

for order in data.keys():

    hmins = [values['hmin'] for res, values in data[order].items()]
    Qm = [values['Qm'] for res, values in data[order].items()]
    error = np.abs(np.array(Qm)-5)/5
    p = plt.loglog(hmins, error, 'o-', label='CG{}'.format(order))
    color = p[-1].get_color()
    y = error[0]*(np.array(hmins)/hmins[0])**order
    plt.loglog(hmins, y, ':', color=color)

plt.legend(loc='lower right')
plt.grid()
plt.title('Convergence of object charge')
plt.xlabel('hmin')
plt.ylabel('$|Q-Qe|/|Q|$')
plt.show()
