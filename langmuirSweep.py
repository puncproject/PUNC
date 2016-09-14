import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from itertools import izip as zip, count


print("Initializing solver")
langm = Langmuir()

langm.Nt = 2

N = 9

dts = np.array([2**(-n) for n in range(3,N)])
Ncs = np.array([2**n for n in range(3,N)])


# Ncs = 256*np.ones(Ncs.shape,dtype=np.int)
dts = 0.001*np.ones(dts.shape)


dxs = langm.Ld[0]/Ncs
relError = np.zeros(len(dxs))

for i,dt,dx,Nc in zip(count(),dts,dxs,Ncs):

 	print("Run %d of %d with (dx,dt)=(%f,%f)"%(i+1,len(dxs),dx,dt))
	langm.dt = dt
	langm.Nc = np.array([Nc,Nc])
	langm.reinit()

	relError[i] = langm.run()
	print("Relative energy error: %e"%(relError[i]))

dps = dxs


order = np.log(relError[-2]/relError[-1])/np.log(dps[-2]/dps[-1])
print("Order of error: %f"%order)

plt.figure()
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.plot(dps,relError,'o-')
plt.grid()
plt.xlabel('Step size')
plt.ylabel('Relative energy error')
plt.show()

# Ncs = np.array([2**n for n in range(2,N)])
# langm.dt = 0.01
# relError = np.zeros(len(Ncs))
#
# for i,Nc in enumerate(Ncs):
#
# 	print("Running %d^2 stepsizes"%Nc)
#
# 	langm.Nc = np.array([Nc,Nc])
# 	langm.reinit()
#
# 	relError[i] = langm.run()
# 	print "Relative energy error: %f%%"%(100*relError[i])
#
# order = np.log(relError[-2]/relError[-1])/np.log(Ncs[-2]/Ncs[-1])
# print "Temporal order: %f"%order
#
# plt.loglog(Ncs,relError,'o-')
# plt.grid()
# plt.xlabel('Temporal step')
# plt.ylabel('Relative energy error')
# plt.show()


print("Finished")
