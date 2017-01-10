import numpy as np
from time import *

N = 10000
L = 300

t0 = time()

p = [0 for x in xrange(L)]
for i in xrange(N):
	p += [float(x)/3 for x in xrange(L)]

t1 = time()

p = np.zeros(L)
for i in xrange(N):
	p += np.linspace(0,float(L)/3,L,endpoint=False)
#	p += np.array([float(x)/3 for x in xrange(L)])

t2 = time()

print "List:",t1-t0
print "Array:",t2-t1
