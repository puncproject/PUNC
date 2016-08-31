
from time import *

t0 = time()

for i in xrange(100):
	a = range(10000)
	for i in xrange(1000):
		a.pop(10)



t1 = time()

for i in xrange(100):
	a = range(10000)
	for i in xrange(1000):
		a[10] = a.pop()

t2 = time()

print "Default:",t1-t0
print "Fill-in:",t2-t1
