# Testing whether it's actually beneficial to allocate small numpy arrays
# in an object versus doing it locally.

from __future__ import print_function, division

import numpy as np
import time

class Test():
	def __init__(self):
		self.arr = np.zeros((3,3))

	def preAlloc(self, N):
		x = 0
		for i in xrange(N):
			self.arr = i + np.array([[0,1,2],[3,4,5],[6,7,8]])
			x += sum(sum(self.arr))
		return x

	def nonAlloc(self, N):
		x = 0
		for i in xrange(N):
			arr = i + np.array([[0,1,2],[3,4,5],[6,7,8]])
			x += sum(sum(arr))
		return x

test = Test()

# Just to get the program started. Omitting this disadvantages whichever
# method is called first.
test.preAlloc(1000000)

print("TEST WHERE ONLY ONE LOCAL ALLOCATION IS NECESSARY")

t = time.time()
print(test.nonAlloc(1000000))
print("Non-allocated:",time.time()-t,"s")

t = time.time()
print(test.preAlloc(1000000))
print("Pre-allocated:",time.time()-t,"s")

print("TEST WHERE MANY LOCAL ALLOCATION IS NECESSARY")

t = time.time()
x = 0
for i in xrange(1000):
	x += test.nonAlloc(1000)
print(x)
print("Non-allocated:",time.time()-t,"s")

t = time.time()
x = 0
for i in xrange(1000):
	x += test.preAlloc(1000)
print(x)
print("Pre-allocated:",time.time()-t,"s")

# Requiring many allocations, like in the latter case, is the most realistic
# use case. However, using local allocations ran slightly faster even when
# requiring many allocations.
