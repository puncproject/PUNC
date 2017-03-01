from punc import *

class TestRandomPoints:
	def test_length(self):
		N = 100
		points = randomPoints(lambda x: x[0],[1,1],N)
		assert len(points)==N
