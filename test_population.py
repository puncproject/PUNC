from punc import *
from mesh import *

class TestRandomPoints(object):
    def test_length(self):
        N = 100
        points = random_points(lambda x: x[0],[1,1],N)
        assert len(points)==N
