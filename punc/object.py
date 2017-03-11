class Object:
    """This class is used to find whether a particle is inside an object or not.
    If the particle is inside the object it returns 'True' and adds the charge
    of the particle to the accumulated charge of the object.
    """
    def __init__(self, geometry):
        self.geometry = geometry
        self.inside = False
        self.charge = 0.0

    def is_inside(self, p, q):
        if self.geometry(p):
            self.inside = True
            self.charge += q
        else:
            self.inside = False
