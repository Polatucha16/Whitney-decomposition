# Copyright (c) <2022>, <Polatucha16>

class Ball:
    def __init__(self, center, radius):
        self.cen = center    # (np.array shape=(1, dim))
        self.rad = radius    # float
        
        
class Plane:
    def __init__(self, center, orientation):
        self.cen = center       # (np.array shape=(1, 3))
        self.ori = orientation  # ortogonal (np.array shape=(3, 3))
        
    def span(self):
        return self.ori[:-1]
    
    def perp(self):
        return self.ori[-1]