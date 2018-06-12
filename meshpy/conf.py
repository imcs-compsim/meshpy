

import numpy as np


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """
    
    def __init__(self, *args, **kwargs):
        print(1)
        
        
        # parameters for float conversion
        self.dtype = np.longdouble


mpy = MeshPy()