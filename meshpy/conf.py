

import numpy as np


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """
    
    def __init__(self, *args, **kwargs):

        # Value for epsilon when checking if two results are equal.
        self.eps_vec = 1e-10
        
        # Geometry types
        self.point = 'geometry_point'
        self.line = 'geometry_line'
        self.surface = 'geometry_surface'
        self.volume = 'geometry_volume'
        self.geometry = [self.point, self.line, self.surface, self.volume]
        
        # Boundary conditions types
        self.dirichlet = 'boundary_condition_dirichlet'
        self.neumann = 'boundary_condition_neumann'
        self.boundary_condition = [self.dirichlet, self.neumann]
        
mpy = MeshPy()