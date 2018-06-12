

import numpy as np


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """
    
    def __init__(self, *args, **kwargs):

        # parameters for float conversion
        self.dtype = np.longdouble
        
        # geometry types
        self.point = 'geometry_point'
        self.line = 'geometry_line'
        self.surface = 'geometry_surface'
        self.volume = 'geometry_volume'
        self.geometry = [self.point, self.line, self.surface, self.volume]
        
        # boundary conditions types
        self.dirichlet = 'boundary_condition_dirichlet'
        self.neumann = 'boundary_condition_neumann'
        self.boundary_condition = [self.dirichlet, self.neumann]
        
        
        # node set names
        self.geo_dat_names = [
            'DNODE-NODE TOPOLOGY',
            'DLINE-NODE TOPOLOGY',
            'DSURF-NODE TOPOLOGY',
            'DVOL-NODE TOPOLOGY'
            ]
        
        # node set names
        self.geo_set_names = [
            'DNODE',
            'DLINE',
            'DSURFACE',
            'DVOLUME'
            ]
        
        # names for the sections with boundary conditions
        self.bc_dat_name = [
                [
                    'DESIGN POINT DIRICH CONDITIONS',
                    'DESIGN LINE DIRICH CONDITIONS',
                    'DESIGN SURF DIRICH CONDITIONS',
                    'DESIGN VOL DIRICH CONDITIONS'
                ],
                [
                    'DESIGN POINT NEUMANN CONDITIONS',
                    'DESIGN LINE DIRICH NEUMANN',
                    'DESIGN SURF DIRICH NEUMANN',
                    'DESIGN VOL DIRICH NEUMANN'
                ]
            ]
        
        # names for the sections with boundary conditions
        self.bc_name = [
            'DPOINT',
            'DLINE',
            'DSURF',
            'DVOL'
            ]
        


mpy = MeshPy()