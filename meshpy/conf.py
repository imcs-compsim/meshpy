

import numpy as np


class MeshPy(object):
    """
    A global object that stores options for the whole meshpy application.
    """
    
    def __init__(self, *args, **kwargs):

        # parameters for float conversion
        self.dtype = np.longdouble
        
        # geometry types
        self.geo_n = 4
        self.geo_point = 0
        self.geo_line = 1
        self.geo_surf = 2
        self.geo_vol = 3
        self.geo = [self.geo_point, self.geo_line, self.geo_surf, self.geo_vol]
        
        # boundary conditions types
        self.bc_n = 2
        self.bc_diri = 0
        self.bc_neum = 1
        self.bc = [self.bc_diri, self.bc_neum]
        
        
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