# -*- coding: utf-8 -*-
"""
This module implements a class to handle boundary conditions in the input file.
"""

# meshyp modules 
from . import BaseMeshItem


class BoundaryCondition(BaseMeshItem):
    """This object represents one boundary condition in the input file."""
    
    def __init__(self, geometry_set, bc_string, format_replacement=None,
            bc_type=None):
        """
        Initialize the object.
        
        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_string: str
            Text that will be displayed in the input file for this boundary
            condition.
        format_replacement: str, list
            Replacement with the str.format() function for bc_string.
        bc_type: mpy.boundary
            Type of the boundary condition (dirichlet or neumann).
        """
        
        BaseMeshItem.__init__(self, is_dat=False)
        self.bc_string = bc_string
        self.bc_type = bc_type
        self.format_replacement = format_replacement  
        self.geometry_set = geometry_set

    def _get_dat(self, **kwargs):
        """
        Add the content of this object to the list of lines.
        
        Args:
        ----
        lines: list(str)
            The contents of this object will be added to the end of lines.
        """
        
        if self.format_replacement:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string
        
        return 'E {} - {}'.format(
            self.geometry_set.n_global,
            dat_string
            )