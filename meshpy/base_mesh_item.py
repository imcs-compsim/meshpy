# -*- coding: utf-8 -*-
"""
This module implements the class that will be used as the base for all items
that are in a mesh.
"""

class BaseMeshItem(object):
    """Base class for all objects that are related to a mesh."""
    
    def __init__(self, data=None, is_dat=True):
        """
        Create the object
        
        Args
        ----
        data: str, list(str)
            Data for this object.
        is_dat: bool
            If the object is imported from a *.dat file
        """
        
        self.data = data
        self.is_dat = is_dat
        
        # Overall index of this item in the mesh. 
        self.n_global = None
    
    def add_to_dat_lines(self, lines, **kwargs):
        """
        Add the content of this object to the list of lines.
        
        Args:
        ----
        lines: list(str)
            The contents of this object will be added to the end of lines.
        """
        
        if isinstance(self.data, str):
            lines.append(self.data)
        elif isinstance(self.data, list):
            lines.extend(self.data)
        else:
            raise TypeError('self.data is neither str nor list. '
                + 'Got {}!'.format(type(self.data)))

    def get_dat_line(self):
        """ By default return the data string. """
        if isinstance(self.data, str):
            return self.data
        else:
            raise TypeError('Expected string, got {}!'.format(type(self.data)))
    
    
    def get_dat_lines(self):
        """ By default return the data list. """
        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, str):
            return [self.data]
        else:
            raise TypeError('Expected list, got {}!'.format(type(self.data)))
