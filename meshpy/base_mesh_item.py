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

    def get_dat_lines(self, **kwargs):
        """Return the content of this object as a list."""

        data = self._get_dat(**kwargs)
        if isinstance(data, str):
            return [data]
        else:
            return data

    def _get_dat(self, **kwargs):
        """Return the content of this object as either a list or a str."""
        return self.data