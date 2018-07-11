# -*- coding: utf-8 -*-
"""
This module implements the class that will be used as the base for all items
that are in a mesh.
"""


class BaseMeshItem(object):
    """Base class for all objects that are related to a mesh."""

    def __init__(self, data=None, is_dat=True, comments=None):
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

        # Comments regarding this Mesh Item.
        if comments is None:
            self.comments = []
        else:
            self.comments = comments

    def get_vtk(self, vtk_writer):
        """Add representation of this object to a vtk_writer."""
        pass

    def get_dat_lines(self, **kwargs):
        """
        Return the content of this object as a list. If comments exist, also add
        those.
        """

        # Get data of object.
        data = self._get_dat(**kwargs)
        if isinstance(data, str):
            data = [data]
        else:
            data = data

        # Get comments if given.
        return_list = []
        if not len(self.comments) == 0:
            return_list.extend(self.comments)

        # Return final list.
        return_list.extend(data)
        return return_list

    def _get_dat(self, **kwargs):
        """Return the content of this object as either a list or a str."""
        return self.data
