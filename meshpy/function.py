# -*- coding: utf-8 -*-
"""
This module implements a basic class to manage functions in the baci input file.
"""

# Meshpy modules.
from . import BaseMeshItem


class Function(BaseMeshItem):
    """Holds information for a function."""

    def __init__(self, data):
        BaseMeshItem.__init__(self, data=data, is_dat=False)

    def __str__(self):
        """
        Return the global index for this function. This is usually used then the
        function is called with the str.format() function.
        """
        if self.n_global:
            return str(self.n_global)
        else:
            raise IndexError('The function does not have a global index!')