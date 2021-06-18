# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
This module implements a basic class to manage functions in the baci input
file.
"""

# Meshpy modules.
from .base_mesh_item import BaseMeshItem


class Function(BaseMeshItem):
    """Holds information for a function."""

    def __init__(self, data):
        super().__init__(data=data, is_dat=False)

    def __deepcopy__(self, memo):
        """
        When deepcopy is called on a mesh, we do not want the same functions to
        be copied, as this will result in multiple equal functions in the input
        file.
        """

        # Add this object to the memo dictionary.
        memo[id(self)] = self

        # Return this object again, as no copy should be created.
        return self

    def __str__(self):
        """
        Return the global index for this function. This is usually used then
        the function is called with the str.format() function.
        """
        if self.n_global:
            return str(self.n_global)
        else:
            raise IndexError('The function does not have a global index!')
