# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This module implements containers to manage boundary conditions and geometry
sets in one object.
"""

# Python modules.
from _collections import OrderedDict

# Meshpy modules.
from .conf import mpy
from .geometry_set import GeometrySetBase


class GeometryName(OrderedDict):
    """
    Group node geometry sets together. This is mainly used for export from mesh
    functions. The sets can be accessed by a unique name. There is no
    distinction between different types of geometry, every name can only be
    used once -> use meaningful names.
    OrderedDict is used as base class so that the test cases can compare the
    output string without special implementation (this should not cost much
    performance).
    """

    def __setitem__(self, key, value):
        """Set a geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError("Expected string, got {}!".format(type(key)))
        elif isinstance(value, GeometrySetBase):
            super().__setitem__(key, value)
        else:
            raise NotImplementedError("GeometryName can only store GeometrySets")


class BoundaryConditionContainer(OrderedDict):
    """
    A class to group boundary conditions together. The key of the dictionary
    are (bc_type, geometry_type).
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        for bc_key in mpy.bc:
            for geometry_key in mpy.geo:
                self[(bc_key, geometry_key)] = []

    def append(self, bc_key, geometry_key, bc):
        """
        Append boundary condition to the container.

        Args
        ----
        bc_key:
            Boundary specific key for the boundary condition.
        geometry_key: mpy.geo
            Geometry type of the boundary condition.
        bc: BoundaryCondition
            The boundary condition to be added to this container.
            If the condition is already in this container, an error
            will be raised.
        """
        if (bc_key, geometry_key) not in self.keys():
            self[(bc_key, geometry_key)] = []
        else:
            if bc in self[(bc_key, geometry_key)]:
                raise ValueError("The boundary condition is already in this mesh!")
        self[(bc_key, geometry_key)].append(bc)


class GeometrySetContainer(OrderedDict):
    """
    A class to group geometry sets together with the key being the geometry
    type.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        for geometry_key in mpy.geo:
            self[geometry_key] = []

    def copy(self):
        """
        When creating a copy of this object, all lists in this object will be
        copied also.
        """

        # Create a new geometry set container.
        copy = GeometrySetContainer()

        # Add a copy of every list from this container to the new one.
        for geometry_key in mpy.geo:
            copy[geometry_key] = self[geometry_key].copy()

        return copy
