# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
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
from .base_mesh_item import BaseMeshItemString
from .boundary_condition import BoundaryConditionBase


class GeometryName(OrderedDict):
    """
    Group node geometry sets together. This is mainly used for export from mesh
    functions. The sets can be accessed by a unique name. There is no
    distinction between different types of geometry, every name can only be
    used once -> use meaningful names.
    """

    def __setitem__(self, key, value):
        """Set a geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError(f"Expected string, got {type(key)}!")
        if isinstance(value, GeometrySetBase):
            super().__setitem__(key, value)
        else:
            raise NotImplementedError("GeometryName can only store GeometrySets")


class ContainerBase(OrderedDict):
    """A base class for containers to be used in MeshPy"""

    def append(self, key, item):
        """Append item to this container and check if the item is already in the list
        corresponding to key."""

        type_ok = False
        for item_type in self.item_types:
            if isinstance(item, item_type):
                type_ok = True
                break
        if not type_ok:
            raise TypeError(
                f"You tried to add an item of type {type(item)}, but only types derived "
                + f"from {self.item_types} can be added"
            )
        if key not in self.keys():
            self[key] = []
        else:
            if item in self[key]:
                raise ValueError("The item is already in this container!")
        self[key].append(item)

    def extend(self, container):
        """Add all items of another container to this container"""

        if not isinstance(container, self.__class__):
            raise TypeError(
                f"Only containers of type {self.__class__} can be merged here, you tried "
                + f"add {type(container)}"
            )
        for key, items in container.items():
            for item in items:
                self.append(key, item)


class BoundaryConditionContainer(ContainerBase):
    """
    A class to group boundary conditions together. The key of the dictionary
    are (bc_type, geometry_type).
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [BaseMeshItemString, BoundaryConditionBase]

        for bc_key in mpy.bc:
            for geometry_key in mpy.geo:
                self[(bc_key, geometry_key)] = []


class GeometrySetContainer(ContainerBase):
    """
    A class to group geometry sets together with the key being the geometry
    type.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [BaseMeshItemString, GeometrySetBase]

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
