# -*- coding: utf-8 -*-
"""
This module implements containers to manage boundary conditions and geometry
sets in one object.
"""

# Python modules.
from _collections import OrderedDict

# Meshpy modules.
from . import mpy, GeometrySet


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
        """Set an geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError('Expected string, got {}!'.format(type(key)))
        elif isinstance(value, GeometrySet):
            OrderedDict.__setitem__(self, key, value)
        else:
            raise NotImplementedError('TODO: This case needs to be '
                + 'implemented')
            OrderedDict.__setitem__(self, key, GeometrySet(nodes=value))


class BoundaryConditionContainer(OrderedDict):
    """
    A class to group boundary conditions together. The key of the dicitonary
    are (bc_type, geometry_type).
    """
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

        for bc_key in mpy.bc:
            for geometry_key in mpy.geo:
                self[(bc_key, geometry_key)] = []


class GeometrySetContainer(OrderedDict):
    """
    A class to group geometry sets together with the key being the geometry
    type.
    """
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

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
