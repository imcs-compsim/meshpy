# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This module implements a class to represent boundary conditions in
MeshPy."""

from meshpy.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from meshpy.core.conf import mpy as _mpy
from meshpy.core.container import ContainerBase as _ContainerBase


class BoundaryConditionBase(_BaseMeshItem):
    """This is a base object, which represents one boundary condition in the
    input file, e.g. Dirichlet, Neumann, coupling or beam-to-solid."""

    def __init__(self, geometry_set, bc_type=None, **kwargs):
        """Initialize the object.

        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_type: mpy.bc
            Type of the boundary condition.
        """

        super().__init__(**kwargs)
        self.bc_type = bc_type
        self.geometry_set = geometry_set

    @classmethod
    def from_dict(cls, geometry_set, bc_key, bc_dict):
        """This function acts as a factory and creates the correct boundary
        condition object from a dictionary parsed from an input file."""

        del bc_dict["E"]

        if bc_key in (
            _mpy.bc.dirichlet,
            _mpy.bc.neumann,
            _mpy.bc.locsys,
            _mpy.bc.beam_to_solid_surface_meshtying,
            _mpy.bc.beam_to_solid_surface_contact,
            _mpy.bc.beam_to_solid_volume_meshtying,
        ) or isinstance(bc_key, str):
            # Normal boundary condition (including beam-to-solid conditions).
            from meshpy.four_c.boundary_condition import (
                BoundaryCondition as _BoundaryCondition,
            )

            boundary_condition = _BoundaryCondition(
                geometry_set, bc_dict, bc_type=bc_key
            )
        elif bc_key is _mpy.bc.point_coupling:
            # Coupling condition.
            from meshpy.core.coupling import Coupling as _Coupling

            boundary_condition = _Coupling(
                geometry_set,
                bc_key,
                bc_dict,
                check_overlapping_nodes=False,
                check_at_init=False,
            )
        else:
            raise ValueError("Got unexpected boundary condition!")

        boundary_condition.check()
        return boundary_condition


class BoundaryConditionContainer(_ContainerBase):
    """A class to group boundary conditions together.

    The key of the dictionary are (bc_type, geometry_type).
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [BoundaryConditionBase]

        for bc_key in _mpy.bc:
            for geometry_key in _mpy.geo:
                self[(bc_key, geometry_key)] = []
