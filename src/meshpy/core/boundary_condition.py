# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2025
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
"""This module implements a class to represent boundary conditions in
MeshPy."""

from meshpy.core.base_mesh_item import BaseMeshItemFull, BaseMeshItemString
from meshpy.core.conf import mpy
from meshpy.core.container import ContainerBase


class BoundaryConditionBase(BaseMeshItemFull):
    """This is a base object, which represents one boundary condition in the
    input file, e.g. Dirichlet, Neumann, coupling or beam-to-solid."""

    def __init__(self, geometry_set, bc_type=None, **kwargs):
        """Initialize the object.

        Args
        ----
        geometry_set: GeometrySet, int
            Geometry that this boundary condition acts on. An integer can be
            given, in the case a dat file is imported. This integer is only
            temporary and will be replaced with the GeometrySet object.
        bc_type: mpy.bc
            Type of the boundary condition.
        """

        super().__init__(**kwargs)
        self.bc_type = bc_type
        self.geometry_set = geometry_set

    @classmethod
    def from_dat(cls, bc_key, line, **kwargs):
        """This function acts as a factory and creates the correct boundary
        condition object from a line in the dat file.

        The geometry set is passed as integer (0 based index) and will
        be connected after the whole input file is parsed.
        """

        # Split up the input line.
        split = line.split()

        if bc_key in (
            mpy.bc.dirichlet,
            mpy.bc.neumann,
            mpy.bc.locsys,
            mpy.bc.beam_to_solid_surface_meshtying,
            mpy.bc.beam_to_solid_surface_contact,
            mpy.bc.beam_to_solid_volume_meshtying,
        ) or isinstance(bc_key, str):
            # Normal boundary condition (including beam-to-solid conditions).
            from meshpy.four_c.boundary_condition import BoundaryCondition

            return BoundaryCondition(
                int(split[1]) - 1, " ".join(split[2:]), bc_type=bc_key, **kwargs
            )
        elif bc_key is mpy.bc.point_coupling:
            # Coupling condition.
            from .coupling import Coupling

            return Coupling(
                int(split[1]) - 1,
                bc_key,
                " ".join(split[2:]),
                check_overlapping_nodes=False,
                check_at_init=False,
                **kwargs,
            )
        raise ValueError("Got unexpected boundary condition!")


class BoundaryConditionContainer(ContainerBase):
    """A class to group boundary conditions together.

    The key of the dictionary are (bc_type, geometry_type).
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [BaseMeshItemString, BoundaryConditionBase]

        for bc_key in mpy.bc:
            for geometry_key in mpy.geo:
                self[(bc_key, geometry_key)] = []
