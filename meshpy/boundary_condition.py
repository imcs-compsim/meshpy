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
"""This module implements a class to handle boundary conditions in the input
file."""

import warnings

from .base_mesh_item import BaseMeshItemFull
from .conf import mpy
from .geometry_set import GeometrySet
from .utility import find_close_nodes


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


class BoundaryCondition(BoundaryConditionBase):
    """This object represents a Dirichlet, Neumann or beam-to-solid boundary
    condition."""

    def __init__(
        self,
        geometry_set,
        bc_string,
        *,
        format_replacement=None,
        bc_type=None,
        double_nodes=None,
        **kwargs,
    ):
        """Initialize the object.

        Args
        ----
        geometry_set: GeometrySet
            Geometry that this boundary condition acts on.
        bc_string: str
            Text that will be displayed in the input file for this boundary
            condition.
        format_replacement: str, list
            Replacement with the str.format() function for bc_string.
        bc_type: mpy.boundary
            Type of the boundary condition.
        double_nodes: mpy.double_nodes
            Depending on this parameter, it will be checked if point Neumann
            conditions do contain nodes at the same spatial positions.
        """

        BoundaryConditionBase.__init__(self, geometry_set, bc_type=bc_type, **kwargs)
        self.bc_string = bc_string
        self.format_replacement = format_replacement
        self.double_nodes = double_nodes

        # Check the parameters for this object.
        self.check()

    def _get_dat(self):
        """Add the content of this object to the list of lines.

        Args:
        ----
        lines: list(str)
            The contents of this object will be added to the end of lines.
        """

        if self.format_replacement is not None:
            dat_string = self.bc_string.format(*self.format_replacement)
        else:
            dat_string = self.bc_string

        return f"E {self.geometry_set.i_global} {dat_string}"

    def check(self):
        """Check for point Neumann boundaries that there is not a double Node
        in the set."""

        if isinstance(self.geometry_set, int):
            # In the case of solid imports this is a integer at initialization.
            return

        if self.double_nodes is mpy.double_nodes.keep:
            return

        if (
            self.bc_type == mpy.bc.neumann
            and self.geometry_set.geometry_type == mpy.geo.point
        ):
            my_nodes = self.geometry_set.get_points()
            partners = find_close_nodes(my_nodes)
            # Create a list with nodes that will not be kept in the set.
            double_node_list = []
            for node_list in partners:
                for i, node in enumerate(node_list):
                    if i > 0:
                        double_node_list.append(node)
            if (
                len(double_node_list) > 0
                and self.double_nodes is mpy.double_nodes.remove
            ):
                # Create the a new geometry set with the unique nodes.
                self.geometry_set = GeometrySet(
                    [node for node in my_nodes if (node not in double_node_list)]
                )
            elif len(double_node_list) > 0:
                warnings.warn(
                    "There are overlapping nodes in this point Neumann boundary, and it is not "
                    "specified on how to handle them!"
                )
