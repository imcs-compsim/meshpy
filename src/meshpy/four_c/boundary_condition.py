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
"""This module implements a class to handle boundary conditions in 4C."""

import warnings

from meshpy.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from meshpy.core.conf import mpy as _mpy
from meshpy.core.geometry_set import GeometrySet as _GeometrySet
from meshpy.utils.nodes import find_close_nodes as _find_close_nodes


class BoundaryCondition(_BoundaryConditionBase):
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

        _BoundaryConditionBase.__init__(self, geometry_set, bc_type=bc_type, **kwargs)
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

        if self.double_nodes is _mpy.double_nodes.keep:
            return

        if (
            self.bc_type == _mpy.bc.neumann
            and self.geometry_set.geometry_type == _mpy.geo.point
        ):
            my_nodes = self.geometry_set.get_points()
            partners = _find_close_nodes(my_nodes)
            # Create a list with nodes that will not be kept in the set.
            double_node_list = []
            for node_list in partners:
                for i, node in enumerate(node_list):
                    if i > 0:
                        double_node_list.append(node)
            if (
                len(double_node_list) > 0
                and self.double_nodes is _mpy.double_nodes.remove
            ):
                # Create the a new geometry set with the unique nodes.
                self.geometry_set = _GeometrySet(
                    [node for node in my_nodes if (node not in double_node_list)]
                )
            elif len(double_node_list) > 0:
                warnings.warn(
                    "There are overlapping nodes in this point Neumann boundary, and it is not "
                    "specified on how to handle them!"
                )
